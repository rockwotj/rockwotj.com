#!/usr/bin/env python3
"""
Fine-tune Gemma 3 1B on prompt injection resistance datasets.

Trains separate models on:
1. philschmid - Baseline model (ChatML template)
2. unstructured - Unstructured injection training (ChatML template)
3. structured - Structured injection training (context tokens)

Usage:
  python train_gemma3.py philschmid    # Train baseline model
  python train_gemma3.py unstructured  # Train unstructured model
  python train_gemma3.py structured    # Train structured model
  python train_gemma3.py all           # Train all three models sequentially

Uses Unsloth for efficient training.
"""

import sys
import pandas as pd
from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import torch

# Training configuration
MAX_SEQ_LENGTH = 32768
DTYPE = None  # Auto-detect. Use Float16 for Tesla T4, V100, Bfloat16 for Ampere+
LOAD_IN_4BIT = True  # Use 4bit quantization to reduce memory usage

# Model configuration
MODEL_NAME = "unsloth/gemma-3-1b-pt"  # Gemma 3 1B pretrained/base model

# Chat templates using Gemma's native format
GEMMA_TEMPLATE = """{% for message in messages %}{% if message['from'] == 'human' %}<start_of_turn>user
{{ message['value'] }}<end_of_turn>
{% elif message['from'] == 'gpt' %}<start_of_turn>model
{{ message['value'] }}<end_of_turn>
{% endif %}{% endfor %}{% if add_generation_prompt %}<start_of_turn>model
{% endif %}"""

# Custom template with context tokens for structured data
STRUCTURED_TEMPLATE = """{% for message in messages %}{% if message['from'] == 'human' %}<start_of_turn>user
{{ message['value'] }}<end_of_turn>
{% elif message['from'] == 'context' %}<start_of_context>{{ message['value'] }}<end_of_context>
{% elif message['from'] == 'gpt' %}<start_of_turn>model
{{ message['value'] }}<end_of_turn>
{% endif %}{% endfor %}{% if add_generation_prompt %}<start_of_turn>model
{% endif %}"""

# Model configurations
MODEL_CONFIGS = {
    "philschmid": {
        "dataset": "philschmid-guanaco-sharegpt-style.parquet",
        "template": "gemma",
        "output_dir": "./models/gemma3-1b-baseline",
        "description": "Baseline model trained on general chat data",
        "max_steps": 1000,
    },
    "unstructured": {
        "dataset": "train_unstructured.parquet",
        "template": "gemma",
        "output_dir": "./models/gemma3-1b-unstructured",
        "description": "Model trained on prompt injections (unstructured)",
        "max_steps": 500,
    },
    "structured": {
        "dataset": "train_structured.parquet",
        "template": "structured",
        "output_dir": "./models/gemma3-1b-structured",
        "description": "Model trained on prompt injections (structured with context tokens)",
        "max_steps": 500,
    },
}


def load_and_format_dataset(parquet_file: str, template_type: str) -> Dataset:
    """Load parquet file and format conversations for training."""

    print(f"  Loading {parquet_file}...")
    df = pd.read_parquet(parquet_file)

    def format_example(row):
        return {
            "messages": row["conversations"],
            "template": template_type,
        }

    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(format_example, remove_columns=["conversations"])

    print(f"  Loaded {len(dataset)} examples")
    return dataset


def apply_chat_template(example, tokenizer, needs_context_tokens: bool):
    """Apply the appropriate chat template based on the template type."""

    if example["template"] == "structured":
        # Use structured template with context tokens
        formatted = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
            chat_template=STRUCTURED_TEMPLATE,
        )
    else:
        # Use Gemma's native template
        formatted = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
            chat_template=GEMMA_TEMPLATE,
        )

    return {"text": formatted}


def train_model(config_name: str):
    """Train a single model based on configuration."""

    config = MODEL_CONFIGS[config_name]

    print("\n" + "=" * 70)
    print(f" Training: {config_name.upper()}")
    print(f" {config['description']}")
    print("=" * 70)

    # Load model and tokenizer
    print("\n1. Loading Gemma 3 1B base model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
    )

    # Add special tokens
    print("\n2. Adding special tokens...")
    needs_context_tokens = config["template"] == "structured"

    special_tokens = {
        "additional_special_tokens": [
            "<start_of_turn>",
            "<end_of_turn>",
        ]
    }

    # Add context tokens only for structured model
    if needs_context_tokens:
        special_tokens["additional_special_tokens"].extend(
            ["<start_of_context>", "<end_of_context>"]
        )
        print("  Added context tokens: <start_of_context>, <end_of_context>")

    print("  Using Gemma's native tokens: <start_of_turn>, <end_of_turn>")

    tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))

    # Configure model for training with LoRA
    print("\n3. Configuring LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # LoRA rank
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        max_seq_length=MAX_SEQ_LENGTH,
    )

    # Load dataset
    print(f"\n4. Loading dataset...")
    dataset = load_and_format_dataset(config["dataset"], config["template"])
    dataset = dataset.shuffle(seed=42)

    # Apply chat templates
    print("\n5. Applying chat templates...")
    formatted_dataset = dataset.map(
        lambda x: apply_chat_template(x, tokenizer, needs_context_tokens),
        remove_columns=["messages", "template"],
        num_proc=4,
    )

    # Print example
    print("\n6. Example formatted conversation:")
    print("-" * 70)
    example_text = formatted_dataset[0]["text"]
    print(example_text[: min(500, len(example_text))])
    if len(example_text) > 500:
        print("...")
    print("-" * 70)

    # Training arguments
    print("\n7. Configuring training...")
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=min(100, config["max_steps"] // 10),
        max_steps=config["max_steps"],
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        report_to="none",
    )

    # Create trainer
    print("\n8. Initializing trainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=formatted_dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=4,
        packing=False,
        args=training_args,
    )

    # Train
    print("\n9. Starting training...")
    print("=" * 70)
    trainer_stats = trainer.train()

    # Save model
    print(f"\n10. Saving model to {config['output_dir']}...")
    model.save_pretrained(f"{config['output_dir']}-lora")
    tokenizer.save_pretrained(f"{config['output_dir']}-lora")

    # Save merged model
    print(f"\n11. Saving merged 16-bit model...")
    model.save_pretrained_merged(
        f"{config['output_dir']}-merged",
        tokenizer,
        save_method="merged_16bit",
    )

    print("\n" + "=" * 70)
    print(f" ✓ {config_name.upper()} MODEL COMPLETE!")
    print("=" * 70)
    print(f"\nModel saved to:")
    print(f"  - LoRA adapters: {config['output_dir']}-lora")
    print(f"  - Merged 16-bit: {config['output_dir']}-merged")
    print(f"\nTraining stats:")
    print(f"  - Total steps: {trainer_stats.global_step}")
    print(f"  - Training loss: {trainer_stats.training_loss:.4f}")
    print("=" * 70 + "\n")

    return trainer_stats


def main():
    if len(sys.argv) < 2:
        print("Usage: python train_gemma3.py [philschmid|unstructured|structured|all]")
        print("\nAvailable models:")
        for name, config in MODEL_CONFIGS.items():
            print(f"  {name:12} - {config['description']}")
        print(f"  all          - Train all three models sequentially")
        sys.exit(1)

    model_type = sys.argv[1].lower()

    print("=" * 70)
    print(" " * 15 + "Gemma 3 1B Fine-tuning for Prompt Injection")
    print("=" * 70)

    if model_type == "all":
        print("\nTraining all three models sequentially...\n")
        results = {}
        for config_name in ["philschmid", "unstructured", "structured"]:
            stats = train_model(config_name)
            results[config_name] = stats

        print("\n" + "=" * 70)
        print(" " * 20 + "ALL MODELS TRAINED!")
        print("=" * 70)
        print("\nSummary:")
        for name, stats in results.items():
            print(
                f"  {name:12} - Loss: {stats.training_loss:.4f}, Steps: {stats.global_step}"
            )
        print("\nNext steps:")
        print("  - Evaluate models on eval_*.parquet datasets")
        print("  - Compare prompt injection resistance across models")
        print("  - Test with real prompt injection attacks")
        print("=" * 70)

    elif model_type in MODEL_CONFIGS:
        train_model(model_type)

        print("\nNext steps:")
        print(
            f"  - Train other models: {', '.join(k for k in MODEL_CONFIGS.keys() if k != model_type)}"
        )
        print(f"  - Evaluate on eval_{model_type}.parquet")
        print(f"  - Compare against other model variants")

    else:
        print(f"Error: Unknown model type '{model_type}'")
        print(f"Available: {', '.join(MODEL_CONFIGS.keys())}, all")
        sys.exit(1)


if __name__ == "__main__":
    main()
