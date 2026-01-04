#!/usr/bin/env python3
"""
Evaluate trained Gemma 3 1B models on prompt injection resistance.

Simple evaluation that shows model responses to injection attempts.

Usage:
  python evaluate_models.py baseline           # Evaluate specific model
  python evaluate_models.py structured
"""

import sys
import pandas as pd
from pathlib import Path
from typing import Dict, List
from unsloth import FastLanguageModel
import torch

# Model paths
MODEL_PATHS = {
    "baseline": "./models/gemma3-1b-baseline-merged",
    "unstructured": "./models/gemma3-1b-unstructured-merged",
    "structured": "./models/gemma3-1b-structured-merged",
}

# Chat templates
CHATML_TEMPLATE = """{% for message in messages %}{% if message['from'] == 'human' %}<|im_start|>user
{{ message['value'] }}<|im_end|>
{% elif message['from'] == 'gpt' %}<|im_start|>assistant
{{ message['value'] }}<|im_end|>
{% endif %}{% endfor %}<|im_start|>assistant
"""

STRUCTURED_TEMPLATE = """{% for message in messages %}{% if message['from'] == 'human' %}<|im_start|>user
{{ message['value'] }}<|im_end|>
{% elif message['from'] == 'context' %}<start_of_context>{{ message['value'] }}<end_of_context>
{% elif message['from'] == 'gpt' %}<|im_start|>assistant
{{ message['value'] }}<|im_end|>
{% endif %}{% endfor %}<|im_start|>assistant
"""

# Generation config
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7


def load_model(model_name: str):
    """Load a trained model and tokenizer."""

    model_path = MODEL_PATHS[model_name]

    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Train it first with:\n"
            f"  python train_gemma3.py {model_name}"
        )

    print(f"Loading {model_name} model from {model_path}...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=32768,
        dtype=None,
        load_in_4bit=True,
    )

    FastLanguageModel.for_inference(model)

    return model, tokenizer


def format_prompt(messages: List[Dict], use_structured: bool, tokenizer) -> str:
    """Format messages into a prompt using the appropriate template."""

    template = STRUCTURED_TEMPLATE if use_structured else CHATML_TEMPLATE

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        chat_template=template,
    )

    return prompt


def generate_response(model, tokenizer, prompt: str) -> str:
    """Generate a response from the model."""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the new tokens (not the prompt)
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )

    return response.strip()


def evaluate_model(model_name: str, eval_file: str):
    """Evaluate a model on an eval dataset."""

    # Determine if we should use structured template
    use_structured = "structured" in eval_file and model_name == "structured"

    print("\n" + "=" * 70)
    print(f" Evaluating: {model_name.upper()}")
    print(f" Dataset: {eval_file}")
    print(f" Template: {'structured' if use_structured else 'chatml'}")
    print("=" * 70)

    # Load model
    model, tokenizer = load_model(model_name)

    # Load eval dataset
    if not Path(eval_file).exists():
        print(f"\n✗ Error: Evaluation file not found: {eval_file}")
        return

    df = pd.read_parquet(eval_file)
    print(f"\nEvaluating {len(df)} examples...\n")

    # Evaluate each example
    for idx, row in df.iterrows():
        conversation = row["conversations"]

        # Extract only user messages (human + context) for the prompt
        user_messages = [
            msg for msg in conversation if msg["from"] in ["human", "context"]
        ]

        print(f"\n{'=' * 70}")
        print(f"Example {idx + 1}/{len(df)}")
        print(f"{'=' * 70}")

        # Show the input messages
        print("\nInput Messages:")
        for msg in user_messages:
            msg_type = msg["from"].upper()
            content = msg["value"][:200] + ("..." if len(msg["value"]) > 200 else "")
            print(f"  [{msg_type}] {content}")

        # Format prompt and generate
        prompt = format_prompt(user_messages, use_structured, tokenizer)
        response = generate_response(model, tokenizer, prompt)

        print(f"\nModel Response:")
        print(f"  {response}")

        print(f"\n{'=' * 70}")

    # Cleanup
    del model
    del tokenizer
    torch.cuda.empty_cache()


def main():
    if len(sys.argv) < 2:
        print("Usage: python evaluate_models.py [baseline|unstructured|structured]")
        print("\nAvailable models:")
        for name in MODEL_PATHS.keys():
            print(f"  {name}")
        sys.exit(1)

    model_name = sys.argv[1].lower()

    if model_name not in MODEL_PATHS:
        print(f"Error: Unknown model '{model_name}'")
        print(f"Available: {', '.join(MODEL_PATHS.keys())}")
        sys.exit(1)

    # Determine which eval file to use
    if model_name == "structured":
        eval_file = "eval_structured.parquet"
    else:
        eval_file = "eval_unstructured.parquet"

    try:
        evaluate_model(model_name, eval_file)
        print("\n✓ Evaluation complete!")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
