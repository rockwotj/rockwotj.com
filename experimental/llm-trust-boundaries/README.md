# LLM Trust Boundaries Experiment

Training and evaluation code for the blog post [What if LLMs Could See Trust Boundaries?](https://rockwotj.com/blog/llm-trust-boundaries)

## Overview

This experiment fine-tunes Gemma 3 1B with LoRA to test whether adding special tokens for untrusted content improves prompt injection resistance. Three models are compared:

- **Baseline** — trained on general chat data only (ShareGPT)
- **Unstructured** — trained on chat + injection examples, with context inlined into user messages
- **Structured** — trained on chat + injection examples, with context wrapped in `<start_of_context>` / `<end_of_context>` special tokens

## Scripts

| Script | Description |
|---|---|
| `generate_injection_dataset.py` | Generate synthetic prompt injection examples using Gemini |
| `split_dataset.py` | Split datasets into train/eval sets (50 eval, ~9K+ train) |
| `train_gemma3.py` | Fine-tune Gemma 3 1B with LoRA using Unsloth |
| `evaluate_models.py` | Run inference, judge with Gemini, and print results |

## Usage

Requires a GPU (tested on NVIDIA L4 with 23GB VRAM).

```bash
# Install dependencies
uv venv --python 3.11
uv pip install -e .

# Split datasets
python split_dataset.py

# Train all three models (~3 hours on L4)
python train_gemma3.py all

# Evaluate
python evaluate_models.py generate   # GPU inference
python evaluate_models.py judge      # Gemini LLM judge (requires gcloud auth)
python evaluate_models.py summary    # Print results table
```

## Data Files

| File | Description |
|---|---|
| `training_dataset_structured.parquet` | Full dataset with context tokens |
| `training_dataset_unstructured.parquet` | Full dataset with context inlined |
| `train_*.parquet` | Training splits |
| `eval_*.parquet` | Evaluation splits (50 examples) |
| `eval_responses.parquet` | Raw model responses from evaluation |
| `eval_results.parquet` | Judged results with injection/quality scores |
