#!/usr/bin/env python3
"""
Split training datasets into training and evaluation sets.
Creates an eval set with 50 samples: 5 without context blocks, 45 with context blocks.
"""

import pandas as pd
import json


def has_context_message(conversations):
    """Check if a conversation has any 'context' messages."""
    if isinstance(conversations, str):
        conversations = json.loads(conversations)

    for msg in conversations:
        if msg.get("from") == "context":
            return True
    return False


def main():
    print("=" * 70)
    print(" " * 20 + "Dataset Splitter")
    print("=" * 70)

    # Load structured dataset
    print("\nLoading structured dataset...")
    struct_df = pd.read_parquet("training_dataset_structured.parquet")
    print(f"  Total examples: {len(struct_df)}")

    # Shuffle with fixed seed for reproducibility
    struct_df = struct_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Separate samples with and without context messages
    print("\nCategorizing samples...")
    struct_df['has_context'] = struct_df['conversations'].apply(has_context_message)

    with_context = struct_df[struct_df['has_context']].copy()
    without_context = struct_df[~struct_df['has_context']].copy()

    print(f"  Samples with context: {len(with_context)}")
    print(f"  Samples without context: {len(without_context)}")

    # Select evaluation samples: 5 without context, 45 with context
    print("\nSelecting evaluation samples...")
    eval_without = without_context.head(5)
    eval_with = with_context.head(45)

    # Combine evaluation samples
    struct_eval = pd.concat([eval_without, eval_with], ignore_index=True)
    struct_eval = struct_eval.drop(columns=['has_context'])

    # Get the original indices before shuffling
    eval_indices = list(struct_eval.index)

    # Remove evaluation samples from training set
    struct_train = struct_df[~struct_df.index.isin(eval_indices)].copy()
    struct_train = struct_train.drop(columns=['has_context'])

    print(f"  Selected {len(eval_without)} without context")
    print(f"  Selected {len(eval_with)} with context")
    print(f"  Total eval samples: {len(struct_eval)}")
    print(f"  Remaining training samples: {len(struct_train)}")

    # Upsample injection examples 4x in structured training set
    # so context-token examples go from ~5% to ~20% of the mix
    struct_train_has_context = struct_train[
        struct_train['conversations'].apply(has_context_message)
    ]
    n_before = len(struct_train)
    struct_train = pd.concat(
        [struct_train] + [struct_train_has_context] * 3,
        ignore_index=True,
    )
    struct_train = struct_train.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"\n  Upsampled structured training set: {n_before} -> {len(struct_train)}")
    print(f"  ({len(struct_train_has_context)} injection examples repeated 4x total)")

    # Save structured datasets
    struct_train.to_parquet("train_structured.parquet", index=False)
    struct_eval.to_parquet("eval_structured.parquet", index=False)

    # Now process unstructured dataset using the SAME indices
    print("\nProcessing unstructured dataset...")
    unstruct_df = pd.read_parquet("training_dataset_unstructured.parquet")

    # Shuffle with the SAME seed to maintain alignment
    unstruct_df = unstruct_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Use the same indices for eval and train splits
    unstruct_eval = unstruct_df.loc[eval_indices].copy()
    unstruct_train = unstruct_df[~unstruct_df.index.isin(eval_indices)].copy()

    # Save unstructured datasets
    unstruct_train.to_parquet("train_unstructured.parquet", index=False)
    unstruct_eval.to_parquet("eval_unstructured.parquet", index=False)

    # Summary
    print("\n" + "=" * 70)
    print(" " * 30 + "SUMMARY")
    print("=" * 70)
    print("\nEvaluation Set Composition:")
    print(f"  • 5 samples without 'context' messages")
    print(f"  • 45 samples with 'context' messages")
    print(f"  • Total: 50 evaluation samples")
    print("\nFiles Created:")
    print(f"  Structured:")
    print(f"    • train_structured.parquet ({len(struct_train)} examples)")
    print(f"    • eval_structured.parquet ({len(struct_eval)} examples)")
    print(f"\n  Unstructured:")
    print(f"    • train_unstructured.parquet ({len(unstruct_train)} examples)")
    print(f"    • eval_unstructured.parquet ({len(unstruct_eval)} examples)")
    print("\nNote: Both eval sets contain the same 50 conversations")
    print("=" * 70)


if __name__ == "__main__":
    main()
