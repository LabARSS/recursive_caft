"""
Download GPQA Extended from HuggingFace and save as parquet
in the project's standard MMLU-compatible format.

Prerequisites:
  - Accept the dataset terms at https://huggingface.co/datasets/Idavidrein/gpqa
  - Run `huggingface-cli login` to authenticate

Usage:
  python scripts/download_gpqa.py
"""

import random
import string

import pandas as pd
from datasets import load_dataset

DATASET_ID = "Idavidrein/gpqa"
DATASET_CONFIG = "gpqa_extended"
SPLIT = "train"
OUTPUT_PATH = "data/source/gpqa.parquet"
SEED = 42

OPTION_LETTERS = list(string.ascii_uppercase)


def main():
    random.seed(SEED)

    ds = load_dataset(DATASET_ID, DATASET_CONFIG, split=SPLIT)
    print(f"Downloaded {len(ds)} rows from {DATASET_ID}/{DATASET_CONFIG}")

    rows = []
    for idx, example in enumerate(ds):
        question = example["Question"]
        correct = example["Correct Answer"]
        incorrect = [
            example["Incorrect Answer 1"],
            example["Incorrect Answer 2"],
            example["Incorrect Answer 3"],
        ]

        options = incorrect.copy()
        answer_index = random.randint(0, len(options))
        options.insert(answer_index, correct)

        rows.append(
            {
                "question": question,
                "options": str(options),
                "answer": OPTION_LETTERS[answer_index],
                "answer_index": answer_index,
                "question_id": str(idx),
            }
        )

    df = pd.DataFrame(rows)
    df.to_parquet(OUTPUT_PATH, index=False)

    print(f"Saved {len(df)} rows to {OUTPUT_PATH}")
    print(f"Columns: {list(df.columns)}")
    print(df.head(2).to_string())


if __name__ == "__main__":
    main()
