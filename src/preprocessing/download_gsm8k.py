"""
Download GSM8K from HuggingFace and save as parquet
with a question_id column added.

Usage:
  python src/preprocessing/download_gsm8k.py
"""

import pandas as pd
from datasets import load_dataset

DATASET_ID = "openai/gsm8k"
DATASET_CONFIG = "main"
SPLIT = "test"
OUTPUT_PATH = "data/source/gsm8k/gsm8k_test.parquet"


def extract_numeric_answer(answer_text: str) -> str:
    """Extract the final numeric answer after '####'."""
    return answer_text.split("####")[-1].strip()


def main():
    ds = load_dataset(DATASET_ID, DATASET_CONFIG, split=SPLIT)
    print(f"Downloaded {len(ds)} rows from {DATASET_ID}/{DATASET_CONFIG}")

    rows = []
    for idx, example in enumerate(ds):
        rows.append(
            {
                "question": example["question"],
                "answer": extract_numeric_answer(example["answer"]),
                "answer_w_steps": example["answer"],
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
