"""
Shared helpers for the v0 base-model embedding-init runs.

flatten_distillation_parquet: the source distillation parquet under
data/out/distillation/ has nested {input, output} columns. The MMLU
dataset classes expect flat columns (question_id, question, options,
answer, base_cluster, thinking). This rewrites the source parquet into
a flat schema the existing dataset code can consume, caching the result
on disk.
"""

from pathlib import Path

import pandas as pd

from core.utils.logger import logger


def flatten_distillation_parquet(src: Path, dst: Path) -> Path:
    if dst.exists():
        logger.info(f"Flattened parquet already at {dst}, skipping.")
        return dst

    dst.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(src)

    flat = pd.DataFrame(
        {
            "question_id": df["input"].apply(lambda r: str(r["question_id"])),
            "question": df["input"].apply(lambda r: r["question"]),
            "options": df["input"].apply(lambda r: str(list(r["options"].values()))),
            "answer": df["input"].apply(lambda r: str(r["gold"]).lower()),
            "base_cluster": df["input"].apply(lambda r: r["subject"]),
            "thinking": df["output"].apply(lambda r: r["thinking"]),
        }
    )

    flat = flat[flat["thinking"].notna() & (flat["thinking"].str.len() > 0)]
    flat.to_parquet(dst, index=False)
    logger.info(f"Wrote flattened parquet to {dst} ({len(flat)} rows)")
    return dst
