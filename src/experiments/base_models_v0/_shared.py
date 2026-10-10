"""
Shared helpers for the v0 base-model embedding-init runs.

flatten_distillation_parquet: the source distillation parquet under
data/out/distillation/ has nested {input, output} columns. The MMLU
dataset classes expect flat columns (question_id, question, options,
answer, base_cluster, distill_reasoning, distill_answer). This rewrites
the source parquet into a flat schema the existing dataset code can
consume, caching the result on disk. The teacher reasoning trace lands in
distill_reasoning and the teacher's answer in distill_answer (gold stays
in answer), matching what MMLUReasoningResponseDataset reads.

Optional thinking-trace truncation: v0 training only needs the embedding
rows for <think>/</think> to see the tokens in realistic context. Long
reasoning chains dominate peak memory (cross-entropy over [tokens, vocab]
fp32 logits) without adding signal proportional to their length. Truncate
the distill_reasoning column at a char cap, trimmed back to the last
whitespace so we don't cut mid-word just before </think>.
"""

from pathlib import Path

import pandas as pd

from core.utils.logger import logger


def flatten_distillation_parquet(
    src: Path,
    dst: Path,
    max_thinking_chars: int | None = None,
) -> Path:
    if dst.exists():
        logger.info(f"Flattened parquet already at {dst}, skipping.")
        return dst

    dst.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(src)

    thinking = df["output"].apply(lambda r: r["thinking"])
    if max_thinking_chars is not None:
        thinking = thinking.apply(lambda t: _truncate_to_whitespace(t, max_thinking_chars))

    # Guard None so a missing answer becomes "" (and is dropped below) rather
    # than the string "none" slipping past the non-empty filter.
    distill_answer = df["output"].apply(
        lambda r: "" if r.get("answer") is None else str(r["answer"]).strip().lower()
    )

    flat = pd.DataFrame(
        {
            "question_id": df["input"].apply(lambda r: str(r["question_id"])),
            "question": df["input"].apply(lambda r: r["question"]),
            "options": df["input"].apply(lambda r: str(list(r["options"].values()))),
            "answer": df["input"].apply(lambda r: str(r["gold"]).lower()),
            "base_cluster": df["input"].apply(lambda r: r["subject"]),
            "distill_reasoning": thinking,
            "distill_answer": distill_answer,
        }
    )

    flat = flat[
        flat["distill_reasoning"].notna()
        & (flat["distill_reasoning"].str.len() > 0)
        & (flat["distill_answer"].str.len() > 0)
    ]
    flat.to_parquet(dst, index=False)
    cap_note = f" (thinking capped at {max_thinking_chars} chars)" if max_thinking_chars else ""
    logger.info(f"Wrote flattened parquet to {dst} ({len(flat)} rows){cap_note}")
    return dst


def _truncate_to_whitespace(text: object, max_chars: int) -> str:
    if not isinstance(text, str):
        return ""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    last_ws = max(cut.rfind(" "), cut.rfind("\n"), cut.rfind("\t"))
    if last_ws > 0:
        cut = cut[:last_ws]
    return cut.rstrip()
