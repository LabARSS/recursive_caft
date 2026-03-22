"""
Preprocess distillation Branch B data to flat MMLU format.

Reads raw distillation parquet (nested input/output schema),
filters out eval questions and answer-leaked rows,
converts to flat MMLU schema compatible with MMLUReasoningResponseDataset.

Usage:
    uv run python src/experiments/distill/train_branches/prepare_cleaned_b_data.py
"""

import re
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[4]

ANSWER_LEAK_RE = re.compile(
    "|".join([
        r"\bcorrect answer\b", r"\bthe answer is\b", r"\banswer is\b",
        r"\banswer:\b", r"\bcorrect option\b", r"\bcorrect choice\b",
        r"\b[a-j]\s+is\s+correct\b", r"\[\[\s*[a-jA-J]\s*\]\]",
    ]),
    flags=re.IGNORECASE,
)


def collect_eval_question_ids(eval_split_dir: str, groups: int) -> set[str]:
    split_root = PROJECT_ROOT / eval_split_dir
    question_ids: set[str] = set()
    for g in range(groups):
        path = split_root / f"group{g}_test.parquet"
        rows = pq.read_table(path, columns=["question_id"]).to_pylist()
        question_ids.update(str(r["question_id"]) for r in rows)
    return question_ids


def main():
    eval_split_dir = "data/out/splits/single_token_entropy/mmlu/qwen_3b"
    eval_groups = 6

    eval_ids = collect_eval_question_ids(eval_split_dir, eval_groups)
    print(f"Eval question IDs to exclude: {len(eval_ids)}")

    for prompt_id in [1, 2, 3]:
        raw_path = PROJECT_ROOT / f"data/out/distillation/mmlu_synth_gptoss_b_t0_8_cleaned_32b_prompt{prompt_id}.parquet"
        if not raw_path.exists():
            print(f"Skipping prompt {prompt_id}: not found")
            continue

        df = pd.read_parquet(raw_path)
        total = len(df)

        rows = []
        for _, row in df.iterrows():
            inp = row["input"]
            out = row["output"]
            qid = str(inp["question_id"])

            if qid in eval_ids:
                continue

            thinking = str(out.get("thinking") or "").strip()
            if not thinking or ANSWER_LEAK_RE.search(thinking):
                continue

            opts_dict = inp["options"]
            opts_list = [opts_dict[k] for k in sorted(opts_dict.keys())]

            rows.append({
                "question": inp["question"],
                "options": str(opts_list),
                "answer": inp["gold"],
                "thinking": thinking,
                "base_cluster": inp.get("subject", ""),
                "question_id": inp["question_id"],
            })

        out_df = pd.DataFrame(rows)
        out_path = PROJECT_ROOT / f"data/out/distillation/mmlu_branch_b_cleaned_prompt{prompt_id}_prepared.parquet"
        out_df.to_parquet(out_path, index=False)

        filtered = total - len(out_df)
        print(f"Prompt {prompt_id}: {total} -> {len(out_df)} rows ({filtered} filtered, {filtered/total*100:.1f}%)")


if __name__ == "__main__":
    main()
