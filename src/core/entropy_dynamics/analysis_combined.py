"""
Combined Student + Proxy Analysis
===================================

Joins per-chunk entropy from student and proxy runs on the same questions,
computes per-chunk:
  - H_s, H_t (student entropy, proxy entropy)
  - ER = H_s / H_t  (entropy ratio)
  - EG = H_s - H_t  (entropy gap)
  - delta_H_s = H_s(k-1) - H_s(k)  (student entropy drop at this chunk)
  - delta_H_t = H_t(k-1) - H_t(k)  (proxy entropy drop)
  - utility = delta_H_s  (how much this chunk helped the student)

Usage:
  python -m core.entropy_dynamics.analysis_combined \
      --student_results artifacts/.../entropy_dynamics_student_mmlu_synth_gptoss_b_t0_8.parquet \
      --proxy_results   artifacts/.../entropy_dynamics_proxy_mmlu_synth_gptoss_b_t0_8.parquet \
      --out             artifacts/.../combined_analysis/
"""

from __future__ import annotations
import argparse, os
from pathlib import Path
import numpy as np
import pandas as pd


def load_and_tag(path: str, role: str) -> pd.DataFrame:
    """Load a results parquet and prefix entropy/answer columns with role."""
    df = pd.read_parquet(path)
    # Normalize column names from old format
    renames = {}
    for c in df.columns:
        cl = c.lower()
        if cl in ("answer_entropy", "h_s", "entropy"):
            renames[c] = f"H_{role[0]}"  # H_s or H_t
        if cl in ("student_answer", "model_answer"):
            renames[c] = f"answer_{role}"
        if cl in ("student_correct", "model_correct"):
            renames[c] = f"correct_{role}"
        if cl in ("student_label", "model_label"):
            renames[c] = f"label_{role}"
    df = df.rename(columns=renames)
    df["question_id"] = df["question_id"].astype(str)
    df["k"] = df["k"].astype(int)
    return df


def join_student_proxy(
    student_path: str,
    proxy_path: str,
    student_label: str = None,
    proxy_label: str = None,
) -> pd.DataFrame:
    """Join student and proxy results on (question_id, k)."""
    ds = load_and_tag(student_path, "s")
    dt = load_and_tag(proxy_path, "t")

    # Filter by label if specified
    if student_label and "label_s" in ds.columns:
        ds = ds[ds["label_s"] == student_label]
    if proxy_label and "label_t" in dt.columns:
        dt = dt[dt["label_t"] == proxy_label]

    # Keep only the columns we need from each side
    s_cols = ["question_id", "k", "H_s", "answer_s", "correct_s",
              "gold_answer", "num_reasoning_tokens", "total_reasoning_tokens"]
    t_cols = ["question_id", "k", "H_t", "answer_t", "correct_t"]

    s_cols = [c for c in s_cols if c in ds.columns]
    t_cols = [c for c in t_cols if c in dt.columns]

    merged = ds[s_cols].merge(dt[t_cols], on=["question_id", "k"], how="inner")
    print(f"  Joined: {len(merged)} rows ({merged['question_id'].nunique()} questions)")
    return merged


def compute_per_chunk_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add ER, EG, delta_H_s, delta_H_t, utility columns."""
    df = df.copy()

    # Entropy Ratio and Entropy Gap
    eps = 1e-8
    df["ER"] = df["H_s"] / (df["H_t"] + eps)
    df["EG"] = df["H_s"] - df["H_t"]

    # Per-question deltas (entropy drop from previous chunk)
    df = df.sort_values(["question_id", "k"]).reset_index(drop=True)
    df["H_s_prev"] = df.groupby("question_id")["H_s"].shift(1)
    df["H_t_prev"] = df.groupby("question_id")["H_t"].shift(1)
    df["delta_H_s"] = df["H_s_prev"] - df["H_s"]  # positive = entropy dropped
    df["delta_H_t"] = df["H_t_prev"] - df["H_t"]

    # Utility: how much this chunk helped the student become correct
    # Simple version: delta_H_s (entropy drop)
    df["utility_basic"] = df["delta_H_s"]

    # Normalized: student drop relative to proxy drop
    df["utility_normalized"] = df["delta_H_s"] / (df["delta_H_t"].abs() + eps)

    # State-aware: delta_H_s only counts if student became correct
    if "correct_s" in df.columns:
        correct_prev = df.groupby("question_id")["correct_s"].shift(1)
        # Chunk flipped wrong→right
        df["flipped_to_correct"] = (~correct_prev.fillna(False)) & df["correct_s"]
        df["utility_flip"] = df["delta_H_s"].where(df["flipped_to_correct"], 0.0)

    return df


def compute_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-step aggregates: mean H_s, H_t, ER, EG, accuracy."""
    agg = df.groupby("k").agg(
        mean_H_s=("H_s", "mean"),
        mean_H_t=("H_t", "mean"),
        mean_ER=("ER", "mean"),
        mean_EG=("EG", "mean"),
        mean_delta_H_s=("delta_H_s", "mean"),
        mean_delta_H_t=("delta_H_t", "mean"),
        mean_utility=("utility_basic", "mean"),
        accuracy_student=("correct_s", "mean"),
        accuracy_proxy=("correct_t", "mean"),
        n=("question_id", "count"),
    ).reset_index()
    return agg


def write_report(df: pd.DataFrame, summary: pd.DataFrame, out_path: Path):
    """Text report with combined metrics."""
    L = []
    L.append("=" * 80)
    L.append("COMBINED STUDENT + PROXY ENTROPY ANALYSIS")
    L.append("=" * 80)
    L.append(f"\nTotal measurements: {len(df)}")
    L.append(f"Questions: {df['question_id'].nunique()}")
    L.append(f"Steps (k): 0 to {df['k'].max()}")

    # Global stats
    L.append(f"\n{'Metric':<30} {'Mean':>10} {'Std':>10} {'Median':>10}")
    L.append("-" * 60)
    for col, label in [
        ("H_s", "H_s (student entropy)"),
        ("H_t", "H_t (proxy entropy)"),
        ("ER", "ER = H_s / H_t"),
        ("EG", "EG = H_s - H_t"),
        ("delta_H_s", "ΔH_s (student drop/chunk)"),
        ("delta_H_t", "ΔH_t (proxy drop/chunk)"),
        ("utility_basic", "utility (= ΔH_s)"),
    ]:
        if col in df.columns:
            vals = df[col].dropna()
            L.append(f"{label:<30} {vals.mean():>10.4f} {vals.std():>10.4f} {vals.median():>10.4f}")

    # Accuracy comparison
    if "correct_s" in df.columns and "correct_t" in df.columns:
        L.append(f"\n\nAccuracy at step 0 vs final step:")
        for role, col in [("student", "correct_s"), ("proxy", "correct_t")]:
            step0 = df[df["k"] == 0]
            stepF = df[df["k"] == df.groupby("question_id")["k"].transform("max")]
            L.append(f"  {role}: step0={step0[col].mean():.1%}, final={stepF[col].mean():.1%}")

    # Agreement analysis
    if "correct_s" in df.columns and "correct_t" in df.columns:
        L.append(f"\n\nStudent-proxy agreement at each step:")
        L.append(f"  {'k':>5} {'both_correct':>14} {'only_student':>14} {'only_proxy':>12} {'both_wrong':>12}")
        for k_val in sorted(df["k"].unique())[:20]:
            sub = df[df["k"] == k_val]
            bc = (sub["correct_s"] & sub["correct_t"]).mean()
            os_ = (sub["correct_s"] & ~sub["correct_t"]).mean()
            op = (~sub["correct_s"] & sub["correct_t"]).mean()
            bw = (~sub["correct_s"] & ~sub["correct_t"]).mean()
            L.append(f"  {k_val:>5} {bc:>13.1%} {os_:>13.1%} {op:>11.1%} {bw:>11.1%}")

    # Per-chunk utility: which chunks have highest/lowest utility?
    if "utility_basic" in df.columns:
        L.append(f"\n\nMean utility (ΔH_s) by step k (top 15 steps):")
        u = df.groupby("k")["utility_basic"].mean().sort_values(ascending=False)
        for k_val in u.head(15).index:
            L.append(f"  k={k_val:>3}: mean utility = {u[k_val]:+.4f}")

    # Flip analysis
    if "flipped_to_correct" in df.columns:
        L.append(f"\n\nFlip analysis (wrong→right transitions):")
        flips = df[df["flipped_to_correct"] == True]
        L.append(f"  Total flips: {len(flips)}")
        if len(flips):
            L.append(f"  Mean ΔH_s at flip: {flips['delta_H_s'].mean():.4f}")
            L.append(f"  Mean ER at flip: {flips['ER'].mean():.4f}")
            L.append(f"  Mean H_t at flip: {flips['H_t'].mean():.4f}")
            # Compare ER at flip vs non-flip
            non_flips = df[(df["flipped_to_correct"] == False) & (df["k"] > 0)]
            L.append(f"  Mean ER at non-flip: {non_flips['ER'].mean():.4f}")
            L.append(f"  → ER difference: {flips['ER'].mean() - non_flips['ER'].mean():+.4f}")

    # Token-level weight suggestion
    L.append(f"\n\n{'='*80}")
    L.append("TOKEN-LEVEL TRAINING WEIGHT SUGGESTION")
    L.append("=" * 80)
    L.append("""
For Rho-1-style token-level loss masking during SFT:
  weight(chunk_k) = max(0, utility_basic(chunk_k))  [normalized to sum=1]

Chunks where utility > 0 (student entropy dropped) get higher weight.
Chunks where utility <= 0 (student entropy flat or rose) get zero weight.

To apply: during SFT, multiply per-token loss by the weight of the chunk
that token belongs to. The student still sees the full reasoning in context,
but gradient updates come only from high-utility chunks.

Alternative weights based on combined metrics:
  weight_ER(k)  = max(0, ER(k) - 1)       — chunks where student is more uncertain than proxy
  weight_EG(k)  = max(0, EG(k))            — chunks where student entropy exceeds proxy
  weight_flip(k) = 1 if flip happened, 0 otherwise  — only chunks that caused state change
""")

    text = "\n".join(L)
    out_path.write_text(text, encoding="utf-8")
    print(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student_results", required=True)
    parser.add_argument("--proxy_results", required=True)
    parser.add_argument("--student_label", default=None,
                        help="Filter student by label if multiple in file")
    parser.add_argument("--proxy_label", default=None,
                        help="Filter proxy by label if multiple in file")
    parser.add_argument("--out", default="artifacts/entropy_dynamics/combined_analysis")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("[1/4] Joining student and proxy results...")
    df = join_student_proxy(
        args.student_results, args.proxy_results,
        args.student_label, args.proxy_label,
    )

    print("[2/4] Computing per-chunk metrics (ER, EG, utility)...")
    df = compute_per_chunk_metrics(df)
    df.to_parquet(out / "combined_per_chunk.parquet", index=False)

    print("[3/4] Computing summary...")
    summary = compute_summary_table(df)
    summary.to_csv(out / "summary_by_step.csv", index=False)

    print("[4/4] Writing report...")
    write_report(df, summary, out / "REPORT.txt")

    print(f"\n✓ Outputs in {out}/")
    print(f"  combined_per_chunk.parquet — full per-(question, k) data with ER/EG/utility")
    print(f"  summary_by_step.csv       — aggregated per step k")
    print(f"  REPORT.txt                — text report with all metrics")


if __name__ == "__main__":
    main()
