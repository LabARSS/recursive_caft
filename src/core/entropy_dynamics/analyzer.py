from __future__ import annotations

from pathlib import Path

#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#import seaborn as sns


def load_results(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics per (student_label, k)."""
    grouped = df.groupby(["student_label", "k"]).agg(
        mean_entropy=("answer_entropy", "mean"),
        std_entropy=("answer_entropy", "std"),
        median_entropy=("answer_entropy", "median"),
        accuracy=("student_correct", "mean"),
        n_samples=("question_id", "count"),
        mean_reasoning_tokens=("num_reasoning_tokens", "mean"),
    ).reset_index()
    return grouped


def compute_half_life(summary: pd.DataFrame) -> pd.DataFrame:
    """For each student, find the smallest k where entropy falls to midpoint.

    Midpoint = (H_baseline + H_final) / 2
    """
    rows = []
    for label, grp in summary.groupby("student_label"):
        grp = grp.sort_values("k")
        h_baseline = grp[grp["k"] == 0]["mean_entropy"].iloc[0]
        h_final = grp["mean_entropy"].iloc[-1]
        midpoint = (h_baseline + h_final) / 2

        k_star = None
        for _, row in grp.iterrows():
            if row["mean_entropy"] <= midpoint:
                k_star = int(row["k"])
                break

        rows.append({
            "student_label": label,
            "h_baseline": h_baseline,
            "h_final": h_final,
            "midpoint": midpoint,
            "k_star": k_star,
            "k_star_tokens": k_star * int(grp["mean_reasoning_tokens"].iloc[1]) if k_star and k_star > 0 else None,
        })

    return pd.DataFrame(rows)


def compute_per_correctness(df: pd.DataFrame) -> pd.DataFrame:
    """Split by teacher correctness (gold_answer match) to test H3.
    """
    if "teacher_correct" not in df.columns:
        return pd.DataFrame()

    grouped = df.groupby(["student_label", "k", "teacher_correct"]).agg(
        mean_entropy=("answer_entropy", "mean"),
        accuracy=("student_correct", "mean"),
        n_samples=("question_id", "count"),
    ).reset_index()
    return grouped


def plot_entropy_curves(
    summary: pd.DataFrame,
    title: str = "Entropy Reduction Curve",
    save_to: str | Path | None = None,
):
    """Plot mean entropy vs k for each student."""
    sns.set_theme(style="whitegrid", context="paper")

    fig, ax = plt.subplots(figsize=(10, 6))

    for label, grp in summary.groupby("student_label"):
        grp = grp.sort_values("k")
        ax.plot(grp["k"], grp["mean_entropy"], marker="o", markersize=3, label=label)
        ax.fill_between(
            grp["k"],
            grp["mean_entropy"] - grp["std_entropy"],
            grp["mean_entropy"] + grp["std_entropy"],
            alpha=0.15,
        )

    ax.set_xlabel("Prefix step k (×window_size tokens)")
    ax.set_ylabel("Mean answer entropy")
    ax.set_title(title)
    ax.legend()

    if save_to:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to, bbox_inches="tight", dpi=150)

    plt.close(fig)
    return fig


def plot_accuracy_curves(
    summary: pd.DataFrame,
    title: str = "Accuracy vs Reasoning Prefix",
    save_to: str | Path | None = None,
):
    """Plot accuracy at each k for each student."""
    sns.set_theme(style="whitegrid", context="paper")

    fig, ax = plt.subplots(figsize=(10, 6))

    for label, grp in summary.groupby("student_label"):
        grp = grp.sort_values("k")
        ax.plot(grp["k"], grp["accuracy"], marker="s", markersize=3, label=label)

    ax.set_xlabel("Prefix step k (×window_size tokens)")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.legend()

    if save_to:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to, bbox_inches="tight", dpi=150)

    plt.close(fig)
    return fig


def plot_entropy_by_correctness(
    per_correctness: pd.DataFrame,
    title: str = "Entropy Dynamics: Correct vs Incorrect Teacher",
    save_to: str | Path | None = None,
):
    """Plot entropy curves split by teacher correctness (H3 test)."""
    if per_correctness.empty:
        print("No teacher_correct column — skipping per-correctness plot.")
        return None

    sns.set_theme(style="whitegrid", context="paper")

    students = per_correctness["student_label"].unique()
    n_students = len(students)
    fig, axes = plt.subplots(1, n_students, figsize=(6 * n_students, 5), sharey=True)
    if n_students == 1:
        axes = [axes]

    for ax, label in zip(axes, students):
        for correct_val, style in [(True, "-"), (False, "--")]:
            grp = per_correctness[
                (per_correctness["student_label"] == label) &
                (per_correctness["teacher_correct"] == correct_val)
            ].sort_values("k")

            if grp.empty:
                continue

            line_label = f"Teacher {'correct' if correct_val else 'incorrect'}"
            ax.plot(grp["k"], grp["mean_entropy"], style, marker="o", markersize=3, label=line_label)

        ax.set_xlabel("k")
        ax.set_ylabel("Mean entropy")
        ax.set_title(label)
        ax.legend(fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()

    if save_to:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to, bbox_inches="tight", dpi=150)

    plt.close(fig)
    return fig


def run_full_analysis(results_path: str | Path, out_dir: str | Path | None = None):
    """Load results, compute all metrics, save plots and tables."""
    df = load_results(results_path)
    out_dir = Path(out_dir or Path(results_path).parent / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Summary
    summary = compute_summary(df)
    summary.to_csv(out_dir / "summary.csv", index=False)

    # Half-life
    half_life = compute_half_life(summary)
    half_life.to_csv(out_dir / "half_life.csv", index=False)
    print("\n=== Half-life ===")
    print(half_life.to_string(index=False))

    # Per-correctness (if available)
    per_corr = compute_per_correctness(df)
    if not per_corr.empty:
        per_corr.to_csv(out_dir / "per_correctness.csv", index=False)

    # Plots
    mode_label = df["mode"].iloc[0] if "mode" in df.columns else ""
    plot_entropy_curves(summary, f"Entropy Reduction ({mode_label})", out_dir / "entropy_curve.pdf")
    plot_accuracy_curves(summary, f"Accuracy ({mode_label})", out_dir / "accuracy_curve.pdf")
    plot_entropy_by_correctness(per_corr, save_to=out_dir / "entropy_by_correctness.pdf")

    print(f"\nAnalysis saved to {out_dir}")
    return summary, half_life
