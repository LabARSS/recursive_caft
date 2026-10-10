from __future__ import annotations

# """
# Progressive Predictive Entropy Analysis
# =========================================

# Definitions
# -----------
# Predictive entropy H_s(k):
#     Shannon entropy of the student's next-token distribution at the answer
#     position, computed after the student has consumed k * 32 tokens of the
#     teacher's reasoning chain. Measured in nats.
#     Lower value → student is more confident in a single token.

# Normalized predictive entropy H̃_s(k):
#     H_s(k) divided by H_s_p99 (the 99th percentile of all H_s values
#     observed for this student across the experiment).
#     Lies in [0, ~1]. Allows comparing students with different vocabularies
#     and different absolute entropy ranges.

# Convergence step k_conv:
#     The smallest k such that the student's predicted answer is the gold
#     answer AND H̃_s(k) < τ. This is the "answer commitment point": the
#     number of reasoning tokens the student needed to lock in the right
#     answer with high confidence. Default τ = 0.30.

# Two complementary ways to express the same phenomenon
#     Absolute     : number of teacher reasoning tokens consumed (k_conv * 32).
#     Relative     : fraction of the AVAILABLE teacher reasoning consumed
#                    (k_conv / K_total).
# Both matter and they answer different questions:
#     Absolute → "how much reasoning does the student need at all?"
#     Relative → "how much of the teacher's CoT was actually informative?"

# Example (same student, two questions):
#     Q1: total CoT = 64 tokens,   k_conv = 2  → 64 abs, 100% rel.
#         Student needed every available token.
#     Q2: total CoT = 1000 tokens, k_conv = 2  → 64 abs, 6.4% rel.
#         Student needed only the first chunk; the rest is redundant.
# Both questions converge at the same absolute step but the scientific
# interpretation is opposite. We must report both and stratify by total
# CoT length.
# """

# from __future__ import annotations

# import argparse
# import os
# from pathlib import Path
# from typing import Optional

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib as mpl

# # ══════════════════════════════════════════════════════════════
# # DEFAULT PATHS — edit for your environment, run from VS Code Run button
# # ══════════════════════════════════════════════════════════════
# DEFAULT_RESULTS = r"C:\Users\danie\OneDrive\Desktop\recursive_caft\src\core\entropy_dynamics\artifacts\entropy_dynamics_results.parquet"
# DEFAULT_TEACHER = r"C:\Users\danie\OneDrive\Desktop\recursive_caft\data\out\distillation\mmlu_synth_gptoss_b_t0_8.parquet"
# DEFAULT_OUT     = r"C:\Users\danie\OneDrive\Desktop\recursive_caft\src\core\entropy_dynamics\artifacts\analysis"
# DEFAULT_TOKENIZER_DIRS = {
#     "qwen_3b":   "Qwen/Qwen2.5-3B",
#     "phi4_mini": "microsoft/Phi-4-mini-instruct",
# }

# # ══════════════════════════════════════════════════════════════
# # CONFIG
# # ══════════════════════════════════════════════════════════════
# WINDOW_SIZE   = 32                # tokens per chunk
# TAU           = 0.30              # normalized entropy threshold for "confident"
# P99_QUANTILE  = 0.99              # used for per-model normalization
# N_HTML_EXAMPLES_PER_BUCKET = 3    # examples per CoT-length bucket
# COT_LENGTH_BUCKETS = ["short", "medium", "long"]

# # Aesthetics
# mpl.rcParams.update({
#     "figure.facecolor":  "white",
#     "axes.facecolor":    "white",
#     "axes.edgecolor":    "#333333",
#     "axes.labelcolor":   "#222222",
#     "axes.titleweight":  "bold",
#     "axes.titlesize":    13,
#     "axes.labelsize":    11,
#     "xtick.color":       "#444444",
#     "ytick.color":       "#444444",
#     "xtick.labelsize":   10,
#     "ytick.labelsize":   10,
#     "legend.frameon":    True,
#     "legend.framealpha": 0.95,
#     "legend.fontsize":   10,
#     "axes.grid":         True,
#     "grid.color":        "#dddddd",
#     "grid.linestyle":    "--",
#     "grid.linewidth":    0.7,
#     "font.family":       "DejaVu Sans",
# })

# STUDENT_PALETTE = {
#     "qwen_3b":   "#2563eb",
#     "phi4_mini": "#dc2626",
#     "llama_3b":  "#059669",
# }
# BUCKET_COLORS = {"short": "#10b981", "medium": "#f59e0b", "long": "#7c3aed"}


# # ══════════════════════════════════════════════════════════════
# # LOADING
# # ══════════════════════════════════════════════════════════════
# def load_results(path: str) -> pd.DataFrame:
#     df = pd.read_parquet(path)
#     rename = {}
#     for c in df.columns:
#         cl = c.lower()
#         if cl in ("answer_entropy", "h_s", "entropy", "entropy_value"):
#             rename[c] = "H_s"
#     df = df.rename(columns=rename)
#     df["question_id"]     = df["question_id"].astype(str)
#     df["student_correct"] = df["student_correct"].astype(bool)
#     df["k"]               = df["k"].astype(int)
#     df["H_s"]             = df["H_s"].astype(float)
#     return df


# def load_teacher(path: str) -> Optional[pd.DataFrame]:
#     if not path or not os.path.exists(path):
#         print(f"  [warn] teacher file not found: {path}")
#         return None
#     raw = pd.read_parquet(path)
#     rows = []
#     for _, r in raw.iterrows():
#         inp, out = r["input"], r["output"]
#         if not isinstance(inp, dict) or not isinstance(out, dict):
#             continue

#         # Extract teacher correctness — try multiple field names
#         is_correct = out.get("is_correct", None)
#         if is_correct is None:
#             t_ans = str(out.get("answer", "")).strip().lower()
#             gold  = str(inp.get("gold", "")).strip().lower()
#             is_correct = (t_ans == gold) if (t_ans and gold) else False

#         rows.append({
#             "question_id":     str(inp.get("question_id", "")),
#             "question":        inp.get("question", ""),
#             "options":         inp.get("options", {}),
#             "gold":            str(inp.get("gold", "")).lower(),
#             "teacher_cot":     out.get("thinking", "") or out.get("raw_response", "") or "",
#             "teacher_answer":  out.get("answer", ""),
#             "teacher_correct": bool(is_correct),
#         })
#     return pd.DataFrame(rows)


# def maybe_load_tokenizer(model_id: str):
#     """Try to load a HF tokenizer for accurate chunking. Returns None on failure."""
#     try:
#         from transformers import AutoTokenizer
#         return AutoTokenizer.from_pretrained(model_id)
#     except Exception as e:
#         print(f"  [warn] tokenizer for {model_id} not loaded: {e}")
#         return None


# # ══════════════════════════════════════════════════════════════
# # NORMALIZATION
# # ══════════════════════════════════════════════════════════════
# def normalize_entropy(df: pd.DataFrame) -> pd.DataFrame:
#     """Add H_s_norm column: per-student entropy divided by p99 of that student."""
#     df = df.copy()
#     df["H_s_norm"] = np.nan
#     norms = {}
#     for student, sub in df.groupby("student_label"):
#         p99 = float(np.quantile(sub["H_s"].dropna(), P99_QUANTILE))
#         if p99 <= 0:
#             p99 = 1.0
#         norms[student] = p99
#         df.loc[sub.index, "H_s_norm"] = sub["H_s"] / p99
#     print("  per-student p99 of H_s used for normalization:")
#     for s, p in norms.items():
#         print(f"    {s:12s}  p99 = {p:.3f}")
#     return df, norms


# # ══════════════════════════════════════════════════════════════
# # CONVERGENCE COMPUTATION
# # ══════════════════════════════════════════════════════════════
# def compute_convergence(df: pd.DataFrame) -> pd.DataFrame:
#     """One row per (student, question) with convergence statistics."""
#     rows = []
#     for (student, qid), grp in df.groupby(["student_label", "question_id"]):
#         grp = grp.sort_values("k").reset_index(drop=True)
#         K_total = int(grp["k"].max())

#         row0 = grp[grp["k"] == 0]
#         h0_norm = float(row0["H_s_norm"].iloc[0]) if not row0.empty else np.nan
#         c0      = bool(row0["student_correct"].iloc[0]) if not row0.empty else False

#         h_final_norm = float(grp.iloc[-1]["H_s_norm"])
#         c_final      = bool(grp.iloc[-1]["student_correct"])

#         rec = {
#             "student":              student,
#             "question_id":          qid,
#             "K_total":              K_total,
#             "tokens_total":         K_total * WINDOW_SIZE,
#             "H_norm_step0":         h0_norm,
#             "H_norm_final":         h_final_norm,
#             "delta_H_norm":         h0_norm - h_final_norm,
#             "correct_step0":        c0,
#             "correct_final":        c_final,
#         }

#         # k of first correct answer (regardless of confidence)
#         cor = grp[grp["student_correct"]]
#         if not cor.empty:
#             rec["k_first_correct"]      = int(cor.iloc[0]["k"])
#             rec["tokens_first_correct"] = rec["k_first_correct"] * WINDOW_SIZE
#             rec["frac_first_correct"]   = rec["k_first_correct"] / max(K_total, 1)
#         else:
#             rec["k_first_correct"]      = None
#             rec["tokens_first_correct"] = None
#             rec["frac_first_correct"]   = None

#         # k of first low-entropy step (regardless of correctness)
#         low = grp[grp["H_s_norm"] < TAU]
#         if not low.empty:
#             rec["k_first_lowH"]      = int(low.iloc[0]["k"])
#             rec["tokens_first_lowH"] = rec["k_first_lowH"] * WINDOW_SIZE
#         else:
#             rec["k_first_lowH"]      = None
#             rec["tokens_first_lowH"] = None

#         # Convergence: BOTH conditions
#         conv = grp[(grp["student_correct"]) & (grp["H_s_norm"] < TAU)]
#         if not conv.empty:
#             rec["k_conv"]      = int(conv.iloc[0]["k"])
#             rec["tokens_conv"] = rec["k_conv"] * WINDOW_SIZE
#             rec["frac_conv"]   = rec["k_conv"] / max(K_total, 1)
#         else:
#             rec["k_conv"]      = None
#             rec["tokens_conv"] = None
#             rec["frac_conv"]   = None

#         rows.append(rec)

#     return pd.DataFrame(rows)


# def stratify_by_cot_length(conv: pd.DataFrame) -> pd.DataFrame:
#     """Add a CoT length tertile per student."""
#     out = conv.copy()
#     out["cot_bucket"] = "unknown"
#     for student, sub in out.groupby("student"):
#         toks = sub["tokens_total"]
#         q33 = toks.quantile(0.33)
#         q67 = toks.quantile(0.67)
#         bucket = pd.cut(
#             toks, bins=[-np.inf, q33, q67, np.inf],
#             labels=["short", "medium", "long"],
#         )
#         out.loc[sub.index, "cot_bucket"] = bucket.astype(str)
#         print(f"  {student}: CoT length tertiles "
#               f"short ≤ {q33:.0f} < medium ≤ {q67:.0f} < long")
#     return out


# # ══════════════════════════════════════════════════════════════
# # TEXT REPORT
# # ══════════════════════════════════════════════════════════════
# def write_report(conv: pd.DataFrame, norms: dict, out_path: Path):
#     L = []
#     p = L.append

#     p("=" * 92)
#     p("PROGRESSIVE PREDICTIVE ENTROPY — CONVERGENCE REPORT")
#     p("=" * 92)
#     p("")
#     p("DEFINITIONS")
#     p("-" * 92)
#     p("H_s(k)         Shannon entropy of the student's answer-token distribution after")
#     p("               consuming k chunks of teacher reasoning. Each chunk = 32 tokens.")
#     p("")
#     p("H̃_s(k)         Normalized entropy: H_s(k) / H_s_p99 (per student).")
#     p("               H_s_p99 is the 99th percentile of all H_s values measured for")
#     p("               that student. Lies approximately in [0, 1].")
#     p("")
#     p("k_conv         Convergence step. Smallest k such that the student picks the")
#     p(f"               correct answer AND H̃_s(k) < {TAU}. The student has 'committed'.")
#     p("")
#     p("tokens_conv    k_conv × 32 — absolute number of reasoning tokens consumed")
#     p("               before convergence.")
#     p("")
#     p("frac_conv      k_conv / K_total — fraction of the AVAILABLE reasoning the")
#     p("               student needed. Different from absolute: short CoTs that fully")
#     p("               converge get frac=1.0 even if absolute is small.")
#     p("")
#     p("Per-student normalization constants H_s_p99:")
#     for s, v in norms.items():
#         p(f"   {s:12s}  H_s_p99 = {v:.4f} nats")
#     p("")
#     p("=" * 92)
#     p("PER-STUDENT SUMMARY")
#     p("=" * 92)

#     for student in sorted(conv["student"].unique()):
#         s = conv[conv["student"] == student]
#         n = len(s)
#         n_c0   = s["correct_step0"].sum()
#         n_cF   = s["correct_final"].sum()
#         n_ever = s["k_first_correct"].notna().sum()
#         n_conv = s["k_conv"].notna().sum()

#         p("")
#         p(f"── {student} ({n} questions) ──")
#         p(f"   Correct at step 0 (no reasoning):  {n_c0:>5}  ({n_c0/n:.1%})")
#         p(f"   Correct at final step:             {n_cF:>5}  ({n_cF/n:.1%})")
#         p(f"   Ever correct in trajectory:        {n_ever:>5}  ({n_ever/n:.1%})")
#         p(f"   Converged (correct & H̃<{TAU}):       {n_conv:>5}  ({n_conv/n:.1%})")
#         p("")

#         cv = s["k_conv"].dropna()
#         if not cv.empty:
#             p("   Convergence step distribution:")
#             p(f"     {'quantile':<10}{'k_conv':>10}{'tokens':>12}{'frac of CoT':>16}")
#             for q in [0.10, 0.25, 0.50, 0.75, 0.90]:
#                 kq = int(cv.quantile(q))
#                 fq = float(s["frac_conv"].quantile(q))
#                 p(f"     p{int(q*100):<9}{kq:>10}{kq*WINDOW_SIZE:>12}{fq:>15.1%}")

#         # Length stratification
#         if "cot_bucket" in s.columns:
#             p("")
#             p("   Stratified by total CoT length (per-student tertiles):")
#             p(f"     {'bucket':<10}{'n':>6}{'%conv':>9}{'med tokens':>13}{'med frac':>12}")
#             for bk in COT_LENGTH_BUCKETS:
#                 sub = s[s["cot_bucket"] == bk]
#                 if sub.empty:
#                     continue
#                 n_b   = len(sub)
#                 n_cb  = sub["k_conv"].notna().sum()
#                 med_t = sub["tokens_conv"].median()
#                 med_f = sub["frac_conv"].median()
#                 p(f"     {bk:<10}{n_b:>6}{n_cb/n_b:>8.1%} "
#                   f"{int(med_t) if pd.notna(med_t) else 'N/A':>12}"
#                   f"{med_f:>11.1%}" if pd.notna(med_f)
#                   else f"     {bk:<10}{n_b:>6}      —")

#     # Cross-student comparison
#     p("")
#     p("=" * 92)
#     p("INTERPRETATION")
#     p("=" * 92)
#     p("")
#     p("Read the metrics in this order:")
#     p("  1. 'Correct at step 0' tells you how many questions the student already knows")
#     p("     without any teacher input. These contribute nothing about the value of")
#     p("     reasoning — they would be selected as 'easy' by any baseline.")
#     p("")
#     p("  2. 'Correct at final step minus correct at step 0' = the lift from")
#     p("     reasoning. This is the share of questions where teacher CoT actually")
#     p("     changed the student's answer.")
#     p("")
#     p("  3. 'Converged' is the strict criterion: correct AND confident. The gap")
#     p("     between 'correct at final' and 'converged' tells you about a model's")
#     p("     calibration: a wide gap means the student gets the right answer but")
#     p("     keeps probability mass on the wrong options.")
#     p("")
#     p("  4. The CoT length stratification answers the central question:")
#     p("     do students need more reasoning when more is available, or do they")
#     p("     converge after a fixed number of tokens regardless of total length?")
#     p("     If median tokens_conv is roughly constant across short/medium/long,")
#     p("     reasoning has a fixed informational radius. If it grows with total")
#     p("     length, the student is 'reading along' rather than locking in early.")
#     p("")

#     text = "\n".join(L)
#     out_path.write_text(text, encoding="utf-8")
#     print(text)


# # ══════════════════════════════════════════════════════════════
# # PLOTS
# # ══════════════════════════════════════════════════════════════
# def plot_survival_curves(conv: pd.DataFrame, out_path: Path):
#     """Survival curves: P(not yet converged | step k)."""
#     fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

#     # Left: convergence in steps
#     ax = axes[0]
#     for student in sorted(conv["student"].unique()):
#         s = conv[conv["student"] == student]
#         n = len(s)
#         vals = s["k_conv"].dropna().sort_values().values
#         if len(vals) == 0:
#             continue
#         x = np.concatenate([[0], vals])
#         y = np.concatenate([[0], np.arange(1, len(vals) + 1) / n])
#         ax.step(x, y, where="post", color=STUDENT_PALETTE.get(student, "#444"),
#                 linewidth=2.5, label=f"{student}  ({len(vals)/n:.0%} converge)")
#     ax.set_xlabel("Step k  (one chunk = 32 reasoning tokens)")
#     ax.set_ylabel("Fraction of questions converged by step k")
#     ax.set_title("Convergence — by chunk index")
#     ax.set_ylim(0, 1)
#     ax.legend(loc="lower right")
#     _annotate(ax, f"Convergence = student gives gold answer AND H̃_s < {TAU}.\n"
#                   "Steeper curve → fewer reasoning chunks needed.")

#     # Right: convergence in absolute tokens
#     ax = axes[1]
#     for student in sorted(conv["student"].unique()):
#         s = conv[conv["student"] == student]
#         n = len(s)
#         vals = s["tokens_conv"].dropna().sort_values().values
#         if len(vals) == 0:
#             continue
#         x = np.concatenate([[0], vals])
#         y = np.concatenate([[0], np.arange(1, len(vals) + 1) / n])
#         ax.step(x, y, where="post", color=STUDENT_PALETTE.get(student, "#444"),
#                 linewidth=2.5, label=student)
#     ax.set_xlabel("Reasoning tokens consumed")
#     ax.set_ylabel("Fraction converged")
#     ax.set_title("Convergence — in absolute tokens")
#     ax.set_ylim(0, 1)
#     ax.legend(loc="lower right")

#     fig.suptitle("How quickly does the student commit to the right answer?",
#                  fontsize=14, fontweight="bold", y=1.02)
#     plt.tight_layout()
#     plt.savefig(out_path, dpi=160, bbox_inches="tight")
#     plt.close()
#     print(f"  saved: {out_path}")


# def plot_cot_length_stratified(conv: pd.DataFrame, out_path: Path):
#     """One subplot per student. Three survival curves: short / medium / long CoT."""
#     students = sorted(conv["student"].unique())
#     fig, axes = plt.subplots(1, len(students), figsize=(7 * len(students), 5.5))
#     if len(students) == 1:
#         axes = [axes]

#     for ax, student in zip(axes, students):
#         s = conv[conv["student"] == student]
#         for bucket in COT_LENGTH_BUCKETS:
#             sub = s[s["cot_bucket"] == bucket]
#             n_b  = len(sub)
#             if n_b == 0:
#                 continue
#             vals = sub["tokens_conv"].dropna().sort_values().values
#             if len(vals) == 0:
#                 continue
#             x = np.concatenate([[0], vals])
#             y = np.concatenate([[0], np.arange(1, len(vals) + 1) / n_b])
#             ax.step(x, y, where="post", color=BUCKET_COLORS[bucket],
#                     linewidth=2.5,
#                     label=f"{bucket} CoT (n={n_b}, {len(vals)/n_b:.0%} conv)")
#         ax.set_xlabel("Reasoning tokens consumed")
#         ax.set_ylabel("Fraction converged")
#         ax.set_title(student)
#         ax.set_ylim(0, 1)
#         ax.legend(loc="lower right", fontsize=9)

#     fig.suptitle(
#         "Does the student need more tokens when more reasoning is available?",
#         fontsize=14, fontweight="bold", y=1.02)
#     _add_global_note(
#         fig,
#         "If short/medium/long curves overlap → reasoning has a fixed informational radius.\n"
#         "If the long-CoT curve sits to the right → the student reads along and needs more.")
#     plt.tight_layout()
#     plt.savefig(out_path, dpi=160, bbox_inches="tight")
#     plt.close()
#     print(f"  saved: {out_path}")


# def plot_token_vs_frac_distribution(conv: pd.DataFrame, out_path: Path):
#     """Two histograms side by side: absolute tokens to converge vs fraction of CoT."""
#     students = sorted(conv["student"].unique())
#     fig, axes = plt.subplots(2, len(students), figsize=(7 * len(students), 9))
#     if len(students) == 1:
#         axes = axes.reshape(2, 1)

#     for j, student in enumerate(students):
#         color = STUDENT_PALETTE.get(student, "#444")
#         s = conv[conv["student"] == student]

#         # Top row: absolute tokens
#         ax = axes[0, j]
#         vals = s["tokens_conv"].dropna().values
#         if len(vals) > 0:
#             ax.hist(vals, bins=40, color=color, alpha=0.8, edgecolor="white")
#             ax.axvline(np.median(vals), color="black", linestyle="--",
#                        linewidth=1.5, label=f"median = {int(np.median(vals))}")
#         ax.set_xlabel("Reasoning tokens consumed before convergence")
#         ax.set_ylabel("Number of questions")
#         ax.set_title(f"{student} — absolute tokens to converge")
#         ax.legend()

#         # Bottom row: relative fraction
#         ax = axes[1, j]
#         vals = s["frac_conv"].dropna().values
#         if len(vals) > 0:
#             ax.hist(vals, bins=20, color=color, alpha=0.8, edgecolor="white")
#             ax.axvline(np.median(vals), color="black", linestyle="--",
#                        linewidth=1.5, label=f"median = {np.median(vals):.0%}")
#         ax.set_xlabel("Fraction of available CoT consumed")
#         ax.set_ylabel("Number of questions")
#         ax.set_title(f"{student} — fraction of CoT to converge")
#         ax.set_xlim(0, 1)
#         ax.legend()

#     fig.suptitle(
#         "Two views of the same convergence: absolute tokens vs fraction of CoT",
#         fontsize=14, fontweight="bold", y=1.00)
#     _add_global_note(
#         fig,
#         "Top row: a question that needs 64 tokens needs 64 tokens, regardless of how much CoT exists.\n"
#         "Bottom row: 64 tokens are 100% of a 64-token CoT but only 6% of a 1000-token CoT.\n"
#         "Compare both to distinguish 'student needs little' from 'teacher writes too much'.")
#     plt.tight_layout()
#     plt.savefig(out_path, dpi=160, bbox_inches="tight")
#     plt.close()
#     print(f"  saved: {out_path}")


# def plot_correct_vs_confident(conv: pd.DataFrame, out_path: Path):
#     """k_first_correct vs k_first_lowH — shows whether correctness and confidence align."""
#     students = sorted(conv["student"].unique())
#     fig, axes = plt.subplots(1, len(students), figsize=(6.5 * len(students), 5.5))
#     if len(students) == 1:
#         axes = [axes]

#     for ax, student in zip(axes, students):
#         s = conv[conv["student"] == student]
#         x = s["k_first_correct"]
#         y = s["k_first_lowH"]
#         m = x.notna() & y.notna()
#         x, y = x[m].values, y[m].values
#         if len(x) == 0:
#             ax.set_title(student)
#             continue

#         # Hexbin to handle dense overlap
#         hb = ax.hexbin(x, y, gridsize=30, cmap="Blues", mincnt=1)
#         plt.colorbar(hb, ax=ax, label="number of questions")

#         lim = max(x.max(), y.max()) + 1
#         ax.plot([0, lim], [0, lim], "k--", linewidth=1, alpha=0.5,
#                 label="x = y (simultaneous)")
#         ax.set_xlim(-0.5, lim)
#         ax.set_ylim(-0.5, lim)

#         n_cf = (x < y).sum()
#         n_kf = (x > y).sum()
#         n_eq = (x == y).sum()

#         info = (
#             f"correct first:    {n_cf/len(x):>5.0%}\n"
#             f"confident first:  {n_kf/len(x):>5.0%}\n"
#             f"simultaneous:     {n_eq/len(x):>5.0%}"
#         )
#         ax.text(0.03, 0.97, info, transform=ax.transAxes, va="top", ha="left",
#                 family="monospace", fontsize=10,
#                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.92))

#         ax.set_xlabel("Step of first correct answer")
#         ax.set_ylabel(f"Step of first H̃_s < {TAU}")
#         ax.set_title(student)
#         ax.legend(loc="lower right", fontsize=9)

#     fig.suptitle("Do correctness and confidence appear at the same step?",
#                  fontsize=14, fontweight="bold", y=1.02)
#     _add_global_note(
#         fig,
#         "Points BELOW the diagonal: model became correct before becoming confident — healthy.\n"
#         "Points ABOVE the diagonal: model was confident in a wrong answer first — anti-pattern,\n"
#         "low entropy is misleading for these examples.")
#     plt.tight_layout()
#     plt.savefig(out_path, dpi=160, bbox_inches="tight")
#     plt.close()
#     print(f"  saved: {out_path}")


# def plot_phase_heatmap(df: pd.DataFrame, conv: pd.DataFrame, out_path: Path,
#                        max_questions: int = 400, max_steps: int = 30):
#     """Heatmap: questions × step, color = H̃_s. Sorted by k_conv, white dot at conv."""
#     students = sorted(df["student_label"].unique())
#     fig, axes = plt.subplots(1, len(students), figsize=(8 * len(students), 7))
#     if len(students) == 1:
#         axes = [axes]

#     for ax, student in zip(axes, students):
#         sub = df[df["student_label"] == student]
#         cv  = conv[conv["student"] == student]

#         order = cv.sort_values("k_conv", na_position="last")["question_id"].tolist()[:max_questions]

#         wide = sub.pivot_table(index="question_id", columns="k",
#                                values="H_s_norm", aggfunc="first")
#         wide = wide.reindex(order).iloc[:, :max_steps]

#         cmap = mpl.cm.RdYlGn_r
#         cmap.set_bad("white")
#         masked = np.ma.masked_invalid(wide.values)

#         im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=0, vmax=1,
#                        interpolation="nearest")
#         ax.set_xlabel("Step k")
#         ax.set_ylabel(f"Questions (top {len(order)}, sorted by k_conv)")
#         ax.set_title(student)

#         ks, ys = [], []
#         for i, qid in enumerate(order):
#             row = cv[cv["question_id"] == qid]
#             if row.empty:
#                 continue
#             kv = row["k_conv"].iloc[0]
#             if pd.notna(kv) and kv < max_steps:
#                 ks.append(int(kv))
#                 ys.append(i)
#         if ks:
#             ax.scatter(ks, ys, s=4, c="white", edgecolors="black", linewidths=0.3)

#         cb = plt.colorbar(im, ax=ax, label="H̃_s (normalized entropy)")
#         cb.outline.set_edgecolor("#888")

#     fig.suptitle("Phase transition heatmap: rows are questions, color is normalized entropy",
#                  fontsize=14, fontweight="bold", y=1.01)
#     _add_global_note(
#         fig,
#         f"Each row is one question. The leftmost cell is step 0 (no reasoning) and we move right.\n"
#         f"Green = student is confident; red = student is uncertain. Rows are sorted by\n"
#         f"convergence step. White dots mark the convergence step (correct AND H̃_s < {TAU}).")
#     plt.tight_layout()
#     plt.savefig(out_path, dpi=160, bbox_inches="tight")
#     plt.close()
#     print(f"  saved: {out_path}")


# def _annotate(ax, text):
#     ax.text(0.02, 0.98, text, transform=ax.transAxes, va="top", ha="left",
#             fontsize=9, color="#555",
#             bbox=dict(boxstyle="round", facecolor="#f8f8f8", edgecolor="#ccc"))


# def _add_global_note(fig, text):
#     fig.text(0.5, -0.02, text, ha="center", va="top",
#              fontsize=10, color="#444", style="italic")


# # ══════════════════════════════════════════════════════════════
# # HTML COLORED EXAMPLES
# # ══════════════════════════════════════════════════════════════
# def _entropy_to_rgb(h_norm: float) -> str:
#     t = float(np.clip(h_norm, 0, 1))
#     if t < 0.5:
#         r = int(60 + 195 * (t * 2))
#         g = 200
#         b = 80
#     else:
#         r = 255
#         g = int(200 * (1 - (t - 0.5) * 2))
#         b = 70
#     return f"rgb({r}, {g}, {b})"


# def chunk_cot_with_tokenizer(cot_text: str, tokenizer, window_size: int) -> list:
#     """Tokenize CoT and split into chunks of `window_size` token IDs.
#     Returns list of decoded chunk strings."""
#     if tokenizer is None:
#         words = cot_text.split()
#         return [
#             " ".join(words[i*window_size:(i+1)*window_size])
#             for i in range((len(words) + window_size - 1) // window_size)
#         ]
#     ids = tokenizer.encode(cot_text, add_special_tokens=False)
#     chunks = []
#     for i in range(0, len(ids), window_size):
#         slice_ids = ids[i:i+window_size]
#         chunks.append(tokenizer.decode(slice_ids, skip_special_tokens=True))
#     return chunks


# def generate_html_examples(df: pd.DataFrame, conv: pd.DataFrame,
#                             teacher_df: pd.DataFrame, student: str,
#                             tokenizer, out_path: Path):
#     """Build an HTML file showing colored chunks for selected examples."""
#     if teacher_df is None:
#         print(f"  [skip html] no teacher data for {student}")
#         return

#     # Pick examples: stratified by CoT length bucket, varied convergence behaviour
#     sub = conv[conv["student"] == student].copy()
#     sub = sub.merge(teacher_df[["question_id", "question", "options", "gold",
#                                  "teacher_cot", "teacher_correct"]],
#                     on="question_id", how="inner")
#     if sub.empty:
#         print(f"  [skip html] empty merged data for {student}")
#         return

#     chosen_qids = []
#     for bucket in COT_LENGTH_BUCKETS:
#         b = sub[sub["cot_bucket"] == bucket]
#         if b.empty:
#             continue
#         b_conv   = b[b["k_conv"].notna()].sort_values("k_conv")
#         b_noconv = b[b["k_conv"].isna()]
#         # one early-converging, one late-converging, one never-converging
#         picks = []
#         if not b_conv.empty:
#             picks.append(b_conv.iloc[0]["question_id"])           # early
#             picks.append(b_conv.iloc[-1]["question_id"])          # late
#         if not b_noconv.empty:
#             picks.append(b_noconv.iloc[0]["question_id"])         # never
#         for qid in picks[:N_HTML_EXAMPLES_PER_BUCKET]:
#             if qid not in chosen_qids:
#                 chosen_qids.append(qid)

#     if not chosen_qids:
#         print(f"  [skip html] no example questions for {student}")
#         return

#     # Build HTML
#     html = [
#         "<!DOCTYPE html><html><head><meta charset='utf-8'>",
#         f"<title>Progressive Entropy — {student}</title>",
#         "<style>",
#         "body{font-family:'SF Pro Display',-apple-system,BlinkMacSystemFont,sans-serif;",
#         "     max-width:1150px;margin:30px auto;padding:25px;background:#f5f5f7;color:#1d1d1f;}",
#         "h1{font-size:28px;margin-bottom:5px;}",
#         "h1 .student{color:#0071e3;font-family:monospace;}",
#         ".intro{background:#fff;border-radius:14px;padding:20px;margin:15px 0;",
#         "       box-shadow:0 1px 4px rgba(0,0,0,0.05);}",
#         ".intro h3{margin-top:0;}",
#         ".gradient-bar{height:18px;border-radius:6px;margin:10px 0;",
#         " background:linear-gradient(to right,rgb(60,200,80),rgb(255,200,80),rgb(255,0,70));}",
#         ".axes{display:flex;justify-content:space-between;font-size:12px;color:#666;}",
#         ".example{background:#fff;border-radius:14px;padding:22px;margin:25px 0;",
#         "         box-shadow:0 2px 8px rgba(0,0,0,0.06);}",
#         ".header{font-size:14px;margin-bottom:10px;}",
#         ".header b{color:#0071e3;}",
#         ".question{font-size:15px;color:#222;margin:12px 0;line-height:1.5;}",
#         ".meta{font-size:12px;color:#555;margin:8px 0 14px 0;}",
#         ".badge{display:inline-block;padding:3px 9px;border-radius:8px;",
#         "       font-size:11px;font-weight:600;margin-right:6px;}",
#         ".badge.correct{background:#dcfce7;color:#166534;}",
#         ".badge.wrong{background:#fee2e2;color:#991b1b;}",
#         ".badge.bucket{background:#dbeafe;color:#1e40af;}",
#         ".cot-flow{line-height:2.4;font-size:13.5px;color:#1d1d1f;",
#         "          padding:14px;background:#fafafa;border-radius:10px;}",
#         ".chunk{padding:4px 7px;border-radius:5px;margin:2px;",
#         "       box-shadow:inset 0 -2px 0 rgba(0,0,0,0.08);}",
#         ".chunk.conv{outline:2.5px solid #111;outline-offset:1px;}",
#         ".hbadge{display:inline-block;padding:1px 6px;font-family:monospace;",
#         "        font-size:10px;font-weight:bold;color:#222;",
#         "        background:#fff;border:1px solid rgba(0,0,0,0.2);",
#         "        border-radius:4px;margin:0 2px;vertical-align:super;}",
#         ".step0{background:#e5e7eb;color:#333;padding:3px 8px;border-radius:6px;",
#         "       font-family:monospace;font-size:11px;margin-right:6px;}",
#         "</style></head><body>",
#         f"<h1>Progressive entropy trajectory — <span class='student'>{student}</span></h1>",
#         "<div class='intro'>",
#         "<h3>How to read this page</h3>",
#         f"<p>Each block below is one MMLU question. The student model receives the teacher's "
#         f"reasoning in chunks of {WINDOW_SIZE} tokens. After each new chunk we measure the "
#         f"student's <b>normalized predictive entropy</b> H̃_s on the answer position. "
#         f"Each chunk in the text below is colored by H̃_s at that step:</p>",
#         "<div class='gradient-bar'></div>",
#         "<div class='axes'><span>H̃_s = 0 (fully confident)</span>"
#         "<span>H̃_s = 1 (maximally uncertain)</span></div>",
#         f"<p>The numeric badge after each chunk is the actual H̃_s value. "
#         f"A black outline marks the <b>convergence step</b>: the first chunk after "
#         f"which the student gives the correct answer with H̃_s &lt; {TAU}.</p>",
#         "</div>",
#     ]

#     for qid in chosen_qids:
#         rows = df[(df["student_label"] == student) & (df["question_id"] == qid)]
#         rows = rows.sort_values("k").reset_index(drop=True)
#         if rows.empty:
#             continue

#         meta = sub[sub["question_id"] == qid].iloc[0]
#         cot   = str(meta["teacher_cot"])
#         chunks = chunk_cot_with_tokenizer(cot, tokenizer, WINDOW_SIZE)

#         gold       = str(meta["gold"]).upper()
#         t_correct  = bool(meta["teacher_correct"])
#         bucket     = meta["cot_bucket"]
#         k_conv     = meta["k_conv"]
#         k_conv_int = int(k_conv) if pd.notna(k_conv) else None

#         h0 = float(rows.iloc[0]["H_s_norm"])
#         hF = float(rows.iloc[-1]["H_s_norm"])
#         ans_F = str(rows.iloc[-1]["student_answer"])

#         html.append("<div class='example'>")
#         html.append(
#             f"<div class='header'>Q<b>{qid}</b> · "
#             f"<span class='badge bucket'>{bucket} CoT ({len(chunks)} chunks)</span>"
#             f"<span class='badge {'correct' if t_correct else 'wrong'}'>"
#             f"teacher {'✓' if t_correct else '✗'}</span>"
#             f"gold = <b>{gold}</b></div>"
#         )
#         html.append(f"<div class='question'>{_escape(meta['question'])[:600]}</div>")

#         if k_conv_int is not None:
#             conv_msg = (f"converged at step {k_conv_int} "
#                         f"(after {k_conv_int*WINDOW_SIZE} reasoning tokens, "
#                         f"{k_conv_int/max(len(chunks),1):.0%} of CoT)")
#         else:
#             conv_msg = "never converged"
#         html.append(
#             f"<div class='meta'>H̃_s trajectory: {h0:.2f} → {hF:.2f}  ·  "
#             f"final answer: <b>{ans_F.upper()}</b>  ·  {conv_msg}</div>"
#         )

#         # The colored CoT flow
#         html.append("<div class='cot-flow'>")
#         # Step 0 (no reasoning yet)
#         html.append(f"<span class='step0'>[Q only]</span>")
#         html.append(f"<span class='hbadge'>H̃={h0:.2f}</span> ")

#         for k_idx in range(1, len(rows)):
#             row    = rows.iloc[k_idx]
#             h_norm = float(row["H_s_norm"])
#             chunk_text = chunks[k_idx - 1] if k_idx - 1 < len(chunks) else ""
#             if not chunk_text.strip():
#                 continue
#             color = _entropy_to_rgb(h_norm)
#             outline = " conv" if (k_conv_int is not None and k_idx == k_conv_int) else ""
#             html.append(
#                 f"<span class='chunk{outline}' style='background:{color};'>"
#                 f"{_escape(chunk_text)}</span>"
#                 f"<span class='hbadge'>H̃={h_norm:.2f}</span> "
#             )
#         html.append("</div>")
#         html.append("</div>")

#     html.append("</body></html>")
#     out_path.write_text("\n".join(html), encoding="utf-8")
#     print(f"  saved: {out_path}")


# def _escape(s: str) -> str:
#     return (str(s)
#             .replace("&", "&amp;")
#             .replace("<", "&lt;")
#             .replace(">", "&gt;"))


# # ══════════════════════════════════════════════════════════════
# # MAIN
# # ══════════════════════════════════════════════════════════════
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--results", default=DEFAULT_RESULTS)
#     parser.add_argument("--teacher", default=DEFAULT_TEACHER)
#     parser.add_argument("--out",     default=DEFAULT_OUT)
#     args = parser.parse_args()

#     out = Path(args.out)
#     out.mkdir(parents=True, exist_ok=True)

#     print("[1/6] Loading results…")
#     df = load_results(args.results)
#     print(f"   {len(df)} measurements, {df['question_id'].nunique()} questions, "
#           f"students = {sorted(df['student_label'].unique())}")

#     print("[2/6] Normalizing entropy per student…")
#     df, norms = normalize_entropy(df)

#     print("[3/6] Computing convergence…")
#     conv = compute_convergence(df)
#     print("[4/6] Stratifying by CoT length tertile…")
#     conv = stratify_by_cot_length(conv)
#     conv.to_parquet(out / "convergence.parquet", index=False)

#     print("[5/6] Loading teacher CoT…")
#     teacher_df = load_teacher(args.teacher)

#     print("[6/6] Plots and report…")
#     plot_survival_curves(conv,           out / "01_convergence_survival.png")
#     plot_cot_length_stratified(conv,     out / "02_convergence_by_cot_length.png")
#     plot_token_vs_frac_distribution(conv, out / "03_tokens_vs_fraction.png")
#     plot_correct_vs_confident(conv,      out / "04_correct_vs_confident.png")
#     plot_phase_heatmap(df, conv,         out / "05_phase_heatmap.png")
#     write_report(conv, norms, out / "REPORT.txt")

#     if teacher_df is not None:
#         print("\nGenerating colored HTML examples…")
#         for student in sorted(df["student_label"].unique()):
#             tok_id = DEFAULT_TOKENIZER_DIRS.get(student)
#             tok = maybe_load_tokenizer(tok_id) if tok_id else None
#             generate_html_examples(
#                 df, conv, teacher_df, student, tok,
#                 out / f"06_examples_{student}.html",
#             )

#     print(f"\n✓ All outputs in: {out}")
#     print("  Read REPORT.txt first — it defines every term and explains every plot.")


# if __name__ == "__main__":
#     main()

"""
Progressive Entropy — State Transition Analysis
=================================================

At each step k we classify the student into one of four states based on
two binary axes: correctness (right/wrong) and confidence (H_s below/above
the per-student median).

    CC = Confident & Correct      (the goal state)
    CW = Confident & Wrong        (dangerous: locked in wrong)
    UC = Uncertain & Correct      (right answer, low conviction)
    UW = Uncertain & Wrong        (the typical starting point)

Main research question
----------------------
For a question that starts in UW at step 0, after how many teacher
reasoning tokens does the student first reach CC — AND does this happen
before the teacher's CoT runs out (so the result is not trivial)?

We also track the inverse case: questions that start CC and degrade to
CW or UW as more (possibly noisy) reasoning is fed in.

Outputs
-------
1.  REPORT.txt              full text report with definitions
2.  01_state_evolution.png  stacked area: share of each state vs step k
3.  02_transition_flow.png  Sankey-style flow: starting state → final state
4.  03_uw_to_cc_dist.png    histogram of tokens needed for UW→CC, plus
                            fraction-of-CoT histogram side by side
5.  04_per_question_grid.png  small heatmap of state per question per step
6.  05_examples_{student}.html  colored CoT chunks with per-chunk state badges
"""

# from __future__ import annotations
# import argparse, os
# from pathlib import Path
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib as mpl

# # ── PATHS ───────────────────────────────────────────────────────────
# DEFAULT_RESULTS = r"C:\Users\danie\OneDrive\Desktop\recursive_caft\src\core\entropy_dynamics\artifacts\entropy_dynamics_results.parquet"
# DEFAULT_TEACHER = r"C:\Users\danie\OneDrive\Desktop\recursive_caft\data\out\distillation\mmlu_synth_gptoss_b_t0_8.parquet"
# DEFAULT_OUT     = r"C:\Users\danie\OneDrive\Desktop\recursive_caft\src\core\entropy_dynamics\artifacts\analysis"
# TOKENIZERS = {"qwen_3b": "Qwen/Qwen2.5-3B", "phi4_mini": "microsoft/Phi-4-mini-instruct"}
# WINDOW_SIZE = 32

# # ── STATES ──────────────────────────────────────────────────────────
# STATE_COLORS = {
#     "CC": "#10b981",   # Confident Correct — green
#     "UC": "#84cc16",   # Uncertain Correct — light green
#     "UW": "#f97316",   # Uncertain Wrong — orange
#     "CW": "#dc2626",   # Confident Wrong — red
# }
# STATE_ORDER = ["CC", "UC", "UW", "CW"]

# mpl.rcParams.update({
#     "figure.facecolor":"white", "axes.facecolor":"white",
#     "axes.titleweight":"bold", "axes.titlesize":13,
#     "axes.grid":True, "grid.linestyle":"--", "grid.linewidth":0.6,
#     "grid.color":"#dddddd", "font.family":"DejaVu Sans",
# })

# # ── LOADING ─────────────────────────────────────────────────────────
# def load_results(path):
#     df = pd.read_parquet(path)
#     rn = {c: "H_s" for c in df.columns
#           if c.lower() in ("answer_entropy","h_s","entropy","entropy_value")}
#     df = df.rename(columns=rn)
#     df["question_id"] = df["question_id"].astype(str)
#     df["k"] = df["k"].astype(int)
#     df["H_s"] = df["H_s"].astype(float)
#     df["student_correct"] = df["student_correct"].astype(bool)
#     return df

# def load_teacher(path):
#     if not path or not os.path.exists(path): return None
#     raw = pd.read_parquet(path)
#     rows = []
#     for _, r in raw.iterrows():
#         inp, out = r["input"], r["output"]
#         if not isinstance(inp, dict) or not isinstance(out, dict): continue
#         rows.append({
#             "question_id": str(inp.get("question_id","")),
#             "question": inp.get("question",""),
#             "options": inp.get("options",{}),
#             "gold": str(inp.get("gold","")).lower(),
#             "teacher_cot": out.get("thinking","") or out.get("raw_response","") or "",
#             "teacher_correct": bool(out.get("is_correct", False)),
#         })
#     return pd.DataFrame(rows)

# def maybe_tokenizer(model_id):
#     try:
#         from transformers import AutoTokenizer
#         return AutoTokenizer.from_pretrained(model_id)
#     except Exception: return None

# # ── STATES ──────────────────────────────────────────────────────────
# def assign_states(df):
#     """Add 'confident' and 'state' columns. Threshold = per-student median of H_s."""
#     df = df.copy()
#     df["confident"] = False
#     thresholds = {}
#     for student, sub in df.groupby("student_label"):
#         thr = float(np.median(sub["H_s"]))
#         thresholds[student] = thr
#         df.loc[sub.index, "confident"] = sub["H_s"] < thr
#     def state_of(r):
#         if r["confident"]    and r["student_correct"]:      return "CC"
#         if r["confident"]    and not r["student_correct"]:  return "CW"
#         if not r["confident"] and r["student_correct"]:     return "UC"
#         return "UW"
#     df["state"] = df.apply(state_of, axis=1)
#     return df, thresholds

# # ── PER-QUESTION TRANSITION ─────────────────────────────────────────
# def compute_transitions(df):
#     rows = []
#     for (student, qid), g in df.groupby(["student_label","question_id"]):
#         g = g.sort_values("k").reset_index(drop=True)
#         K_total = int(g["k"].max())
#         s0  = g.iloc[0]["state"]
#         sF  = g.iloc[-1]["state"]
#         # First step entering CC
#         cc = g[g["state"]=="CC"]
#         k_cc = int(cc.iloc[0]["k"]) if not cc.empty else None
#         # Did UW→CC happen before CoT ended?
#         uw_to_cc = (s0=="UW") and (k_cc is not None) and (k_cc < K_total)
#         # Did CC degrade?
#         cc_to_wrong = (s0 in ("CC","UC")) and (sF in ("CW","UW"))
#         rows.append({
#             "student": student, "question_id": qid,
#             "K_total": K_total, "tokens_total": K_total*WINDOW_SIZE,
#             "state_0": s0, "state_final": sF,
#             "k_first_CC": k_cc,
#             "tokens_first_CC": k_cc*WINDOW_SIZE if k_cc is not None else None,
#             "frac_first_CC": k_cc/max(K_total,1) if k_cc is not None else None,
#             "uw_to_cc_before_end": uw_to_cc,
#             "degraded": cc_to_wrong,
#         })
#     return pd.DataFrame(rows)

# # ── PLOTS ───────────────────────────────────────────────────────────
# def plot_state_evolution(df, out):
#     students = sorted(df["student_label"].unique())
#     fig, axes = plt.subplots(1, len(students), figsize=(7*len(students), 5), sharey=True)
#     if len(students)==1: axes=[axes]
#     for ax, st in zip(axes, students):
#         sub = df[df["student_label"]==st]
#         pivot = sub.groupby(["k","state"]).size().unstack(fill_value=0)
#         pivot = pivot.reindex(columns=STATE_ORDER, fill_value=0)
#         share = pivot.div(pivot.sum(axis=1), axis=0)
#         x = share.index.values * WINDOW_SIZE
#         ax.stackplot(x, *[share[s] for s in STATE_ORDER],
#                      colors=[STATE_COLORS[s] for s in STATE_ORDER],
#                      labels=STATE_ORDER, alpha=0.85)
#         ax.set_xlabel("Teacher reasoning tokens consumed")
#         ax.set_ylabel("Share of questions")
#         ax.set_title(st); ax.set_ylim(0,1); ax.set_xlim(0, x.max())
#         ax.legend(loc="center right", fontsize=9)
#     fig.suptitle("State distribution as a function of reasoning tokens",
#                  fontsize=14, y=1.02)
#     fig.text(0.5,-0.04,
#         "CC=confident correct, UC=uncertain correct, UW=uncertain wrong, CW=confident wrong.\n"
#         "Confident = H_s below per-student median. Growth of green = reasoning is helping.",
#         ha="center", fontsize=10, style="italic", color="#444")
#     plt.tight_layout(); plt.savefig(out, dpi=160, bbox_inches="tight"); plt.close()
#     print(f"  saved: {out}")

# def plot_transition_flow(trans, out):
#     students = sorted(trans["student"].unique())
#     fig, axes = plt.subplots(1, len(students), figsize=(7*len(students), 5.5))
#     if len(students)==1: axes=[axes]
#     for ax, st in zip(axes, students):
#         sub = trans[trans["student"]==st]
#         flow = sub.groupby(["state_0","state_final"]).size().unstack(fill_value=0)
#         flow = flow.reindex(index=STATE_ORDER, columns=STATE_ORDER, fill_value=0)
#         flow_pct = flow.div(flow.sum(axis=1), axis=0).fillna(0) * 100
#         im = ax.imshow(flow_pct.values, cmap="Blues", vmin=0, vmax=100, aspect="auto")
#         ax.set_xticks(range(4)); ax.set_xticklabels(STATE_ORDER)
#         ax.set_yticks(range(4)); ax.set_yticklabels(STATE_ORDER)
#         ax.set_xlabel("Final state"); ax.set_ylabel("Initial state (k=0)")
#         ax.set_title(st)
#         for i in range(4):
#             for j in range(4):
#                 v = flow_pct.values[i,j]
#                 n = flow.values[i,j]
#                 if v > 0:
#                     color = "white" if v > 50 else "#222"
#                     ax.text(j, i, f"{v:.0f}%\nn={n}", ha="center", va="center",
#                             fontsize=9, color=color)
#         plt.colorbar(im, ax=ax, label="% of row")
#     fig.suptitle("Initial state → final state transitions",
#                  fontsize=14, y=1.02)
#     fig.text(0.5,-0.04,
#         "Each row sums to 100%. Diagonal cells = state stayed the same.\n"
#         "Top-left UW→CC cell is the central success: wrong & uncertain learned the answer.",
#         ha="center", fontsize=10, style="italic", color="#444")
#     plt.tight_layout(); plt.savefig(out, dpi=160, bbox_inches="tight"); plt.close()
#     print(f"  saved: {out}")

# def plot_uw_to_cc(trans, out):
#     students = sorted(trans["student"].unique())
#     fig, axes = plt.subplots(2, len(students), figsize=(7*len(students), 9))
#     if len(students)==1: axes = axes.reshape(2,1)
#     for j, st in enumerate(students):
#         sub = trans[(trans["student"]==st) & (trans["state_0"]=="UW") & trans["uw_to_cc_before_end"]]
#         # Top: absolute tokens
#         ax = axes[0,j]
#         vals = sub["tokens_first_CC"].dropna().values
#         if len(vals):
#             ax.hist(vals, bins=30, color=STATE_COLORS["CC"], alpha=0.85, edgecolor="white")
#             ax.axvline(np.median(vals), color="black", linestyle="--",
#                        label=f"median = {int(np.median(vals))} tokens")
#             ax.legend()
#         ax.set_xlabel("Reasoning tokens to reach CC")
#         ax.set_ylabel("Number of questions")
#         ax.set_title(f"{st} — UW → CC (n={len(vals)})")
#         # Bottom: fraction of CoT
#         ax = axes[1,j]
#         vals = sub["frac_first_CC"].dropna().values
#         if len(vals):
#             ax.hist(vals, bins=20, color=STATE_COLORS["CC"], alpha=0.85, edgecolor="white")
#             ax.axvline(np.median(vals), color="black", linestyle="--",
#                        label=f"median = {np.median(vals):.0%}")
#             ax.legend()
#         ax.set_xlim(0,1)
#         ax.set_xlabel("Fraction of available CoT")
#         ax.set_ylabel("Number of questions")
#         ax.set_title(f"{st} — fraction of CoT to UW→CC")
#     fig.suptitle("How much teacher reasoning is needed to flip UW → CC?",
#                  fontsize=14, y=1.00)
#     fig.text(0.5,-0.03,
#         "Only questions that started UW and reached CC strictly before the CoT ended.\n"
#         "Top row = absolute tokens. Bottom row = fraction of available CoT (must be < 100%).",
#         ha="center", fontsize=10, style="italic", color="#444")
#     plt.tight_layout(); plt.savefig(out, dpi=160, bbox_inches="tight"); plt.close()
#     print(f"  saved: {out}")

# # ── HTML ────────────────────────────────────────────────────────────
# def chunk_with_tok(text, tok, w):
#     if tok is None:
#         words = text.split()
#         return [" ".join(words[i*w:(i+1)*w]) for i in range((len(words)+w-1)//w)]
#     ids = tok.encode(text, add_special_tokens=False)
#     return [tok.decode(ids[i:i+w], skip_special_tokens=True) for i in range(0,len(ids),w)]

# def html_examples(df, trans, teacher_df, student, tok, out):
#     if teacher_df is None: return
#     sub = trans[trans["student"]==student].merge(teacher_df, on="question_id", how="inner")
#     if sub.empty: return
#     # Pick: 2 successful UW→CC, 2 degradations, 2 stable CC
#     picks = []
#     a = sub[(sub["state_0"]=="UW") & sub["uw_to_cc_before_end"]].sort_values("k_first_CC")
#     if not a.empty: picks += list(a.head(2)["question_id"])
#     b = sub[sub["degraded"]]
#     if not b.empty: picks += list(b.head(2)["question_id"])
#     c = sub[(sub["state_0"]=="CC") & (sub["state_final"]=="CC")]
#     if not c.empty: picks += list(c.head(2)["question_id"])
#     picks = list(dict.fromkeys(picks))
#     if not picks: return

#     html = ["<!DOCTYPE html><html><head><meta charset='utf-8'><style>",
#         "body{font-family:-apple-system,sans-serif;max-width:1100px;margin:25px auto;",
#         "padding:20px;background:#f5f5f7;color:#1d1d1f;}",
#         "h1{font-size:26px;}",
#         ".legend{background:#fff;border-radius:12px;padding:18px;margin:15px 0;}",
#         ".st{display:inline-block;padding:3px 9px;border-radius:6px;font-weight:600;",
#         "font-size:11px;color:white;margin:2px;}",
#         ".ex{background:#fff;border-radius:12px;padding:20px;margin:18px 0;",
#         "box-shadow:0 2px 6px rgba(0,0,0,0.06);}",
#         ".q{font-size:14px;color:#333;margin:8px 0;}",
#         ".meta{font-size:12px;color:#666;margin:8px 0;}",
#         ".flow{line-height:2.5;font-size:13px;padding:12px;background:#fafafa;",
#         "border-radius:8px;}",
#         ".chunk{padding:3px 6px;border-radius:4px;margin:1px;",
#         "box-shadow:inset 0 -3px 0 rgba(0,0,0,0.15);}",
#         ".sb{display:inline-block;padding:1px 5px;font-family:monospace;font-size:10px;",
#         "color:white;font-weight:bold;border-radius:3px;margin:0 1px;vertical-align:super;}",
#         ".hb{display:inline-block;padding:1px 5px;font-family:monospace;font-size:10px;",
#         "background:#eee;color:#222;border-radius:3px;margin:0 1px;vertical-align:super;}",
#         "</style></head><body>",
#         f"<h1>State trajectories — {student}</h1>",
#         "<div class='legend'><b>States:</b> ",
#         f"<span class='st' style='background:{STATE_COLORS['CC']}'>CC = confident correct</span>",
#         f"<span class='st' style='background:{STATE_COLORS['UC']}'>UC = uncertain correct</span>",
#         f"<span class='st' style='background:{STATE_COLORS['UW']}'>UW = uncertain wrong</span>",
#         f"<span class='st' style='background:{STATE_COLORS['CW']}'>CW = confident wrong</span>",
#         "<p>Each chunk = 32 reasoning tokens. Background color = state AFTER the chunk. "
#         "Badges show H_s value and state code.</p></div>",
#     ]
#     for qid in picks:
#         rows = df[(df["student_label"]==student) & (df["question_id"]==qid)].sort_values("k").reset_index(drop=True)
#         meta = sub[sub["question_id"]==qid].iloc[0]
#         chunks = chunk_with_tok(str(meta["teacher_cot"]), tok, WINDOW_SIZE)
#         html.append("<div class='ex'>")
#         html.append(f"<b>Q{qid}</b> · gold=<b>{str(meta['gold']).upper()}</b> · "
#                     f"start={meta['state_0']} → final={meta['state_final']}")
#         html.append(f"<div class='q'>{str(meta['question'])[:400]}</div>")
#         if meta["state_0"]=="UW" and meta["uw_to_cc_before_end"]:
#             kfc = int(meta["k_first_CC"])
#             html.append(f"<div class='meta'>UW→CC at chunk {kfc} ({kfc*WINDOW_SIZE} tokens, "
#                         f"{kfc/max(meta['K_total'],1):.0%} of CoT)</div>")
#         elif meta["degraded"]:
#             html.append(f"<div class='meta'>Started correct, degraded to {meta['state_final']}</div>")
#         html.append("<div class='flow'>")
#         for k_idx in range(len(rows)):
#             row = rows.iloc[k_idx]
#             state = row["state"]
#             color = STATE_COLORS[state]
#             chunk_text = chunks[k_idx-1] if 1<=k_idx<=len(chunks) else "[Q only]"
#             chunk_text = chunk_text.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
#             html.append(
#                 f"<span class='chunk' style='background:{color}33;'>{chunk_text}</span>"
#                 f"<span class='sb' style='background:{color};'>{state}</span>"
#                 f"<span class='hb'>H={row['H_s']:.2f}</span> "
#             )
#         html.append("</div></div>")
#     html.append("</body></html>")
#     out.write_text("\n".join(html), encoding="utf-8")
#     print(f"  saved: {out}")

# # ── REPORT ──────────────────────────────────────────────────────────
# def write_report(trans, thresholds, out):
#     L = []
#     L.append("="*88)
#     L.append("STATE TRANSITION REPORT")
#     L.append("="*88)
#     L.append("\nDEFINITIONS")
#     L.append("  H_s         Shannon entropy of student answer distribution at step k.")
#     L.append("  Confident   H_s below per-student median. Thresholds:")
#     for s,t in thresholds.items(): L.append(f"                {s}: {t:.3f}")
#     L.append("  CC/CW/UC/UW Confident-Correct / Confident-Wrong / Uncertain-Correct / Uncertain-Wrong.")
#     L.append("  UW → CC (before end): the question started uncertain-wrong AND reached")
#     L.append("                confident-correct STRICTLY BEFORE the teacher CoT ended.")
#     L.append("                This is the central success metric: it proves reasoning was")
#     L.append("                actually informative, not that the model just saw the full CoT.\n")
#     for st in sorted(trans["student"].unique()):
#         s = trans[trans["student"]==st]
#         n = len(s)
#         L.append("─"*88)
#         L.append(f"STUDENT: {st}  ({n} questions)")
#         L.append("─"*88)
#         L.append("\nInitial state distribution (at k=0):")
#         for state in STATE_ORDER:
#             n_s = (s["state_0"]==state).sum()
#             L.append(f"  {state}: {n_s:>5}  ({n_s/n:.1%})")
#         L.append("\nFinal state distribution:")
#         for state in STATE_ORDER:
#             n_s = (s["state_final"]==state).sum()
#             L.append(f"  {state}: {n_s:>5}  ({n_s/n:.1%})")
#         # Central success metric
#         uw0 = s[s["state_0"]=="UW"]
#         succ = uw0[uw0["uw_to_cc_before_end"]]
#         L.append(f"\nUW → CC analysis:")
#         L.append(f"  questions starting UW:                {len(uw0):>5}")
#         L.append(f"  reached CC at any point:              {uw0['k_first_CC'].notna().sum():>5}")
#         L.append(f"  reached CC BEFORE CoT ended:          {len(succ):>5}  "
#                  f"({len(succ)/max(len(uw0),1):.1%} of UW)")
#         if len(succ):
#             L.append(f"  median tokens needed:                 {int(succ['tokens_first_CC'].median())}")
#             L.append(f"  median fraction of CoT:               {succ['frac_first_CC'].median():.1%}")
#         # Degradation
#         good0 = s[s["state_0"].isin(["CC","UC"])]
#         deg = good0[good0["degraded"]]
#         L.append(f"\nDegradation analysis:")
#         L.append(f"  questions starting CC or UC:          {len(good0):>5}")
#         L.append(f"  degraded to wrong by final step:      {len(deg):>5}  "
#                  f"({len(deg)/max(len(good0),1):.1%})")
#         L.append("")
#     out.write_text("\n".join(L), encoding="utf-8")
#     print("\n".join(L))

# # ── MAIN ────────────────────────────────────────────────────────────
# def main():
#     p = argparse.ArgumentParser()
#     p.add_argument("--results", default=DEFAULT_RESULTS)
#     p.add_argument("--teacher", default=DEFAULT_TEACHER)
#     p.add_argument("--out", default=DEFAULT_OUT)
#     args = p.parse_args()
#     out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

#     print("[1] Loading…")
#     df = load_results(args.results)
#     print(f"   {len(df)} measurements, {df['question_id'].nunique()} questions")

#     print("[2] Assigning states…")
#     df, thresholds = assign_states(df)

#     print("[3] Computing transitions…")
#     trans = compute_transitions(df)
#     trans.to_parquet(out/"transitions.parquet", index=False)

#     print("[4] Loading teacher…")
#     teacher_df = load_teacher(args.teacher)

#     print("[5] Plots & report…")
#     plot_state_evolution(df, out/"01_state_evolution.png")
#     plot_transition_flow(trans, out/"02_transition_flow.png")
#     plot_uw_to_cc(trans, out/"03_uw_to_cc_dist.png")
#     write_report(trans, thresholds, out/"REPORT.txt")

#     if teacher_df is not None:
#         for st in sorted(df["student_label"].unique()):
#             tok = maybe_tokenizer(TOKENIZERS.get(st))
#             html_examples(df, trans, teacher_df, st, tok, out/f"05_examples_{st}.html")

#     print(f"\n✓ Done. Outputs in: {out}")

# if __name__ == "__main__":
#     main()

"""
Progressive Entropy — Interactive HTML Dashboard
==================================================
Builds ONE self-contained HTML file with four tabs:
  1. Overview      — key numbers, definitions, transition matrix
  2. Statistics    — full text report
  3. Examples      — colored CoT chunks with per-chunk answer/H_s/state
  4. Plots         — interactive Plotly charts (state evolution, transitions,
                     UW→CC distribution, per-question heatmap)

Run from VS Code Run button. Open the resulting HTML in any browser.
"""
# import argparse, os, json, base64
# from pathlib import Path
# import numpy as np
# import pandas as pd
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import plotly.io as pio

# # ── PATHS ───────────────────────────────────────────────────────────
# DEFAULT_RESULTS = r"C:\Users\danie\OneDrive\Desktop\recursive_caft\src\core\entropy_dynamics\artifacts\entropy_dynamics_results.parquet"
# DEFAULT_TEACHER = r"C:\Users\danie\OneDrive\Desktop\recursive_caft\data\out\distillation\mmlu_synth_gptoss_b_t0_8.parquet"
# DEFAULT_OUT     = r"C:\Users\danie\OneDrive\Desktop\recursive_caft\src\core\entropy_dynamics\artifacts\dashboard.html"
# TOKENIZERS = {"qwen_3b": "Qwen/Qwen2.5-3B", "phi4_mini": "microsoft/Phi-4-mini-instruct"}
# WINDOW_SIZE = 32
# N_EXAMPLES_PER_CATEGORY = 4

# STATE_COLORS = {"CC": "#10b981", "UC": "#84cc16", "UW": "#f97316", "CW": "#dc2626"}
# STATE_NAMES = {
#     "CC": "Confident & Correct",
#     "UC": "Uncertain & Correct",
#     "UW": "Uncertain & Wrong",
#     "CW": "Confident & Wrong",
# }
# STATE_ORDER = ["CC", "UC", "UW", "CW"]


# # ── LOADING ─────────────────────────────────────────────────────────
# def load_results(path):
#     df = pd.read_parquet(path)
#     rn = {c: "H_s" for c in df.columns
#           if c.lower() in ("answer_entropy", "h_s", "entropy", "entropy_value")}
#     df = df.rename(columns=rn)
#     df["question_id"] = df["question_id"].astype(str)
#     df["k"] = df["k"].astype(int)
#     df["H_s"] = df["H_s"].astype(float)
#     df["student_correct"] = df["student_correct"].astype(bool)
#     if "student_answer" not in df.columns:
#         df["student_answer"] = ""
#     df["student_answer"] = df["student_answer"].astype(str).str.strip().str.lower()
#     return df


# def load_teacher(path):
#     if not path or not os.path.exists(path):
#         return None
#     raw = pd.read_parquet(path)
#     rows = []
#     for _, r in raw.iterrows():
#         inp, out = r["input"], r["output"]
#         if not isinstance(inp, dict) or not isinstance(out, dict):
#             continue
#         rows.append({
#             "question_id": str(inp.get("question_id", "")),
#             "question": inp.get("question", ""),
#             "options": inp.get("options", {}),
#             "gold": str(inp.get("gold", "")).lower(),
#             "teacher_cot": out.get("thinking", "") or out.get("raw_response", "") or "",
#             "teacher_answer": str(out.get("answer", "")).lower(),
#             "teacher_correct": bool(out.get("is_correct", False)),
#         })
#     return pd.DataFrame(rows)


# def maybe_tokenizer(model_id):
#     try:
#         from transformers import AutoTokenizer
#         return AutoTokenizer.from_pretrained(model_id)
#     except Exception:
#         return None


# # ── STATES ──────────────────────────────────────────────────────────
# def assign_states(df):
#     df = df.copy()
#     df["confident"] = False
#     thresholds = {}
#     for student, sub in df.groupby("student_label"):
#         thr = float(np.median(sub["H_s"]))
#         thresholds[student] = thr
#         df.loc[sub.index, "confident"] = sub["H_s"] < thr

#     def state_of(r):
#         if r["confident"] and r["student_correct"]:        return "CC"
#         if r["confident"] and not r["student_correct"]:    return "CW"
#         if not r["confident"] and r["student_correct"]:    return "UC"
#         return "UW"
#     df["state"] = df.apply(state_of, axis=1)
#     return df, thresholds


# # ── PER-QUESTION TRANSITIONS ────────────────────────────────────────
# def compute_transitions(df):
#     rows = []
#     for (student, qid), g in df.groupby(["student_label", "question_id"]):
#         g = g.sort_values("k").reset_index(drop=True)
#         K_total = int(g["k"].max())
#         s0 = g.iloc[0]["state"]
#         sF = g.iloc[-1]["state"]
#         cc = g[g["state"] == "CC"]
#         k_cc = int(cc.iloc[0]["k"]) if not cc.empty else None
#         wrong0 = s0 in ("UW", "CW")
#         right_after = (g["student_correct"]).any()
#         k_first_right = int(g[g["student_correct"]].iloc[0]["k"]) if right_after else None
#         rows.append({
#             "student": student,
#             "question_id": qid,
#             "K_total": K_total,
#             "tokens_total": K_total * WINDOW_SIZE,
#             "state_0": s0,
#             "state_final": sF,
#             "k_first_CC": k_cc,
#             "tokens_first_CC": k_cc * WINDOW_SIZE if k_cc is not None else None,
#             "frac_first_CC": k_cc / max(K_total, 1) if k_cc is not None else None,
#             "k_first_right": k_first_right,
#             "frac_first_right": k_first_right / max(K_total, 1) if k_first_right is not None else None,
#             "uw_to_cc_before_end": (s0 == "UW") and (k_cc is not None) and (k_cc < K_total),
#             "wrong0_to_right": wrong0 and right_after,
#             "right0_to_wrong_final": (s0 in ("CC", "UC")) and (sF in ("CW", "UW")),
#         })
#     return pd.DataFrame(rows)


# def compute_full_transition_matrix(df):
#     """Counts every state→state transition across consecutive chunks."""
#     out = {}
#     for student, sub in df.groupby("student_label"):
#         sub = sub.sort_values(["question_id", "k"])
#         prev_q, prev_state = None, None
#         mat = {f: {t: 0 for t in STATE_ORDER} for f in STATE_ORDER}
#         for _, row in sub.iterrows():
#             if row["question_id"] != prev_q:
#                 prev_q, prev_state = row["question_id"], row["state"]
#                 continue
#             mat[prev_state][row["state"]] += 1
#             prev_state = row["state"]
#         out[student] = mat
#     return out


# # ── PLOTLY CHARTS ───────────────────────────────────────────────────
# def fig_state_evolution(df):
#     students = sorted(df["student_label"].unique())
#     fig = make_subplots(rows=1, cols=len(students), subplot_titles=students,
#                         shared_yaxes=True)
#     for i, st in enumerate(students, 1):
#         sub = df[df["student_label"] == st]
#         pivot = sub.groupby(["k", "state"]).size().unstack(fill_value=0)
#         pivot = pivot.reindex(columns=STATE_ORDER, fill_value=0)
#         share = pivot.div(pivot.sum(axis=1), axis=0)
#         x = share.index.values * WINDOW_SIZE
#         for state in STATE_ORDER:
#             fig.add_trace(go.Scatter(
#                 x=x, y=share[state],
#                 stackgroup=f"s{i}", name=f"{state} — {STATE_NAMES[state]}",
#                 line=dict(width=0.5, color=STATE_COLORS[state]),
#                 hovertemplate=f"<b>{state}</b><br>tokens: %{{x}}<br>share: %{{y:.1%}}<extra></extra>",
#                 showlegend=(i == 1),
#             ), row=1, col=i)
#         fig.update_xaxes(title_text="Reasoning tokens", row=1, col=i)
#     fig.update_yaxes(title_text="Share of questions", row=1, col=1, range=[0, 1])
#     fig.update_layout(
#         title="State distribution as function of reasoning tokens",
#         height=460, hovermode="x unified",
#         legend=dict(orientation="h", y=-0.2),
#     )
#     return fig


# def fig_transition_heatmap(trans):
#     students = sorted(trans["student"].unique())
#     fig = make_subplots(rows=1, cols=len(students), subplot_titles=students)
#     for i, st in enumerate(students, 1):
#         sub = trans[trans["student"] == st]
#         flow = sub.groupby(["state_0", "state_final"]).size().unstack(fill_value=0)
#         flow = flow.reindex(index=STATE_ORDER, columns=STATE_ORDER, fill_value=0)
#         flow_pct = flow.div(flow.sum(axis=1), axis=0).fillna(0) * 100
#         text = [[f"{flow_pct.values[r,c]:.0f}%<br>n={flow.values[r,c]}"
#                  for c in range(4)] for r in range(4)]
#         fig.add_trace(go.Heatmap(
#             z=flow_pct.values, x=STATE_ORDER, y=STATE_ORDER, text=text,
#             texttemplate="%{text}", colorscale="Blues", zmin=0, zmax=100,
#             showscale=(i == len(students)),
#             hovertemplate="from %{y} → to %{x}<br>%{text}<extra></extra>",
#         ), row=1, col=i)
#         fig.update_xaxes(title_text="Final state", row=1, col=i)
#     fig.update_yaxes(title_text="Initial state (k=0)", row=1, col=1, autorange="reversed")
#     fig.update_layout(title="Initial → Final state transition matrix",
#                       height=480)
#     return fig


# def fig_uw_to_cc(trans):
#     students = sorted(trans["student"].unique())
#     fig = make_subplots(rows=2, cols=len(students),
#                         subplot_titles=[f"{s} — tokens" for s in students] +
#                                        [f"{s} — fraction of CoT" for s in students])
#     for i, st in enumerate(students, 1):
#         sub = trans[(trans["student"] == st) & (trans["state_0"] == "UW")
#                     & trans["uw_to_cc_before_end"]]
#         v = sub["tokens_first_CC"].dropna().values
#         if len(v):
#             fig.add_trace(go.Histogram(x=v, marker_color=STATE_COLORS["CC"],
#                                        nbinsx=30, showlegend=False,
#                                        hovertemplate="tokens: %{x}<br>n=%{y}<extra></extra>"),
#                           row=1, col=i)
#             fig.add_vline(x=float(np.median(v)), line_dash="dash", row=1, col=i,
#                           annotation_text=f"median={int(np.median(v))}")
#         v = sub["frac_first_CC"].dropna().values
#         if len(v):
#             fig.add_trace(go.Histogram(x=v, marker_color=STATE_COLORS["CC"],
#                                        nbinsx=20, showlegend=False,
#                                        hovertemplate="frac: %{x:.0%}<br>n=%{y}<extra></extra>"),
#                           row=2, col=i)
#             fig.add_vline(x=float(np.median(v)), line_dash="dash", row=2, col=i,
#                           annotation_text=f"median={np.median(v):.0%}")
#         fig.update_xaxes(title_text="Tokens", row=1, col=i)
#         fig.update_xaxes(title_text="Fraction of CoT", row=2, col=i, range=[0, 1])
#     fig.update_layout(title="UW → CC: tokens needed and as fraction of CoT",
#                       height=720, showlegend=False)
#     return fig


# def fig_per_question_heatmap(df, trans, max_q=200, max_k=30):
#     students = sorted(df["student_label"].unique())
#     fig = make_subplots(rows=1, cols=len(students), subplot_titles=students)
#     state_to_num = {"CC": 3, "UC": 2, "UW": 1, "CW": 0}
#     for i, st in enumerate(students, 1):
#         sub = df[df["student_label"] == st]
#         cv = trans[trans["student"] == st].sort_values("k_first_CC", na_position="last")
#         order = cv["question_id"].tolist()[:max_q]
#         wide = sub.pivot_table(index="question_id", columns="k", values="state",
#                                aggfunc="first")
#         wide = wide.reindex(order).iloc[:, :max_k]
#         z = wide.applymap(lambda s: state_to_num.get(s, np.nan)).values
#         fig.add_trace(go.Heatmap(
#             z=z, colorscale=[[0, STATE_COLORS["CW"]], [0.33, STATE_COLORS["UW"]],
#                              [0.66, STATE_COLORS["UC"]], [1, STATE_COLORS["CC"]]],
#             zmin=0, zmax=3, showscale=(i == len(students)),
#             colorbar=dict(tickvals=[0, 1, 2, 3], ticktext=["CW", "UW", "UC", "CC"]),
#             hovertemplate="step %{x}<br>question %{y}<extra></extra>",
#         ), row=1, col=i)
#         fig.update_xaxes(title_text="Step k", row=1, col=i)
#     fig.update_yaxes(title_text="Question (sorted by k_first_CC)", row=1, col=1)
#     fig.update_layout(title="Per-question state heatmap",
#                       height=620)
#     return fig


# # ── EXAMPLE TABLE ROWS ──────────────────────────────────────────────
# def chunk_with_tok(text, tok, w):
#     if tok is None:
#         words = text.split()
#         return [" ".join(words[i*w:(i+1)*w]) for i in range((len(words)+w-1)//w)]
#     ids = tok.encode(text, add_special_tokens=False)
#     return [tok.decode(ids[i:i+w], skip_special_tokens=True) for i in range(0, len(ids), w)]


# def build_examples_html(df, trans, teacher_df, students_tok):
#     """Pick examples per category, render HTML strings."""
#     if teacher_df is None:
#         return ""
#     sections = []
#     categories = [
#         ("Wrong → Right (UW/CW start, ever right)",
#          lambda s: s[s["state_0"].isin(["UW", "CW"]) & s["wrong0_to_right"]]),
#         ("Right → Wrong (CC/UC start, wrong final)",
#          lambda s: s[s["state_0"].isin(["CC", "UC"]) & s["right0_to_wrong_final"]]),
#         ("UW → CC before CoT ends",
#          lambda s: s[(s["state_0"] == "UW") & s["uw_to_cc_before_end"]].sort_values("k_first_CC")),
#         ("Stable CC throughout",
#          lambda s: s[(s["state_0"] == "CC") & (s["state_final"] == "CC")]),
#         ("Never converged",
#          lambda s: s[s["k_first_CC"].isna()]),
#     ]
#     for student in sorted(df["student_label"].unique()):
#         tok = students_tok.get(student)
#         sub_t = trans[trans["student"] == student].merge(
#             teacher_df, on="question_id", how="inner")
#         sections.append(f"<h2 style='margin-top:30px;'>{student}</h2>")
#         for cat_name, picker in categories:
#             picks = picker(sub_t).head(N_EXAMPLES_PER_CATEGORY)
#             if picks.empty:
#                 continue
#             sections.append(f"<h3>{cat_name} <span class='muted'>(showing {len(picks)})</span></h3>")
#             for _, meta in picks.iterrows():
#                 qid = meta["question_id"]
#                 rows = df[(df["student_label"] == student)
#                           & (df["question_id"] == qid)].sort_values("k").reset_index(drop=True)
#                 chunks = chunk_with_tok(str(meta["teacher_cot"]), tok, WINDOW_SIZE)
#                 gold = str(meta["gold"]).upper()
#                 tcol = STATE_COLORS["CC"] if meta["teacher_correct"] else STATE_COLORS["CW"]

#                 # Header
#                 header = (f"<div class='ex'>"
#                           f"<div class='ex-h'><b>Q{qid}</b> · gold=<b>{gold}</b> · "
#                           f"teacher answer=<b>{str(meta['teacher_answer']).upper()}</b> "
#                           f"<span class='dot' style='background:{tcol}'></span> · "
#                           f"start=<b>{meta['state_0']}</b> → final=<b>{meta['state_final']}</b> · "
#                           f"CoT length={int(meta['K_total'])} chunks "
#                           f"({int(meta['tokens_total'])} tokens)</div>")
#                 if pd.notna(meta["k_first_CC"]):
#                     kfc = int(meta["k_first_CC"])
#                     header += (f"<div class='muted'>First CC at chunk {kfc} "
#                                f"({kfc*WINDOW_SIZE} tokens, {kfc/max(meta['K_total'],1):.0%} of CoT)</div>")
#                 if pd.notna(meta["k_first_right"]):
#                     kfr = int(meta["k_first_right"])
#                     header += (f"<div class='muted'>First correct answer at chunk {kfr} "
#                                f"({kfr*WINDOW_SIZE} tokens, {kfr/max(meta['K_total'],1):.0%} of CoT)</div>")
#                 header += f"<div class='qtext'>{_esc(meta['question'])[:500]}</div>"

#                 # Per-chunk table
#                 table = ["<table class='chunktbl'><tr><th>k</th><th>tokens</th>"
#                          "<th>chunk text</th><th>answer</th><th>H_s</th><th>state</th></tr>"]
#                 for k_idx in range(len(rows)):
#                     row = rows.iloc[k_idx]
#                     state = row["state"]
#                     color = STATE_COLORS[state]
#                     chunk_text = chunks[k_idx-1] if 1 <= k_idx <= len(chunks) else "<em>(question only)</em>"
#                     chunk_text = _esc(chunk_text) if not chunk_text.startswith("<em>") else chunk_text
#                     ans = str(row["student_answer"]).upper() or "?"
#                     correct_mark = "✓" if row["student_correct"] else "✗"
#                     correct_color = STATE_COLORS["CC"] if row["student_correct"] else STATE_COLORS["CW"]
#                     bg = f"{color}22"
#                     table.append(
#                         f"<tr style='background:{bg}'>"
#                         f"<td>{k_idx}</td><td>{k_idx*WINDOW_SIZE}</td>"
#                         f"<td class='chk'>{chunk_text}</td>"
#                         f"<td><b>{ans}</b> <span style='color:{correct_color}'>{correct_mark}</span></td>"
#                         f"<td>{row['H_s']:.3f}</td>"
#                         f"<td><span class='stb' style='background:{color}'>{state}</span></td>"
#                         f"</tr>"
#                     )
#                 table.append("</table>")
#                 sections.append(header + "".join(table) + "</div>")
#     return "\n".join(sections)


# def _esc(s):
#     return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


# # ── REPORT TEXT ─────────────────────────────────────────────────────
# def build_report_text(trans, thresholds, full_trans):
#     L = []
#     L.append("STATE TRANSITION REPORT")
#     L.append("=" * 70)
#     L.append("\nDefinitions")
#     L.append("  H_s         Shannon entropy of the student's answer-token distribution.")
#     L.append("  Confident   H_s < per-student median. Per-student thresholds:")
#     for s, t in thresholds.items():
#         L.append(f"                {s}: {t:.3f}")
#     L.append("  States      CC=Confident&Correct, CW=Confident&Wrong,")
#     L.append("              UC=Uncertain&Correct, UW=Uncertain&Wrong\n")

#     for st in sorted(trans["student"].unique()):
#         s = trans[trans["student"] == st]
#         n = len(s)
#         L.append("─" * 70)
#         L.append(f"STUDENT: {st}  ({n} questions)")
#         L.append("─" * 70)
#         L.append("\nInitial state distribution (k=0):")
#         for state in STATE_ORDER:
#             ns = (s["state_0"] == state).sum()
#             L.append(f"  {state}: {ns:>5}  ({ns/n:.1%})")
#         L.append("\nFinal state distribution:")
#         for state in STATE_ORDER:
#             ns = (s["state_final"] == state).sum()
#             L.append(f"  {state}: {ns:>5}  ({ns/n:.1%})")

#         # UW → CC (the central success metric)
#         uw0 = s[s["state_0"] == "UW"]
#         succ = uw0[uw0["uw_to_cc_before_end"]]
#         L.append(f"\nUW → CC (the central success metric):")
#         L.append(f"  questions starting UW:                       {len(uw0):>5}")
#         L.append(f"  reached CC at any point:                     {uw0['k_first_CC'].notna().sum():>5}")
#         L.append(f"  reached CC strictly BEFORE CoT ended:        {len(succ):>5}  "
#                  f"({len(succ)/max(len(uw0),1):.1%} of UW)")
#         if len(succ):
#             L.append(f"  median tokens needed:                        {int(succ['tokens_first_CC'].median())}")
#             L.append(f"  median fraction of CoT:                      {succ['frac_first_CC'].median():.1%}")
#             L.append(f"  mean fraction of CoT:                        {succ['frac_first_CC'].mean():.1%}")

#         # First-correct (less strict)
#         wrong0 = s[s["state_0"].isin(["UW", "CW"])]
#         ever_right = wrong0[wrong0["wrong0_to_right"]]
#         L.append(f"\nWrong → Right (any confidence):")
#         L.append(f"  questions starting wrong:                    {len(wrong0):>5}")
#         L.append(f"  ever became right:                           {len(ever_right):>5}  "
#                  f"({len(ever_right)/max(len(wrong0),1):.1%})")
#         if len(ever_right):
#             L.append(f"  median tokens to first right answer:         {int(ever_right['k_first_right'].median()*WINDOW_SIZE)}")
#             L.append(f"  median fraction of CoT:                      {ever_right['frac_first_right'].median():.1%}")

#         # Degradation
#         good0 = s[s["state_0"].isin(["CC", "UC"])]
#         deg = good0[good0["right0_to_wrong_final"]]
#         L.append(f"\nRight → Wrong (degradation):")
#         L.append(f"  questions starting right:                    {len(good0):>5}")
#         L.append(f"  degraded to wrong by final step:             {len(deg):>5}  "
#                  f"({len(deg)/max(len(good0),1):.1%})")

#         # Full per-step transition counts
#         L.append(f"\nAll consecutive-chunk transitions (chunk k → chunk k+1):")
#         m = full_trans[st]
#         header_label = "from\\to"
#         L.append(f"  {header_label:>10}" + "".join(f"{t:>10}" for t in STATE_ORDER))
#         for f in STATE_ORDER:
#             L.append(f"  {f:>10}" + "".join(f"{m[f][t]:>10}" for t in STATE_ORDER))
#         L.append("")

#     return "\n".join(L)


# # ── DASHBOARD HTML ──────────────────────────────────────────────────
# def build_dashboard(df, trans, thresholds, teacher_df, students_tok, out_path):
#     full_trans = compute_full_transition_matrix(df)
#     report_text = build_report_text(trans, thresholds, full_trans)
#     examples_html = build_examples_html(df, trans, teacher_df, students_tok)

#     fig1 = fig_state_evolution(df)
#     fig2 = fig_transition_heatmap(trans)
#     fig3 = fig_uw_to_cc(trans)
#     fig4 = fig_per_question_heatmap(df, trans)

#     plots_html = (
#         '<div id="p1"></div><div id="p2"></div>'
#         '<div id="p3"></div><div id="p4"></div>'
#     )

#     plot_scripts = (
#         f"Plotly.newPlot('p1', {pio.to_json(fig1)});"
#         f"Plotly.newPlot('p2', {pio.to_json(fig2)});"
#         f"Plotly.newPlot('p3', {pio.to_json(fig3)});"
#         f"Plotly.newPlot('p4', {pio.to_json(fig4)});"
#     )
#     plot_scripts = (
#         "var f1=" + pio.to_json(fig1) + ";Plotly.newPlot('p1',f1.data,f1.layout);"
#         "var f2=" + pio.to_json(fig2) + ";Plotly.newPlot('p2',f2.data,f2.layout);"
#         "var f3=" + pio.to_json(fig3) + ";Plotly.newPlot('p3',f3.data,f3.layout);"
#         "var f4=" + pio.to_json(fig4) + ";Plotly.newPlot('p4',f4.data,f4.layout);"
#     )

#     # Overview cards
#     cards = []
#     for st in sorted(trans["student"].unique()):
#         s = trans[trans["student"] == st]
#         n = len(s)
#         uw0 = s[s["state_0"] == "UW"]
#         succ = uw0[uw0["uw_to_cc_before_end"]]
#         deg = s[s["right0_to_wrong_final"]]
#         med_frac = succ["frac_first_CC"].median() if len(succ) else None
#         cards.append(f"""
#         <div class='card'>
#           <h3>{st}</h3>
#           <table class='kv'>
#             <tr><td>questions</td><td>{n}</td></tr>
#             <tr><td>start UW</td><td>{len(uw0)} ({len(uw0)/n:.0%})</td></tr>
#             <tr><td>UW → CC before end</td><td>{len(succ)} ({len(succ)/max(len(uw0),1):.0%} of UW)</td></tr>
#             <tr><td>median frac of CoT for UW→CC</td><td>{med_frac:.0%}</td></tr>
#             <tr><td>degraded right→wrong</td><td>{len(deg)} ({len(deg)/n:.0%})</td></tr>
#           </table>
#         </div>
#         """ if med_frac is not None else f"""
#         <div class='card'><h3>{st}</h3>
#           <table class='kv'>
#             <tr><td>questions</td><td>{n}</td></tr>
#             <tr><td>start UW</td><td>{len(uw0)}</td></tr>
#             <tr><td>UW → CC before end</td><td>{len(succ)}</td></tr>
#             <tr><td>degraded right→wrong</td><td>{len(deg)}</td></tr>
#           </table>
#         </div>
#         """)
#     cards_html = "<div class='cards'>" + "".join(cards) + "</div>"

#     html = f"""<!DOCTYPE html><html><head><meta charset='utf-8'>
# <title>Progressive Entropy Dashboard</title>
# <script src='https://cdn.plot.ly/plotly-latest.min.js'></script>
# <style>
# body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
# margin:0;background:#f5f5f7;color:#1d1d1f;}}
# .wrap{{max-width:1300px;margin:0 auto;padding:20px;}}
# h1{{font-size:30px;margin:10px 0;}}
# .tabs{{display:flex;gap:5px;border-bottom:2px solid #e5e7eb;margin:20px 0;}}
# .tab{{padding:12px 24px;cursor:pointer;background:#fff;border-radius:8px 8px 0 0;
# font-weight:600;color:#666;border:1px solid #e5e7eb;border-bottom:none;}}
# .tab.active{{background:#0071e3;color:white;}}
# .panel{{display:none;background:#fff;border-radius:0 12px 12px 12px;padding:25px;
# box-shadow:0 2px 6px rgba(0,0,0,0.06);}}
# .panel.active{{display:block;}}
# .cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:15px;margin:20px 0;}}
# .card{{background:#fafafa;border-radius:10px;padding:18px;border:1px solid #eee;}}
# .card h3{{margin-top:0;color:#0071e3;}}
# .kv{{width:100%;font-size:13px;}}
# .kv td{{padding:4px 0;}}
# .kv td:last-child{{text-align:right;font-weight:600;}}
# .def{{background:#f0f9ff;border-left:4px solid #0071e3;padding:15px;border-radius:6px;margin:15px 0;}}
# .def b{{color:#0071e3;}}
# pre{{background:#1d1d1f;color:#e5e5e7;padding:20px;border-radius:8px;
# overflow:auto;font-size:12px;line-height:1.5;}}
# .ex{{background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:18px;margin:18px 0;}}
# .ex-h{{font-size:13px;}}
# .qtext{{font-size:13px;color:#444;margin:10px 0;padding:10px;background:#fafafa;border-radius:6px;}}
# .muted{{font-size:11px;color:#666;}}
# .dot{{display:inline-block;width:10px;height:10px;border-radius:50%;}}
# .chunktbl{{width:100%;border-collapse:collapse;font-size:12px;margin-top:10px;}}
# .chunktbl th{{background:#1d1d1f;color:white;padding:8px;text-align:left;}}
# .chunktbl td{{padding:8px;border-bottom:1px solid #eee;vertical-align:top;}}
# .chk{{font-family:'SF Mono',Menlo,monospace;font-size:11px;line-height:1.5;max-width:550px;}}
# .stb{{display:inline-block;padding:3px 8px;border-radius:5px;color:white;
# font-weight:bold;font-size:11px;}}
# </style></head><body><div class='wrap'>
# <h1>Progressive entropy — interactive dashboard</h1>

# <div class='tabs'>
# <div class='tab active' onclick='show(0)'>Overview</div>
# <div class='tab' onclick='show(1)'>Statistics</div>
# <div class='tab' onclick='show(2)'>Examples</div>
# <div class='tab' onclick='show(3)'>Plots</div>
# </div>

# <div class='panel active'>
# <h2>Key numbers</h2>
# {cards_html}
# <div class='def'>
# <b>States:</b> CC = Confident & Correct, UC = Uncertain & Correct,
# UW = Uncertain & Wrong, CW = Confident & Wrong.
# A step is "confident" if H_s is below the per-student median.
# <br><br>
# <b>Central success metric (UW → CC before CoT end):</b>
# the question started uncertain-wrong, AND the student reached confident-correct
# strictly before the teacher CoT ran out. This is the only case where we can
# say reasoning was actually informative — if the student only reaches CC at
# the very last chunk, we can't tell if reasoning helped or it was just final answer.
# <br><br>
# <b>Degradation (right → wrong):</b> the student started right (CC or UC) but
# ended wrong (CW or UW). These are cases where extra reasoning hurt.
# </div>
# </div>

# <div class='panel'>
# <h2>Statistics report</h2>
# <pre>{_esc(report_text)}</pre>
# </div>

# <div class='panel'>
# <h2>Examples</h2>
# <p class='muted'>Each row of the table is one chunk of 32 reasoning tokens.
# Background color = state after the chunk. Shows the chunk text, the model's
# answer letter, the correctness mark, the H_s value and the state code.</p>
# {examples_html}
# </div>

# <div class='panel'>
# <h2>Plots (interactive)</h2>
# {plots_html}
# </div>

# </div>
# <script>
# function show(i){{
#   document.querySelectorAll('.tab').forEach((t,j)=>t.classList.toggle('active',i===j));
#   document.querySelectorAll('.panel').forEach((p,j)=>p.classList.toggle('active',i===j));
#   if(i===3){{window.dispatchEvent(new Event('resize'));}}
# }}
# {plot_scripts}
# </script>
# </body></html>"""
#     out_path.write_text(html, encoding="utf-8")
#     print(f"  saved: {out_path}")


# # ── MAIN ────────────────────────────────────────────────────────────
# def main():
#     p = argparse.ArgumentParser()
#     p.add_argument("--results", default=DEFAULT_RESULTS)
#     p.add_argument("--teacher", default=DEFAULT_TEACHER)
#     p.add_argument("--out", default=DEFAULT_OUT)
#     args = p.parse_args()
#     out = Path(args.out)
#     out.parent.mkdir(parents=True, exist_ok=True)

#     print("[1] Loading results…")
#     df = load_results(args.results)
#     print(f"   {len(df)} measurements, {df['question_id'].nunique()} questions")

#     print("[2] Assigning states…")
#     df, thresholds = assign_states(df)

#     print("[3] Computing transitions…")
#     trans = compute_transitions(df)

#     print("[4] Loading teacher CoT…")
#     teacher_df = load_teacher(args.teacher)

#     print("[5] Loading tokenizers for chunking…")
#     students_tok = {}
#     for st in df["student_label"].unique():
#         students_tok[st] = maybe_tokenizer(TOKENIZERS.get(st))

#     print("[6] Building dashboard…")
#     build_dashboard(df, trans, thresholds, teacher_df, students_tok, out)
#     print(f"\n✓ Open in a browser: {out}")


# if __name__ == "__main__":
#     main()

"""
Progressive Entropy — Interactive HTML Dashboard
==================================================
Builds ONE self-contained HTML file with four tabs:
  1. Overview      — key numbers, definitions, transition matrix
  2. Statistics    — full text report
  3. Examples      — colored CoT chunks with per-chunk answer/H_s/state
  4. Plots         — interactive Plotly charts (state evolution, transitions,
                     UW→CC distribution, per-question heatmap)

Run from VS Code Run button. Open the resulting HTML in any browser.
"""
import argparse, os, json, base64
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# ── PATHS ───────────────────────────────────────────────────────────
DEFAULT_RESULTS = r"C:\Users\danie\OneDrive\Desktop\recursive_caft\src\core\entropy_dynamics\artifacts\entropy_dynamics_results_b.parquet"
DEFAULT_TEACHER = r"C:\Users\danie\OneDrive\Desktop\recursive_caft\data\out\distillation\mmlu_synth_gptoss_b_t0_8.parquet"
DEFAULT_OUT     = r"C:\Users\danie\OneDrive\Desktop\recursive_caft\src\core\entropy_dynamics\artifacts\dashboard.html"
TOKENIZERS = {"qwen_3b": "Qwen/Qwen2.5-3B", "phi4_mini": "microsoft/Phi-4-mini-instruct"}
WINDOW_SIZE = 32
N_EXAMPLES_PER_CATEGORY = 4

STATE_COLORS = {"CC": "#10b981", "UC": "#84cc16", "UW": "#f97316", "CW": "#dc2626"}
STATE_NAMES = {
    "CC": "Confident & Correct",
    "UC": "Uncertain & Correct",
    "UW": "Uncertain & Wrong",
    "CW": "Confident & Wrong",
}
STATE_ORDER = ["CC", "UC", "UW", "CW"]


# ── LOADING ─────────────────────────────────────────────────────────
def load_results(path):
    df = pd.read_parquet(path)
    rn = {c: "H_s" for c in df.columns
          if c.lower() in ("answer_entropy", "h_s", "entropy", "entropy_value")}
    df = df.rename(columns=rn)
    df["question_id"] = df["question_id"].astype(str)
    df["k"] = df["k"].astype(int)
    df["H_s"] = df["H_s"].astype(float)
    df["student_correct"] = df["student_correct"].astype(bool)
    if "student_answer" not in df.columns:
        df["student_answer"] = ""
    df["student_answer"] = df["student_answer"].astype(str).str.strip().str.lower()
    return df


def load_teacher(path):
    if not path or not os.path.exists(path):
        return None
    raw = pd.read_parquet(path)
    rows = []
    for _, r in raw.iterrows():
        inp, out = r["input"], r["output"]
        if not isinstance(inp, dict) or not isinstance(out, dict):
            continue
        rows.append({
            "question_id": str(inp.get("question_id", "")),
            "question": inp.get("question", ""),
            "options": inp.get("options", {}),
            "gold": str(inp.get("gold", "")).lower(),
            "teacher_cot": out.get("thinking", "") or out.get("raw_response", "") or "",
            "teacher_answer": str(out.get("answer", "")).lower(),
            "teacher_correct": bool(out.get("is_correct", False)),
        })
    return pd.DataFrame(rows)


def maybe_tokenizer(model_id):
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(model_id)
    except Exception:
        return None


# ── STATES ──────────────────────────────────────────────────────────
def assign_states(df):
    df = df.copy()
    df["confident"] = False
    thresholds = {}
    for student, sub in df.groupby("student_label"):
        thr = float(np.median(sub["H_s"]))
        thresholds[student] = thr
        df.loc[sub.index, "confident"] = sub["H_s"] < thr

    def state_of(r):
        if r["confident"] and r["student_correct"]:        return "CC"
        if r["confident"] and not r["student_correct"]:    return "CW"
        if not r["confident"] and r["student_correct"]:    return "UC"
        return "UW"
    df["state"] = df.apply(state_of, axis=1)
    return df, thresholds


# ── PER-QUESTION TRANSITIONS ────────────────────────────────────────
def compute_transitions(df):
    rows = []
    for (student, qid), g in df.groupby(["student_label", "question_id"]):
        g = g.sort_values("k").reset_index(drop=True)
        K_total = int(g["k"].max())
        s0 = g.iloc[0]["state"]
        sF = g.iloc[-1]["state"]
        cc = g[g["state"] == "CC"]
        k_cc = int(cc.iloc[0]["k"]) if not cc.empty else None
        wrong0 = s0 in ("UW", "CW")
        right_after = (g["student_correct"]).any()
        k_first_right = int(g[g["student_correct"]].iloc[0]["k"]) if right_after else None
        rows.append({
            "student": student,
            "question_id": qid,
            "K_total": K_total,
            "tokens_total": K_total * WINDOW_SIZE,
            "state_0": s0,
            "state_final": sF,
            "k_first_CC": k_cc,
            "tokens_first_CC": k_cc * WINDOW_SIZE if k_cc is not None else None,
            "frac_first_CC": k_cc / max(K_total, 1) if k_cc is not None else None,
            "k_first_right": k_first_right,
            "frac_first_right": k_first_right / max(K_total, 1) if k_first_right is not None else None,
            "uw_to_cc_before_end": (s0 == "UW") and (k_cc is not None) and (k_cc < K_total),
            "wrong0_to_right": wrong0 and right_after,
            "right0_to_wrong_final": (s0 in ("CC", "UC")) and (sF in ("CW", "UW")),
        })
    return pd.DataFrame(rows)


def compute_full_transition_matrix(df):
    """Counts every state→state transition across consecutive chunks."""
    out = {}
    for student, sub in df.groupby("student_label"):
        sub = sub.sort_values(["question_id", "k"])
        prev_q, prev_state = None, None
        mat = {f: {t: 0 for t in STATE_ORDER} for f in STATE_ORDER}
        for _, row in sub.iterrows():
            if row["question_id"] != prev_q:
                prev_q, prev_state = row["question_id"], row["state"]
                continue
            mat[prev_state][row["state"]] += 1
            prev_state = row["state"]
        out[student] = mat
    return out


# ── PLOTLY CHARTS ───────────────────────────────────────────────────
def fig_state_evolution(df, n_bins=20):
    """
    State distribution as a function of relative CoT progress.
    x-axis = percent of each question's reasoning length (0–100%).
    """
    students = sorted(df["student_label"].unique())
    fig = make_subplots(
        rows=1, cols=len(students),
        subplot_titles=students,
        shared_yaxes=True
    )

    for i, st in enumerate(students, 1):
        sub = df[df["student_label"] == st].copy()

        # Per-question relative progress in [0, 1]
        sub["K_total"] = sub.groupby("question_id")["k"].transform("max")
        sub["progress"] = np.where(sub["K_total"] > 0, sub["k"] / sub["K_total"], 0.0)
        sub["bin"] = np.floor(sub["progress"] * n_bins).astype(int)
        sub.loc[sub["bin"] == n_bins, "bin"] = n_bins - 1  # just in case progress == 1.0

        pivot = (
            sub.groupby(["bin", "state"])
               .size()
               .unstack(fill_value=0)
               .reindex(columns=STATE_ORDER, fill_value=0)
               .reindex(range(n_bins), fill_value=0)
        )
        share = pivot.div(pivot.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
        x = (share.index.values + 0.5) * (100 / n_bins)

        for state in STATE_ORDER:
            fig.add_trace(go.Scatter(
                x=x,
                y=share[state],
                stackgroup=f"s{i}",
                name=f"{state} — {STATE_NAMES[state]}",
                line=dict(width=0.5, color=STATE_COLORS[state]),
                hovertemplate=(
                    f"<b>{state}</b><br>"
                    f"progress: %{{x:.0f}}%<br>"
                    f"share: %{{y:.1%}}<extra></extra>"
                ),
                showlegend=(i == 1),
            ), row=1, col=i)

        fig.update_xaxes(title_text="% of CoT", range=[0, 100], row=1, col=i)

    fig.update_yaxes(title_text="Share of questions", row=1, col=1, range=[0, 1])
    fig.update_layout(
        title="State distribution as a function of CoT progress",
        height=460,
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.2),
    )
    return fig


def fig_transition_heatmap(trans):
    students = sorted(trans["student"].unique())
    fig = make_subplots(rows=1, cols=len(students), subplot_titles=students)
    for i, st in enumerate(students, 1):
        sub = trans[trans["student"] == st]
        flow = sub.groupby(["state_0", "state_final"]).size().unstack(fill_value=0)
        flow = flow.reindex(index=STATE_ORDER, columns=STATE_ORDER, fill_value=0)
        flow_pct = flow.div(flow.sum(axis=1), axis=0).fillna(0) * 100
        text = [[f"{flow_pct.values[r,c]:.0f}%<br>n={flow.values[r,c]}"
                 for c in range(4)] for r in range(4)]
        fig.add_trace(go.Heatmap(
            z=flow_pct.values, x=STATE_ORDER, y=STATE_ORDER, text=text,
            texttemplate="%{text}", colorscale="Blues", zmin=0, zmax=100,
            showscale=(i == len(students)),
            hovertemplate="from %{y} → to %{x}<br>%{text}<extra></extra>",
        ), row=1, col=i)
        fig.update_xaxes(title_text="Final state", row=1, col=i)
    fig.update_yaxes(title_text="Initial state (k=0)", row=1, col=1, autorange="reversed")
    fig.update_layout(title="Initial → Final state transition matrix",
                      height=480)
    return fig


def fig_uw_to_cc(trans):
    students = sorted(trans["student"].unique())
    fig = make_subplots(rows=2, cols=len(students),
                        subplot_titles=[f"{s} — tokens" for s in students] +
                                       [f"{s} — fraction of CoT" for s in students])
    for i, st in enumerate(students, 1):
        sub = trans[(trans["student"] == st) & (trans["state_0"] == "UW")
                    & trans["uw_to_cc_before_end"]]
        v = sub["tokens_first_CC"].dropna().values
        if len(v):
            fig.add_trace(go.Histogram(x=v, marker_color=STATE_COLORS["CC"],
                                       nbinsx=30, showlegend=False,
                                       hovertemplate="tokens: %{x}<br>n=%{y}<extra></extra>"),
                          row=1, col=i)
            fig.add_vline(x=float(np.median(v)), line_dash="dash", row=1, col=i,
                          annotation_text=f"median={int(np.median(v))}")
        v = sub["frac_first_CC"].dropna().values
        if len(v):
            fig.add_trace(go.Histogram(x=v, marker_color=STATE_COLORS["CC"],
                                       nbinsx=20, showlegend=False,
                                       hovertemplate="frac: %{x:.0%}<br>n=%{y}<extra></extra>"),
                          row=2, col=i)
            fig.add_vline(x=float(np.median(v)), line_dash="dash", row=2, col=i,
                          annotation_text=f"median={np.median(v):.0%}")
        fig.update_xaxes(title_text="Tokens", row=1, col=i)
        fig.update_xaxes(title_text="Fraction of CoT", row=2, col=i, range=[0, 1])
    fig.update_layout(title="UW → CC: tokens needed and as fraction of CoT",
                      height=720, showlegend=False)
    return fig


def fig_per_question_heatmap(df, trans, max_q=200, max_k=30):
    """One row per student, four columns by initial state (CC/UC/UW/CW)."""
    students = sorted(df["student_label"].unique())
    fig = make_subplots(
        rows=len(students), cols=4,
        subplot_titles=[f"{st} — start={s}" for st in students for s in STATE_ORDER],
        vertical_spacing=0.10, horizontal_spacing=0.04,
    )
    state_to_num = {"CW": 0, "UW": 1, "UC": 2, "CC": 3}
    for i, st in enumerate(students, 1):
        sub = df[df["student_label"] == st]
        cv = trans[trans["student"] == st]
        for j, init_state in enumerate(STATE_ORDER, 1):
            ids = cv[cv["state_0"] == init_state].sort_values(
                "k_first_CC", na_position="last")["question_id"].tolist()[:max_q]
            if not ids:
                continue
            wide = sub[sub["question_id"].isin(ids)].pivot_table(
                index="question_id", columns="k", values="state", aggfunc="first")
            wide = wide.reindex(ids).iloc[:, :max_k]
            z = wide.applymap(lambda s: state_to_num.get(s, np.nan)).values
            fig.add_trace(go.Heatmap(
                z=z,
                colorscale=[[0, STATE_COLORS["CW"]], [0.33, STATE_COLORS["UW"]],
                            [0.66, STATE_COLORS["UC"]], [1, STATE_COLORS["CC"]]],
                zmin=0, zmax=3,
                showscale=(i == 1 and j == 4),
                colorbar=dict(tickvals=[0, 1, 2, 3], ticktext=["CW", "UW", "UC", "CC"]),
                hovertemplate="step %{x}<br>question %{y}<extra></extra>",
            ), row=i, col=j)
            fig.update_xaxes(title_text="step k" if i == len(students) else "",
                             row=i, col=j)
    fig.update_layout(
        title="Per-question heatmap, split by initial state at k=0",
        height=420 * len(students),
    )
    return fig


def hex_to_rgba(hex_color, alpha=0.33):
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def fig_sankey(trans):
    """Sankey: initial state → final state, one per student."""
    students = sorted(trans["student"].unique())
    fig = make_subplots(
        rows=1, cols=len(students),
        specs=[[{"type": "sankey"}] * len(students)],
        subplot_titles=students,
    )
    for i, st in enumerate(students, 1):
        sub = trans[trans["student"] == st]
        flow = sub.groupby(["state_0", "state_final"]).size().reset_index(name="n")
        labels = [f"{s} (start)" for s in STATE_ORDER] + [f"{s} (final)" for s in STATE_ORDER]
        node_colors = [STATE_COLORS[s] for s in STATE_ORDER] * 2
        idx_init = {s: k for k, s in enumerate(STATE_ORDER)}
        idx_fin = {s: k + 4 for k, s in enumerate(STATE_ORDER)}
        sources, targets, values, link_colors = [], [], [], []
        for _, r in flow.iterrows():
            sources.append(idx_init[r["state_0"]])
            targets.append(idx_fin[r["state_final"]])
            values.append(int(r["n"]))
            link_colors.append(hex_to_rgba(STATE_COLORS[r["state_0"]], 0.33))

        fig.add_trace(go.Sankey(
            node=dict(label=labels, color=node_colors, pad=20, thickness=18),
            link=dict(source=sources, target=targets, value=values, color=link_colors),
        ), row=1, col=i)

    fig.update_layout(title="Sankey: initial state → final state",
                      height=520, font_size=11)
    return fig

# ── EXAMPLE TABLE ROWS ──────────────────────────────────────────────
def chunk_with_tok(text, tok, w):
    if tok is None:
        words = text.split()
        return [" ".join(words[i*w:(i+1)*w]) for i in range((len(words)+w-1)//w)]
    ids = tok.encode(text, add_special_tokens=False)
    return [tok.decode(ids[i:i+w], skip_special_tokens=True) for i in range(0, len(ids), w)]


def build_examples_html(df, trans, teacher_df, students_tok):
    """Pick examples per category, render HTML strings."""
    if teacher_df is None:
        return ""
    sections = []
    categories = [
        ("Wrong → Right (UW/CW start, ever right)",
         lambda s: s[s["state_0"].isin(["UW", "CW"]) & s["wrong0_to_right"]]),
        ("Right → Wrong (CC/UC start, wrong final)",
         lambda s: s[s["state_0"].isin(["CC", "UC"]) & s["right0_to_wrong_final"]]),
        ("UW → CC before CoT ends",
         lambda s: s[(s["state_0"] == "UW") & s["uw_to_cc_before_end"]].sort_values("k_first_CC")),
        ("Stable CC throughout",
         lambda s: s[(s["state_0"] == "CC") & (s["state_final"] == "CC")]),
        ("Never converged",
         lambda s: s[s["k_first_CC"].isna()]),
    ]
    for student in sorted(df["student_label"].unique()):
        tok = students_tok.get(student)
        sub_t = trans[trans["student"] == student].merge(
            teacher_df, on="question_id", how="inner")
        sections.append(f"<h2 style='margin-top:30px;'>{student}</h2>")
        for cat_name, picker in categories:
            picks = picker(sub_t).head(N_EXAMPLES_PER_CATEGORY)
            if picks.empty:
                continue
            sections.append(f"<h3>{cat_name} <span class='muted'>(showing {len(picks)})</span></h3>")
            for _, meta in picks.iterrows():
                qid = meta["question_id"]
                rows = df[(df["student_label"] == student)
                          & (df["question_id"] == qid)].sort_values("k").reset_index(drop=True)
                chunks = chunk_with_tok(str(meta["teacher_cot"]), tok, WINDOW_SIZE)
                gold = str(meta["gold"]).upper()
                tcol = STATE_COLORS["CC"] if meta["teacher_correct"] else STATE_COLORS["CW"]

                # Header
                header = (f"<div class='ex'>"
                          f"<div class='ex-h'><b>Q{qid}</b> · gold=<b>{gold}</b> · "
                          f"teacher answer=<b>{str(meta['teacher_answer']).upper()}</b> "
                          f"<span class='dot' style='background:{tcol}'></span> · "
                          f"start=<b>{meta['state_0']}</b> → final=<b>{meta['state_final']}</b> · "
                          f"CoT length={int(meta['K_total'])} chunks "
                          f"({int(meta['tokens_total'])} tokens)</div>")
                if pd.notna(meta["k_first_CC"]):
                    kfc = int(meta["k_first_CC"])
                    header += (f"<div class='muted'>First CC at chunk {kfc} "
                               f"({kfc*WINDOW_SIZE} tokens, {kfc/max(meta['K_total'],1):.0%} of CoT)</div>")
                if pd.notna(meta["k_first_right"]):
                    kfr = int(meta["k_first_right"])
                    header += (f"<div class='muted'>First correct answer at chunk {kfr} "
                               f"({kfr*WINDOW_SIZE} tokens, {kfr/max(meta['K_total'],1):.0%} of CoT)</div>")
                header += f"<div class='qtext'>{_esc(meta['question'])[:500]}</div>"

                # Per-chunk table
                table = ["<table class='chunktbl'><tr><th>k</th><th>tokens</th>"
                         "<th>chunk text</th><th>answer</th><th>H_s</th><th>state</th></tr>"]
                for k_idx in range(len(rows)):
                    row = rows.iloc[k_idx]
                    state = row["state"]
                    color = STATE_COLORS[state]
                    chunk_text = chunks[k_idx-1] if 1 <= k_idx <= len(chunks) else "<em>(question only)</em>"
                    chunk_text = _esc(chunk_text) if not chunk_text.startswith("<em>") else chunk_text
                    ans = str(row["student_answer"]).upper() or "?"
                    correct_mark = "✓" if row["student_correct"] else "✗"
                    correct_color = STATE_COLORS["CC"] if row["student_correct"] else STATE_COLORS["CW"]
                    bg = f"{color}22"
                    table.append(
                        f"<tr style='background:{bg}'>"
                        f"<td>{k_idx}</td><td>{k_idx*WINDOW_SIZE}</td>"
                        f"<td class='chk'>{chunk_text}</td>"
                        f"<td><b>{ans}</b> <span style='color:{correct_color}'>{correct_mark}</span></td>"
                        f"<td>{row['H_s']:.3f}</td>"
                        f"<td><span class='stb' style='background:{color}'>{state}</span></td>"
                        f"</tr>"
                    )
                table.append("</table>")
                sections.append(header + "".join(table) + "</div>")
    return "\n".join(sections)


def _esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


# ── REPORT TEXT ─────────────────────────────────────────────────────
def build_report_text(trans, thresholds, full_trans):
    L = []
    L.append("STATE TRANSITION REPORT")
    L.append("=" * 70)
    L.append("\nDefinitions")
    L.append("  H_s         Shannon entropy of the student's answer-token distribution.")
    L.append("  Confident   H_s < per-student median. Per-student thresholds:")
    for s, t in thresholds.items():
        L.append(f"                {s}: {t:.3f}")
    L.append("  States      CC=Confident&Correct, CW=Confident&Wrong,")
    L.append("              UC=Uncertain&Correct, UW=Uncertain&Wrong\n")

    for st in sorted(trans["student"].unique()):
        s = trans[trans["student"] == st]
        n = len(s)
        L.append("─" * 70)
        L.append(f"STUDENT: {st}  ({n} questions)")
        L.append("─" * 70)
        L.append("\nInitial state distribution (k=0):")
        for state in STATE_ORDER:
            ns = (s["state_0"] == state).sum()
            L.append(f"  {state}: {ns:>5}  ({ns/n:.1%})")
        L.append("\nFinal state distribution:")
        for state in STATE_ORDER:
            ns = (s["state_final"] == state).sum()
            L.append(f"  {state}: {ns:>5}  ({ns/n:.1%})")

        # UW → CC (the central success metric)
        uw0 = s[s["state_0"] == "UW"]
        succ = uw0[uw0["uw_to_cc_before_end"]]
        L.append(f"\nUW → CC (the central success metric):")
        L.append(f"  questions starting UW:                       {len(uw0):>5}")
        L.append(f"  reached CC at any point:                     {uw0['k_first_CC'].notna().sum():>5}")
        L.append(f"  reached CC strictly BEFORE CoT ended:        {len(succ):>5}  "
                 f"({len(succ)/max(len(uw0),1):.1%} of UW)")
        if len(succ):
            L.append(f"  median tokens needed:                        {int(succ['tokens_first_CC'].median())}")
            L.append(f"  median fraction of CoT:                      {succ['frac_first_CC'].median():.1%}")
            L.append(f"  mean fraction of CoT:                        {succ['frac_first_CC'].mean():.1%}")

        # First-correct (less strict)
        wrong0 = s[s["state_0"].isin(["UW", "CW"])]
        ever_right = wrong0[wrong0["wrong0_to_right"]]
        L.append(f"\nWrong → Right (any confidence):")
        L.append(f"  questions starting wrong:                    {len(wrong0):>5}")
        L.append(f"  ever became right:                           {len(ever_right):>5}  "
                 f"({len(ever_right)/max(len(wrong0),1):.1%})")
        if len(ever_right):
            L.append(f"  median tokens to first right answer:         {int(ever_right['k_first_right'].median()*WINDOW_SIZE)}")
            L.append(f"  median fraction of CoT:                      {ever_right['frac_first_right'].median():.1%}")

        # Degradation
        good0 = s[s["state_0"].isin(["CC", "UC"])]
        deg = good0[good0["right0_to_wrong_final"]]
        L.append(f"\nRight → Wrong (degradation):")
        L.append(f"  questions starting right:                    {len(good0):>5}")
        L.append(f"  degraded to wrong by final step:             {len(deg):>5}  "
                 f"({len(deg)/max(len(good0),1):.1%})")

        # Full per-step transition counts
        L.append(f"\nAll consecutive-chunk transitions (chunk k → chunk k+1):")
        m = full_trans[st]
        header_label = "from\\to"
        L.append(f"  {header_label:>10}" + "".join(f"{t:>10}" for t in STATE_ORDER))
        for f in STATE_ORDER:
            L.append(f"  {f:>10}" + "".join(f"{m[f][t]:>10}" for t in STATE_ORDER))
        L.append("")

    return "\n".join(L)


# ── DASHBOARD HTML ──────────────────────────────────────────────────
def build_dashboard(df, trans, thresholds, teacher_df, students_tok, out_path):
    full_trans = compute_full_transition_matrix(df)
    report_text = build_report_text(trans, thresholds, full_trans)
    examples_html = build_examples_html(df, trans, teacher_df, students_tok)

    #fig1 = fig_state_evolution(df)
    #fig2 = fig_transition_heatmap(trans)
    fig3 = fig_uw_to_cc(trans)
    #fig4 = fig_per_question_heatmap(df, trans)
    fig5 = fig_sankey(trans)

    plots_html = (
        '<div id="p1"></div><div id="p2"></div><div id="p5"></div>'
        '<div id="p3"></div><div id="p4"></div>'
    )

    plot_scripts = (
   #     "var f1=" + pio.to_json(fig1) + ";Plotly.newPlot('p1',f1.data,f1.layout);"
   #     "var f2=" + pio.to_json(fig2) + ";Plotly.newPlot('p2',f2.data,f2.layout);"
         "var f3=" + pio.to_json(fig3) + ";Plotly.newPlot('p3',f3.data,f3.layout);"
   #     "var f4=" + pio.to_json(fig4) + ";Plotly.newPlot('p4',f4.data,f4.layout);"
        "var f5=" + pio.to_json(fig5) + ";Plotly.newPlot('p5',f5.data,f5.layout);"
    )

    # Overview cards
    cards = []
    for st in sorted(trans["student"].unique()):
        s = trans[trans["student"] == st]
        n = len(s)
        uw0 = s[s["state_0"] == "UW"]
        succ = uw0[uw0["uw_to_cc_before_end"]]
        deg = s[s["right0_to_wrong_final"]]
        med_frac = succ["frac_first_CC"].median() if len(succ) else None
        cards.append(f"""
        <div class='card'>
          <h3>{st}</h3>
          <table class='kv'>
            <tr><td>questions</td><td>{n}</td></tr>
            <tr><td>start UW</td><td>{len(uw0)} ({len(uw0)/n:.0%})</td></tr>
            <tr><td>UW → CC before end</td><td>{len(succ)} ({len(succ)/max(len(uw0),1):.0%} of UW)</td></tr>
            <tr><td>median frac of CoT for UW→CC</td><td>{med_frac:.0%}</td></tr>
            <tr><td>degraded right→wrong</td><td>{len(deg)} ({len(deg)/n:.0%})</td></tr>
          </table>
        </div>
        """ if med_frac is not None else f"""
        <div class='card'><h3>{st}</h3>
          <table class='kv'>
            <tr><td>questions</td><td>{n}</td></tr>
            <tr><td>start UW</td><td>{len(uw0)}</td></tr>
            <tr><td>UW → CC before end</td><td>{len(succ)}</td></tr>
            <tr><td>degraded right→wrong</td><td>{len(deg)}</td></tr>
          </table>
        </div>
        """)
    cards_html = "<div class='cards'>" + "".join(cards) + "</div>"

    html = f"""<!DOCTYPE html><html><head><meta charset='utf-8'>
<title>Progressive Entropy Dashboard</title>
<script src='https://cdn.plot.ly/plotly-latest.min.js'></script>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
margin:0;background:#f5f5f7;color:#1d1d1f;}}
.wrap{{max-width:1300px;margin:0 auto;padding:20px;}}
h1{{font-size:30px;margin:10px 0;}}
.tabs{{display:flex;gap:5px;border-bottom:2px solid #e5e7eb;margin:20px 0;}}
.tab{{padding:12px 24px;cursor:pointer;background:#fff;border-radius:8px 8px 0 0;
font-weight:600;color:#666;border:1px solid #e5e7eb;border-bottom:none;}}
.tab.active{{background:#0071e3;color:white;}}
.panel{{display:none;background:#fff;border-radius:0 12px 12px 12px;padding:25px;
box-shadow:0 2px 6px rgba(0,0,0,0.06);}}
.panel.active{{display:block;}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:15px;margin:20px 0;}}
.card{{background:#fafafa;border-radius:10px;padding:18px;border:1px solid #eee;}}
.card h3{{margin-top:0;color:#0071e3;}}
.kv{{width:100%;font-size:13px;}}
.kv td{{padding:4px 0;}}
.kv td:last-child{{text-align:right;font-weight:600;}}
.def{{background:#f0f9ff;border-left:4px solid #0071e3;padding:15px;border-radius:6px;margin:15px 0;}}
.def b{{color:#0071e3;}}
pre{{background:#1d1d1f;color:#e5e5e7;padding:20px;border-radius:8px;
overflow:auto;font-size:12px;line-height:1.5;}}
.ex{{background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:18px;margin:18px 0;}}
.ex-h{{font-size:13px;}}
.qtext{{font-size:13px;color:#444;margin:10px 0;padding:10px;background:#fafafa;border-radius:6px;}}
.muted{{font-size:11px;color:#666;}}
.dot{{display:inline-block;width:10px;height:10px;border-radius:50%;}}
.chunktbl{{width:100%;border-collapse:collapse;font-size:12px;margin-top:10px;}}
.chunktbl th{{background:#1d1d1f;color:white;padding:8px;text-align:left;}}
.chunktbl td{{padding:8px;border-bottom:1px solid #eee;vertical-align:top;}}
.chk{{font-family:'SF Mono',Menlo,monospace;font-size:11px;line-height:1.5;max-width:550px;}}
.stb{{display:inline-block;padding:3px 8px;border-radius:5px;color:white;
font-weight:bold;font-size:11px;}}
</style></head><body><div class='wrap'>
<h1>Progressive entropy — interactive dashboard</h1>

<div class='tabs'>
<div class='tab active' onclick='show(0)'>Overview</div>
<div class='tab' onclick='show(1)'>Statistics</div>
<div class='tab' onclick='show(2)'>Examples</div>
<div class='tab' onclick='show(3)'>Plots</div>
</div>

<div class='panel active'>
<h2>Key numbers</h2>
{cards_html}
<div class='def'>
<b>States:</b> CC = Confident & Correct, UC = Uncertain & Correct,
UW = Uncertain & Wrong, CW = Confident & Wrong.
A step is "confident" if H_s is below the per-student median.
<br><br>
<b>Central success metric (UW → CC before CoT end):</b>
the question started uncertain-wrong, AND the student reached confident-correct
strictly before the teacher CoT ran out. This is the only case where we can
say reasoning was actually informative — if the student only reaches CC at
the very last chunk, we can't tell if reasoning helped or it was just final answer.
<br><br>
<b>Degradation (right → wrong):</b> the student started right (CC or UC) but
ended wrong (CW or UW). These are cases where extra reasoning hurt.
</div>
</div>

<div class='panel'>
<h2>Statistics report</h2>
<pre>{_esc(report_text)}</pre>
</div>

<div class='panel'>
<h2>Examples</h2>
<p class='muted'>Each row of the table is one chunk of 32 reasoning tokens.
Background color = state after the chunk. Shows the chunk text, the model's
answer letter, the correctness mark, the H_s value and the state code.</p>
{examples_html}
</div>

<div class='panel'>
<h2>Plots (interactive)</h2>
{plots_html}
</div>

</div>
<script>
function show(i){{
  document.querySelectorAll('.tab').forEach((t,j)=>t.classList.toggle('active',i===j));
  document.querySelectorAll('.panel').forEach((p,j)=>p.classList.toggle('active',i===j));
  if(i===3){{window.dispatchEvent(new Event('resize'));}}
}}
{plot_scripts}
</script>
</body></html>"""
    out_path.write_text(html, encoding="utf-8")
    print(f"  saved: {out_path}")


# ── MAIN ────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default=DEFAULT_RESULTS)
    p.add_argument("--teacher", default=DEFAULT_TEACHER)
    p.add_argument("--out", default=DEFAULT_OUT)
    args = p.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print("[1] Loading results…")
    df = load_results(args.results)
    print(f"   {len(df)} measurements, {df['question_id'].nunique()} questions")

    print("[2] Assigning states…")
    df, thresholds = assign_states(df)

    print("[3] Computing transitions…")
    trans = compute_transitions(df)

    print("[4] Loading teacher CoT…")
    teacher_df = load_teacher(args.teacher)

    print("[5] Loading tokenizers for chunking…")
    students_tok = {}
    for st in df["student_label"].unique():
        students_tok[st] = maybe_tokenizer(TOKENIZERS.get(st))

    print("[6] Building dashboard…")
    build_dashboard(df, trans, thresholds, teacher_df, students_tok, out)
    print(f"\n✓ Open in a browser: {out}")


if __name__ == "__main__":
    main()