# Distillation by complexity splits (MMLU)

Students are distilled on a single complexity bin (`groupN`, N = 0..5, from easiest to hardest)
of the teacher-generated traces and then evaluated on every complexity bin of a held-out
random test sample, plus on the full random sample.

## Layout

```
mmlu/<model>/groupN/
  groupN_summary_reasoning_evals_cap{2048,4096}.json   # in-bin eval (train bin == test bin)
  groupN_summary_common_probe_binK_cap2048.json        # trained on bin N, tested on bin K (100 questions)
  groupN_summary_common_random600_cap2048.json         # trained on bin N, tested on the random 600-question sample
  train_N_distilled_*.parquet                          # distillation training data for bin N
mmlu/<model>/random_seed42/
  <model>_random_seed42_summary_common_probe_binK_cap2048.json   # trained on a random bin (seed 42), tested on bin K
  <model>_random_seed42_summary_common_random600_cap2048.json    # trained on a random bin (seed 42), tested on random600
```

Every summary lists accuracy per checkpoint (epochs 5, 10, 20, 35, 50) with generation capped at 2048 tokens.
`probe_binK` test sets have 100 questions each; `random600` has 600.

Models: `Phi-4-mini-instruct`, `Qwen2.5-3B-Instruct`, `llama_3b`.
Random-bin baselines (`random_seed42`) are available for Qwen and Llama.

## Charts

Chart-only HTML pages (open `charts/index.html`; each page loads `charts/common.js`, which holds the data):

- `charts/common_test_accuracy_by_epoch.html` — random600 and balanced600 accuracy by epoch, one panel per model, colour = training bin.
- `charts/common_test_best_epoch.html` — best epoch per training bin on random600 and balanced600.
- `charts/in_bin_accuracy_by_epoch.html` — in-bin test accuracy by epoch, one panel per bin (cap2048 / cap4096).
- `charts/best_epoch_accuracy_vs_bin.html` — best-epoch in-bin accuracy vs complexity bin.
- `charts/training_gain_by_bin.html` — training gain (best epoch − epoch 10) per bin and difficulty × gain map.
- `charts/generation_cap.html` — Δ accuracy cap4096 − cap2048 and Phi-4-mini truncation rate.
- `charts/training_bin_transfer.html` — balanced600: gain on own bin vs transfer to the other bins.
- `charts/training_bin_ranking.html` — mean rank of training bins across models/tests and advantage vs cross-bin mean.
- `charts/best_bin_vs_random.html` — best bin vs random seed42 (same size) and the repository random-arm.

## Summary

Cells are `best accuracy over epochs (epoch) / accuracy at epoch 50`.

### Phi-4-mini-instruct

| trained on | bin0 | bin1 | bin2 | bin3 | bin4 | bin5 | random600 |
|---|---|---|---|---|---|---|---|
| group0 | 0.67 (ep20) / 0.53 | 0.43 (ep35) / 0.42 | 0.31 (ep35) / 0.30 | 0.21 (ep20) / 0.15 | 0.30 (ep20) / 0.22 | 0.18 (ep20) / 0.13 | 0.35 (ep50) / 0.35 |
| group1 | 0.59 (ep35) / 0.59 | 0.42 (ep20) / 0.40 | 0.40 (ep35) / 0.28 | 0.27 (ep50) / 0.27 | 0.29 (ep20) / 0.28 | 0.24 (ep35) / 0.20 | 0.36 (ep50) / 0.36 |
| group2 | 0.67 (ep35) / 0.65 | 0.46 (ep35) / 0.43 | 0.37 (ep50) / 0.37 | 0.33 (ep35) / 0.22 | 0.32 (ep50) / 0.32 | 0.26 (ep35) / 0.25 | 0.41 (ep50) / 0.41 |
| group3 | 0.68 (ep35) / 0.65 | 0.40 (ep35) / 0.37 | 0.31 (ep35) / 0.29 | 0.29 (ep20) / 0.23 | 0.32 (ep20) / 0.25 | 0.32 (ep35) / 0.29 | 0.36 (ep35) / 0.35 |
| group4 | 0.60 (ep20) / 0.56 | 0.48 (ep20) / 0.42 | 0.34 (ep35) / 0.31 | 0.34 (ep50) / 0.34 | 0.33 (ep35) / 0.28 | 0.36 (ep35) / 0.36 | 0.38 (ep35) / 0.34 |
| group5 | 0.61 (ep35) / 0.59 | 0.36 (ep35) / 0.35 | 0.28 (ep50) / 0.28 | 0.30 (ep50) / 0.30 | 0.37 (ep35) / 0.32 | 0.31 (ep50) / 0.31 | 0.39 (ep50) / 0.39 |

### Qwen2.5-3B-Instruct

| trained on | bin0 | bin1 | bin2 | bin3 | bin4 | bin5 | random600 |
|---|---|---|---|---|---|---|---|
| group0 | 0.66 (ep35) / 0.63 | 0.52 (ep50) / 0.52 | 0.42 (ep5) / 0.32 | 0.35 (ep10) / 0.20 | 0.38 (ep10) / 0.31 | 0.27 (ep50) / 0.27 | 0.37 (ep5) / 0.37 |
| group1 | 0.63 (ep20) / 0.53 | 0.55 (ep50) / 0.55 | 0.39 (ep10) / 0.34 | 0.30 (ep10) / 0.27 | 0.37 (ep20) / 0.37 | 0.29 (ep10) / 0.26 | 0.34 (ep35) / 0.32 |
| group2 | 0.60 (ep10) / 0.58 | 0.49 (ep10) / 0.42 | 0.36 (ep35) / 0.28 | 0.33 (ep20) / 0.32 | 0.36 (ep5) / 0.31 | 0.33 (ep10) / 0.25 | 0.38 (ep50) / 0.38 |
| group3 | 0.64 (ep50) / 0.64 | 0.47 (ep20) / 0.45 | 0.35 (ep10) / 0.24 | 0.34 (ep50) / 0.34 | 0.42 (ep5) / 0.24 | 0.28 (ep10) / 0.26 | 0.36 (ep20) / 0.35 |
| group4 | 0.61 (ep10) / 0.57 | 0.47 (ep10) / 0.33 | 0.41 (ep10) / 0.29 | 0.29 (ep5) / 0.29 | 0.37 (ep10) / 0.33 | 0.29 (ep35) / 0.23 | 0.39 (ep10) / 0.34 |
| group5 | 0.65 (ep5) / 0.59 | 0.46 (ep5) / 0.43 | 0.37 (ep35) / 0.23 | 0.36 (ep10) / 0.26 | 0.36 (ep10) / 0.33 | 0.22 (ep20) / 0.21 | 0.36 (ep10) / 0.32 |
| random (seed42) | 0.61 (ep5) / 0.59 | 0.43 (ep10) / 0.35 | 0.35 (ep35) / 0.30 | 0.34 (ep10) / 0.27 | 0.35 (ep5) / 0.33 | 0.28 (ep5) / 0.27 | 0.36 (ep5) / 0.33 |

### llama_3b

| trained on | bin0 | bin1 | bin2 | bin3 | bin4 | bin5 | random600 |
|---|---|---|---|---|---|---|---|
| group0 | 0.62 (ep20) / 0.61 | 0.40 (ep50) / 0.40 | 0.29 (ep50) / 0.29 | 0.35 (ep5) / 0.32 | 0.30 (ep10) / 0.29 | 0.23 (ep35) / 0.21 | 0.33 (ep35) / 0.32 |
| group1 | 0.61 (ep10) / 0.60 | 0.40 (ep10) / 0.35 | 0.36 (ep10) / 0.30 | 0.40 (ep10) / 0.33 | 0.34 (ep35) / 0.27 | 0.22 (ep10) / 0.20 | 0.37 (ep5) / 0.33 |
| group2 | 0.57 (ep20) / 0.56 | 0.42 (ep35) / 0.42 | 0.38 (ep35) / 0.31 | 0.36 (ep5) / 0.29 | 0.30 (ep20) / 0.25 | 0.30 (ep5) / 0.26 | 0.35 (ep10) / 0.31 |
| group3 | 0.61 (ep10) / 0.59 | 0.40 (ep10) / 0.30 | 0.35 (ep20) / 0.34 | 0.41 (ep10) / 0.30 | 0.34 (ep10) / 0.22 | 0.28 (ep10) / 0.22 | 0.37 (ep10) / 0.34 |
| group4 | 0.64 (ep50) / 0.64 | 0.34 (ep10) / 0.32 | 0.35 (ep20) / 0.32 | 0.37 (ep20) / 0.37 | 0.36 (ep5) / 0.26 | 0.28 (ep10) / 0.17 | 0.35 (ep10) / 0.33 |
| group5 | 0.63 (ep10) / 0.59 | 0.41 (ep10) / 0.37 | 0.39 (ep5) / 0.24 | 0.37 (ep10) / 0.32 | 0.38 (ep5) / 0.23 | 0.30 (ep10) / 0.25 | 0.38 (ep10) / 0.35 |
| random (seed42) | 0.59 (ep10) / 0.55 | 0.41 (ep5) / 0.36 | 0.33 (ep10) / 0.28 | 0.38 (ep50) / 0.38 | 0.35 (ep5) / 0.24 | 0.29 (ep20) / 0.24 | 0.36 (ep10) / 0.32 |
