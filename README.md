# Code for "Recursive Offline Distillation via Entropy Gain Sampling and Answer Mixing" paper

## Abstract

Distillation of reasoning chains from Large Language Models (LLMs) into their smaller siblings is a standard training approach.
Frequently, researchers do not have full access to a teacher model and its internal representations, which limits the resulting performance.
We propose a recursive offline distillation method combining two components: entropy-based sampling for selective distillation and answer mixing.
Specifically, for three open models (3-4B), at each epoch, we sample reasoning chains weighted by entropy difference with an auxiliary medium model (70-72B) and, next, supplement thinking traces with single-token ground truth answers.
Our approach consistently outperforms naive distillation with comparable data budgets ($57.15 \pm 1.01$ vs 51.61 on average) and offers a much steeper learning curve.
Additionally, we present a comparative analysis of methods to generate and normalize distillation traces.


## Repository structure

```text
data/
  source/               Source datasets: MMLU-Pro STEM, GPQA, and GSM8K
  out/                  Teacher traces, entropy estimates, and prepared splits
src/
  core/
    datasets/           Dataset adapters, response formats, and sequence packing
    prompts/            Answer, reasoning, and trace augmentation prompts
    complexity_estimation/  Single-token and sequence entropy estimators
    dataset_samplers/   Random, student entropy, and entropy gain sampling
    training/           Embedding initialization, LoRA, and recursive resampling
    evaluation/         Model and checkpoint evaluation
    distillation/       Remote teacher trace generation
    entropy_dynamics/   Entropy measurements along reasoning traces
    analysis/           Shared analysis and plotting utilities
    utils/              Data processing, device selection, and runtime utilities
  experiments/          Model- and dataset-specific experiment entry points
  preprocessing/        Dataset downloads and packing budget notebook
  postprocessing/       Entropy normalization, splits, and trace preparation
  analysis/             Notebooks for results and ablations
artifacts/              Checkpoints, sampled data, evaluations, and summaries
```

The experiment families under `src/experiments/` are:

| Directory | Purpose |
| --- | --- |
| `base_models_v0/` | Initialize and train the `<think>` / `</think>` token embeddings for the students. |
| `estimate_single_token_entropy/` | Estimate uncertainty on MMLU-Pro, GPQA, and GSM8K with student and proxy models. |
| `generate_reasoning_traces/` | Generate direct reasoning traces, explanations, and corrected answers. |
| `distillation_by_metrics/` | Recursive sampling experiments, including entropy gain, student entropy, random sampling, and answer mixing. |
| `distillation_on_synthetic_traces/` | Distillation baselines and trace format, truncation, and reasoning budget ablations. |
| `distillation_by_complexity_splits/` | Distill on six entropy-based groups and evaluate transfer between groups. |
| `sft_by_complexity_splits/` | Supervised fine-tuning on entropy-based groups. |
| `active_learning_analysis/` | Notebooks comparing entropy and larger-model proxies. |

## Setup

Use Python 3.13 or later and `uv`. The training and evaluation configurations target NVIDIA GPUs with CUDA and bfloat16 support. Evaluation and packed training use FlashAttention 2; building it requires a compatible CUDA toolkit and compiler. Proxy entropy estimation loads 70–72B models, so plan GPU memory separately from the 3–4B student runs.

Install dependencies:

```bash
uv sync
uv sync --extra evals # for FA2
```