"""
Common-test evaluation of the ALREADY-TRAINED complexity-split checkpoints. NO retraining —
walks existing checkpoint dirs (artifacts/distillation_by_complexity_splits/mmlu/{model}/group{N})
and runs inference only, exactly like the in-bin evals in shared.py.

Runners: eval_qwen_3b.py, eval_phi4mini.py, eval_llama_3b.py.

Two test sets per model, built once and cached next to the split parquets:
  common_random600.parquet   — 600 questions from the model's held-out pool (union of
                               group{N}_test). If the pool of questions held out by ALL THREE
                               models is >= 600, the SAME question_ids are used for every model
                               (sorted ids + fixed seed) -> cross-model comparable.
  common_probe_bin{N}.parquet — 100 held-out questions per bin (the model's OWN bins). Evaluated
                               SEPARATELY per bin: balanced-600 accuracy = mean of the six probes,
                               and the per-bin summaries give the 6×6 transfer matrix
                               (trained on bin g -> accuracy on bin b) for free.

Leakage guards (script fails loudly):
  1. held-out pool ∩ ANY group's train (this model) == 0, by question_id
  2. no duplicate question_id inside a test set
  3. each probe is exactly N_COMMON/6 rows, all from *_test.parquet
Bins partition the corpus, so a question from bin k's test was never in ANY bin's train —
guard 1 asserts that instead of assuming it.
"""


import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from core.datasets.mmlu.mmlu_reasoning_response_dataset import MMLUReasoningResponseDataset
from core.datasets.qa_dataset import QADatasetConfig
from core.datasets.qa_dataset_adapter import QADatasetAdapter
from core.evaluation.multi_checkpoint_evaluator import (
    GenerationConfig,
    MultiCheckpointEvaluator,
    MultiCheckpointEvaluatorConfig,
)
from core.training.thinking_tokens import setup_thinking_tokens
from core.utils.logger import logger
from experiments.distillation_by_complexity_splits.mmlu.shared import ROOT, base_model_path, splits_dir

ALL_MODELS = ["Qwen2.5-3B-Instruct", "Phi-4-mini-instruct", "llama_3b"]
SEED = 42
N_COMMON = 600


def _groups_present(model_name: str) -> list[int]:
    return [g for g in range(6) if splits_dir(model_name).joinpath(f"group{g}_test.parquet").exists()]


def _load(model_name: str, kind: str) -> pd.DataFrame:
    parts = []
    for g in _groups_present(model_name):
        df = pd.read_parquet(splits_dir(model_name).joinpath(f"group{g}_{kind}.parquet"))
        df["bin"] = g
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def build_common_tests(model_name: str):
    out_rand = splits_dir(model_name).joinpath("common_random600.parquet")

    tests = _load(model_name, "test")
    trains = _load(model_name, "train")
    train_ids = set(trains["question_id"])

    # guard 1+2: the held-out pool is clean and unique
    assert not (set(tests["question_id"]) & train_ids), "LEAK: test pool intersects a train split"
    assert tests["question_id"].is_unique, "duplicate question_id in the held-out pool"

    # random600: prefer the pool held out by EVERY model -> identical ids across models
    shared_ids = set(tests["question_id"])
    for other in ALL_MODELS:
        if other != model_name and _groups_present(other):
            shared_ids -= set(_load(other, "train")["question_id"])
    rng = np.random.default_rng(SEED)
    if len(shared_ids) >= N_COMMON:
        ids = set(rng.choice(sorted(shared_ids), size=N_COMMON, replace=False))
        logger.info(f"random600 from cross-model clean pool ({len(shared_ids)} ids) — same questions for all models")
    else:
        ids = set(rng.choice(sorted(set(tests["question_id"])), size=N_COMMON, replace=False))
        logger.warning(
            f"cross-model clean pool too small ({len(shared_ids)}): sampling from this model's own held-out pool — "
            "same protocol across models, NOT same questions"
        )
    rand = tests[tests["question_id"].isin(ids)]
    assert len(rand) == N_COMMON and not (set(rand["question_id"]) & train_ids)
    rand.drop(columns=["bin"]).to_parquet(out_rand, index=False)

    # balanced600 = six per-bin probes of N_COMMON/6 questions each, evaluated SEPARATELY:
    # balanced accuracy = mean of the six probe accuracies (equal n), and the per-probe
    # summaries give the cross-bin transfer matrix directly.
    groups = _groups_present(model_name)
    per = N_COMMON // len(groups)
    probe_paths = []
    for g in groups:
        probe = tests[tests["bin"] == g].sample(n=per, random_state=SEED)
        assert len(probe) == per and not (set(probe["question_id"]) & train_ids)
        p = splits_dir(model_name).joinpath(f"common_probe_bin{g}.parquet")
        probe.drop(columns=["bin"]).to_parquet(p, index=False)
        probe_paths.append((g, p))

    logger.info(f"built {out_rand.name} ({len(rand)}) and {len(probe_paths)} probes × {per}")
    return out_rand, probe_paths


def run(model_name: str, caps: list[int] = [2048], groups: list[int] | None = None):
    MODEL_NAME = base_model_path(model_name)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    setup_thinking_tokens(tokenizer)

    rand_path, probe_paths = build_common_tests(model_name)

    for group in groups if groups is not None else _groups_present(model_name):
        ckpt_dir = ROOT.joinpath(f"artifacts/distillation_by_complexity_splits/mmlu/{model_name}/group{group}")
        if not ckpt_dir.exists():
            logger.warning(f"group {group}: no checkpoints dir, skipping")
            continue
        for cap in caps:
            tasks = [("common_random600", rand_path)] + [(f"common_probe_bin{g}", p) for g, p in probe_paths]
            for label, test_path in tasks:
                logger.info(f"{model_name} group{group} · {label} · cap{cap}")
                MultiCheckpointEvaluator(
                    config=MultiCheckpointEvaluatorConfig(
                        checkpoints_dir=ckpt_dir.as_posix(),
                        eval_dataset=QADatasetAdapter(
                            dataset=MMLUReasoningResponseDataset(
                                config=QADatasetConfig(
                                    path=test_path.as_posix(),
                                    dataset_id=f"group{group}_{label}_cap{cap}",
                                ),
                                tokenizer=tokenizer,
                            ),
                            add_thinking_start_token=True,
                        ),
                        generation=GenerationConfig(
                            max_new_tokens=cap + 10, max_thinking_tokens=cap, max_batch_size=256
                        ),
                        summary_filename=f"group{group}_summary_{label}_cap{cap}.json",
                    ),
                    tokenizer=tokenizer,
                ).evaluate_all()
