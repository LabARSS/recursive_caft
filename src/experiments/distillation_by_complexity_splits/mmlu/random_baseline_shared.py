"""RANDOM baseline for the complexity-split experiment: train on a random subset of the SAME pool
and the SAME budget as one bin arm, then evaluate ONLY on the two common tests.

    python src/experiments/distillation_by_complexity_splits/mmlu/train_random_baseline_qwen_3b.py

Mirrors distillation_by_complexity_splits/mmlu/shared.py exactly — same LoRATrainer, LoRA args,
packing budget, 8192-token trace truncation, same save_schedule [5,10,20,35,50] — with two
differences:
  1. the training set is a random sample of n examples from the UNION of all group{N}_train
     (n defaults to one group's train size, so the data budget matches a single-bin arm);
  2. no reasoning_evals on the per-bin test — only common_random600 and the six
     common_probe_bin{B} slices built by common_eval_shared.build_common_tests, so the numbers
     drop straight into the same comparison as the bin arms.

Seeds: pass seeds=[42, 43, 44] to train several draws; each lands in its own checkpoint dir.
Leakage: asserted — the sampled train ids never intersect any common-test id.
"""

from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

from core.datasets.causal_dataset_adapter import CausalDatasetAdapter
from core.datasets.mmlu.mmlu_reasoning_response_dataset import MMLUReasoningResponseDataset
from core.datasets.qa_dataset import QADatasetConfig
from core.datasets.qa_dataset_adapter import QADatasetAdapter
from core.evaluation.multi_checkpoint_evaluator import (
    GenerationConfig,
    MultiCheckpointEvaluator,
    MultiCheckpointEvaluatorConfig,
)
from core.training.base_trainer import PackingConfig
from core.training.lora_trainer import (
    LoRASpecificTrainingArgs,
    LoRATrainer,
    LoRATrainerConfig,
    LoRATrainingArgs,
    phi4_mini_lora_target_modules,
)
from core.training.packing_budgets import packing_budget
from core.training.thinking_tokens import setup_thinking_tokens
from core.utils.datasets import merge_mmlu_on_question_id, truncate_column
from core.utils.logger import logger

from experiments.distillation_by_complexity_splits.mmlu.common_eval_shared import build_common_tests
from experiments.distillation_by_complexity_splits.mmlu.shared import ROOT, base_model_path, model_nick, splits_dir

DISTILL = "data/out/distillation/mmlu_distilled_deepseek_v4_flash_regenerate_incorrect_w_large.parquet"


def _groups(model_name: str) -> list[int]:
    return [g for g in range(6) if splits_dir(model_name).joinpath(f"group{g}_train.parquet").exists()]


def build_random_train(model_name: str, seed: int, n_train: int | None, banned: set) -> tuple[Path, int]:
    """Random sample of n_train rows from the union of all group{N}_train, disjoint from the tests."""
    groups = _groups(model_name)
    parts = [pd.read_parquet(splits_dir(model_name).joinpath(f"group{g}_train.parquet")) for g in groups]
    pool = pd.concat(parts, ignore_index=True)
    pool = pool[~pool["question_id"].isin(banned)]
    assert pool["question_id"].is_unique, "duplicate question_id in the training pool"

    n = n_train or len(parts[0])  # one bin arm's budget
    assert n <= len(pool), f"asked for {n} examples, pool has {len(pool)}"
    sample = pool.sample(n=n, random_state=seed)
    assert not (set(sample["question_id"]) & banned), "LEAK: sampled train row is in a common test"

    out = splits_dir(model_name).joinpath(f"random_baseline_seed{seed}_n{n}.parquet")
    sample.to_parquet(out, index=False)
    logger.info(f"random train: {n} of {len(pool)} pool rows (seed {seed}) -> {out.name}")
    return out, n


def run(model_name: str, seeds: list[int] = [42], n_train: int | None = None,
        save_schedule: list[int] = [5, 10, 20, 35, 50], caps: list[int] = [2048]):
    MODEL_NAME = base_model_path(model_name)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    setup_thinking_tokens(tokenizer)

    lora_training_args = LoRASpecificTrainingArgs(train_thinking_token_embeddings=True)
    if model_nick(model_name) == "phi4_mini":
        lora_training_args.target_modules = phi4_mini_lora_target_modules

    # the very same test files the bin arms are scored on
    rand_path, probe_paths = build_common_tests(model_name)
    banned = set(pd.read_parquet(rand_path)["question_id"])
    for _, p in probe_paths:
        banned |= set(pd.read_parquet(p)["question_id"])
    logger.info(f"common tests hold {len(banned)} distinct question_ids — excluded from training")

    for seed in seeds:
        out_path = ROOT.joinpath(f"artifacts/distillation_by_complexity_splits/mmlu/{model_name}/random_seed{seed}")
        out_path.mkdir(parents=True, exist_ok=True)
        sample_path, n = build_random_train(model_name, seed, n_train, banned)

        # same merge + 8192 truncation as the bin arms
        train_dataset_path = out_path.joinpath("train_random_distilled_deepseek_v4_flash_regenerate_incorrect_w_large.parquet")
        merge_mmlu_on_question_id(
            main_path=sample_path,
            extra_paths=[ROOT.joinpath(DISTILL)],
            extra_columns=[{
                "distill_reasoning": "distill_reasoning",
                "distill_answer": "distill_answer",
                "distill_ans_correct": "distill_ans_correct",
            }],
            save_path=train_dataset_path,
            aggregation_function=lambda df: truncate_column(df, col="distill_reasoning", max_len=8192),
        )

        logger.info(f"Training random baseline · {model_name} · seed {seed} · n={n}")
        trainer = LoRATrainer(
            config=LoRATrainerConfig(
                out_path=out_path.as_posix(),
                model_id=MODEL_NAME,
                train_dataset=CausalDatasetAdapter(
                    dataset=MMLUReasoningResponseDataset(
                        config=QADatasetConfig(
                            path=train_dataset_path.as_posix(),
                            dataset_id=f"mmlu_train_random_seed{seed}_distilled_deepseek_v4_flash_regenerate_incorrect_w_large",
                        ),
                        tokenizer=tokenizer,
                    )
                ),
                training_args=LoRATrainingArgs(num_train_epochs=save_schedule[-1], per_device_train_batch_size=1),
                lora_training_args=lora_training_args,
                packing=PackingConfig(budget=packing_budget(model_nick(model_name))),
                save_schedule=save_schedule,
            ),
            tokenizer=tokenizer,
        )
        trainer.train()
        trainer.unload()

        # common-test evals only — no per-bin reasoning_evals
        for cap in caps:
            for label, test_path in [("common_random600", rand_path)] + [
                (f"common_probe_bin{g}", p) for g, p in probe_paths
            ]:
                logger.info(f"eval · {model_name} · random seed {seed} · {label} · cap{cap}")
                MultiCheckpointEvaluator(
                    config=MultiCheckpointEvaluatorConfig(
                        checkpoints_dir=out_path.as_posix(),
                        eval_dataset=QADatasetAdapter(
                            dataset=MMLUReasoningResponseDataset(
                                config=QADatasetConfig(
                                    path=test_path.as_posix(),
                                    dataset_id=f"random_seed{seed}_{label}_cap{cap}",
                                ),
                                tokenizer=tokenizer,
                            ),
                            add_thinking_start_token=True,
                        ),
                        generation=GenerationConfig(
                            max_new_tokens=cap + 10, max_thinking_tokens=cap, max_batch_size=256
                        ),
                        summary_filename=f"random_seed{seed}_summary_{label}_cap{cap}.json",
                    ),
                    tokenizer=tokenizer,
                ).evaluate_all()
