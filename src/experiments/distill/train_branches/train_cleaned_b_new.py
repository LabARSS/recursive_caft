"""
Shared orchestration for cleaned Branch B CoT training and evaluation.

Follows the same pattern as sft_by_complexity_splits experiments:
1. Train with LoRATrainer
2. Evaluate all checkpoints post-training with MultiCheckpointEvaluator
"""

import re
from collections.abc import Callable
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer

from core.datasets.causal_dataset import CausalDatasetConfig
from core.datasets.causal_dataset_adapter import CausalDatasetAdapter
from core.datasets.distillation.distillation_branch_b_cot_dataset import DistillationBranchBCoTDataset
from core.datasets.mmlu.mmlu_cot_response_dataset import MMLUCoTResponseDataset
from core.datasets.qa_dataset import QADatasetConfig
from core.datasets.qa_dataset_adapter import QADatasetAdapter
from core.evaluation.multi_checkpoint_evaluator import (
    GenerationConfig,
    MultiCheckpointEvaluator,
    MultiCheckpointEvaluatorConfig,
)
from core.training.lora_trainer import LoRATrainer, LoRATrainerConfig, LoRATrainingArgs, LoRASpecificTrainingArgs
from core.utils.logger import logger

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
PROJECT_ROOT = Path(__file__).resolve().parents[4]

ANSWER_LEAK_RE = re.compile(
    "|".join([
        r"\bcorrect answer\b", r"\bthe answer is\b", r"\banswer is\b",
        r"\banswer:\b", r"\bcorrect option\b", r"\bcorrect choice\b",
        r"\b[a-j]\s+is\s+correct\b", r"\[\[\s*[a-jA-J]\s*\]\]",
    ]),
    flags=re.IGNORECASE,
)


class FilteredCausalDatasetAdapter(CausalDatasetAdapter):
    """CausalDatasetAdapter that filters rows before tokenization."""

    def __init__(self, dataset, row_filter: Callable[[dict], bool]):
        super().__init__(dataset)
        self.row_filter = row_filter

    def process_dataset(self, path_override: str | None = None):
        ds = self._load_ds(path_override)
        ds = ds.filter(self.row_filter, num_proc=4)
        return ds.map(
            lambda row: self.process_row(row).model_dump(),
            num_proc=4,
            remove_columns=ds.column_names,
            load_from_cache_file=False,
        )


def _collect_eval_question_ids(eval_split_dir: str, groups: int) -> set[str]:
    """Collect question IDs from eval test splits to exclude from training."""
    split_root = PROJECT_ROOT / eval_split_dir
    question_ids: set[str] = set()
    for group_idx in range(groups):
        path = split_root / f"group{group_idx}_test.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Eval split not found: {path}")
        rows = pq.read_table(path, columns=["question_id"]).to_pylist()
        question_ids.update(str(row["question_id"]) for row in rows)
    return question_ids


def _reasoning_is_usable(reasoning: str | None) -> bool:
    """Check that reasoning exists and doesn't leak the answer."""
    if not reasoning or not str(reasoning).strip():
        return False
    return ANSWER_LEAK_RE.search(str(reasoning)) is None


def _build_train_row_filter(eval_question_ids: set[str]) -> Callable[[dict], bool]:
    """Exclude eval questions and rows with leaked/empty reasoning."""

    def row_filter(row: dict) -> bool:
        qid = str(row["input"]["question_id"])
        if qid in eval_question_ids:
            return False
        reasoning = str(((row.get("output") or {}).get("thinking")) or "").strip()
        return _reasoning_is_usable(reasoning)

    return row_filter


def run_cleaned_b_training(
    *,
    prompt_id: int = 3,
    eval_split_dir: str = "data/out/splits/single_token_entropy/mmlu/qwen_3b",
    eval_groups: int = 6,
    per_device_train_batch_size: int = 1,
    effective_train_batch_size: int = 120,
    num_train_epochs: int = 20,
    learning_rate: float = 1e-4,
    cot_eval_max_new_tokens: int = 8192,
    cot_eval_max_batch_size: int = 64,
    run_tag: str = "",
):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    train_data_path = (
        PROJECT_ROOT
        / f"data/out/distillation/mmlu_synth_gptoss_b_t0_8_cleaned_32b_prompt{prompt_id}.parquet"
    )
    if not train_data_path.exists():
        raise FileNotFoundError(f"Train parquet not found: {train_data_path}")

    eval_question_ids = _collect_eval_question_ids(eval_split_dir, eval_groups)
    train_row_filter = _build_train_row_filter(eval_question_ids)

    run_suffix = f"_{run_tag}" if run_tag else ""
    out_path = str(
        PROJECT_ROOT / f"artifacts/sft_distill/branch_b_cleaned_prompt{prompt_id}{run_suffix}"
    )
    save_schedule = sorted(
        set(e for e in [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20] if e <= num_train_epochs)
        | {num_train_epochs}
    )

    # --- Training ---
    logger.info(f"Training: prompt_id={prompt_id}, epochs={num_train_epochs}, out={out_path}")

    trainer = LoRATrainer(
        config=LoRATrainerConfig(
            out_path=out_path,
            model_id=MODEL_NAME,
            train_dataset=FilteredCausalDatasetAdapter(
                dataset=DistillationBranchBCoTDataset(
                    tokenizer=tokenizer,
                    config=CausalDatasetConfig(path=str(train_data_path)),
                ),
                row_filter=train_row_filter,
            ),
            training_args=LoRATrainingArgs(
                num_train_epochs=num_train_epochs,
                per_device_train_batch_size=per_device_train_batch_size,
                effective_train_batch_size=effective_train_batch_size,
                learning_rate=learning_rate,
                warmup_ratio=0.06,
                torch_compile=False,
            ),
            lora_training_args=LoRASpecificTrainingArgs(
                r=16,
                alpha=32,
                lora_dropout=0.05,
                use_rslora=True,
            ),
            save_schedule=save_schedule,
        ),
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.unload()

    # --- CoT Evaluation (post-training) ---
    logger.info("Starting post-training CoT evaluation...")

    eval_split_root = PROJECT_ROOT / eval_split_dir

    cot_evaluator = MultiCheckpointEvaluator(
        config=MultiCheckpointEvaluatorConfig(
            checkpoints_dir=out_path,
            eval_dataset=[
                QADatasetAdapter(
                    dataset=MMLUCoTResponseDataset(
                        tokenizer=tokenizer,
                        config=QADatasetConfig(
                            path=str(eval_split_root / f"group{j}_test.parquet"),
                            dataset_id=f"mmlu_cot_response_group{j}_test",
                        ),
                    )
                )
                for j in range(eval_groups)
            ],
            base_model_id=MODEL_NAME,
            generation=GenerationConfig(
                max_new_tokens=cot_eval_max_new_tokens,
                max_batch_size=cot_eval_max_batch_size,
            ),
            summary_filename="summary_cot.json",
        ),
        tokenizer=tokenizer,
    )
    cot_evaluator.evaluate_all()
