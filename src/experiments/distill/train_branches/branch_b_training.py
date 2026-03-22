"""
Shared orchestration for Branch B reasoning training and evaluation.

1. Train with LoRATrainer using preprocessed data + MMLUReasoningResponseDataset
2. Evaluate all checkpoints post-training with MultiCheckpointEvaluator
"""

from pathlib import Path

from transformers import AutoTokenizer

from core.datasets.causal_dataset_adapter import CausalDatasetAdapter
from core.datasets.mmlu.mmlu_cot_response_dataset import MMLUCoTResponseDataset
from core.datasets.mmlu.mmlu_reasoning_response_dataset import MMLUReasoningResponseDataset
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


def run_branch_b_training(
    *,
    prompt_id: int = 1,
    eval_split_dir: str = "data/out/splits/single_token_entropy/mmlu/qwen_3b",
    eval_groups: int = 6,
    per_device_train_batch_size: int = 1,
    num_train_epochs: int = 20,
    cot_eval_max_new_tokens: int = 8192,
    cot_eval_max_batch_size: int = 64,
    run_tag: str = "",
):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.thinking_start_token = "<think>"
    tokenizer.thinking_end_token = "</think>"

    train_data_path = (
        PROJECT_ROOT
        / f"data/out/distillation/mmlu_branch_b_cleaned_prompt{prompt_id}_prepared.parquet"
    )
    if not train_data_path.exists():
        raise FileNotFoundError(
            f"Preprocessed data not found: {train_data_path}. "
            f"Run prepare_cleaned_b_data.py first."
        )

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
            train_dataset=CausalDatasetAdapter(
                dataset=MMLUReasoningResponseDataset(
                    tokenizer=tokenizer,
                    config=QADatasetConfig(
                        path=str(train_data_path),
                        dataset_id=f"distill_branch_b_prompt{prompt_id}",
                    ),
                )
            ),
            training_args=LoRATrainingArgs(
                num_train_epochs=num_train_epochs,
                per_device_train_batch_size=per_device_train_batch_size,
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
