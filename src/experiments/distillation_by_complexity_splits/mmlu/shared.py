import os
from pathlib import Path

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

ROOT = Path(__file__).parent.joinpath("../../../..")

# `model_name` names the artifacts/ output dir (and the model dir under BASE_MODELS_DIR);
# the nick is the repo-wide key: artifacts/base_models_v0/<nick>, data/out/splits/.../<nick>,
# packing budgets.
MODEL_NICKS = {
    "Qwen2.5-3B-Instruct": "qwen_3b",
    "Phi-4-mini-instruct": "phi4_mini",
    "llama_3b": "llama_3b",
}


def model_nick(model_name: str) -> str:
    return MODEL_NICKS.get(model_name, model_name)


def base_model_path(model_name: str) -> str:
    """$BASE_MODELS_DIR/<model_name> if the env var is set (e.g. /mnt/data198/LLM/agents),
    otherwise the repo's artifacts/base_models_v0/<nick>."""
    base = os.environ.get("BASE_MODELS_DIR")
    if base:
        return Path(base).joinpath(model_name).as_posix()
    return ROOT.joinpath(f"artifacts/base_models_v0/{model_nick(model_name)}").as_posix()


def splits_dir(model_name: str) -> Path:
    return ROOT.joinpath(f"data/out/splits/single_token_entropy/mmlu/{model_nick(model_name)}")


def run(model_name: str, save_schedule: list[int] = [5, 10, 20, 35, 50], max_thinking_tokens: list[int] = [2048]):
    MODEL_NAME = base_model_path(model_name)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    setup_thinking_tokens(tokenizer)

    lora_training_args = LoRASpecificTrainingArgs(train_thinking_token_embeddings=True)
    if model_nick(model_name) == "phi4_mini":
        lora_training_args.target_modules = phi4_mini_lora_target_modules

    OUT_PATHS = [
        Path(__file__).parent.joinpath(
            f"../../../../artifacts/distillation_by_complexity_splits/mmlu/{model_name}/group{group}"
        )
        for group in range(6)
    ]

    for group, out_path in enumerate(OUT_PATHS):
        logger.info(f"Training on group {group}...")

        train_dataset_path = out_path.joinpath(
            f"train_{group}_distilled_deepseek_v4_flash_regenerate_incorrect_w_large.parquet"
        )
        merge_mmlu_on_question_id(
            main_path=splits_dir(model_name).joinpath(f"group{group}_train.parquet"),
            extra_paths=[
                Path(__file__).parent.joinpath(
                    "../../../../data/out/distillation/mmlu_distilled_deepseek_v4_flash_regenerate_incorrect_w_large.parquet"
                ),
            ],
            extra_columns=[
                {
                    "distill_reasoning": "distill_reasoning",
                    "distill_answer": "distill_answer",
                    "distill_ans_correct": "distill_ans_correct",
                },
            ],
            save_path=train_dataset_path,
            aggregation_function=lambda df: truncate_column(df, col="distill_reasoning", max_len=8192),
        )

        trainer = LoRATrainer(
            config=LoRATrainerConfig(
                out_path=out_path.as_posix(),
                model_id=MODEL_NAME,
                train_dataset=CausalDatasetAdapter(
                    dataset=MMLUReasoningResponseDataset(
                        config=QADatasetConfig(
                            path=train_dataset_path.as_posix(),
                            dataset_id=f"mmlu_train_{group}_distilled_deepseek_v4_flash_regenerate_incorrect_w_large",
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

    for tokens_cap in max_thinking_tokens:
        for group, out_path in enumerate(OUT_PATHS):
            logger.info(f"Reasoning token evals on group {group}...")

            eval_dataset_id = f"group{group}_mmlu_test_cap{tokens_cap}"
            summary_filename = f"group{group}_summary_reasoning_evals_cap{tokens_cap}.json"

            cot_evaluator = MultiCheckpointEvaluator(
                config=MultiCheckpointEvaluatorConfig(
                    checkpoints_dir=out_path.as_posix(),
                    eval_dataset=QADatasetAdapter(
                        dataset=MMLUReasoningResponseDataset(
                            config=QADatasetConfig(
                                path=splits_dir(model_name).joinpath(f"group{group}_test.parquet").as_posix(),
                                dataset_id=eval_dataset_id,
                            ),
                            tokenizer=tokenizer,
                        ),
                        add_thinking_start_token=True,
                    ),
                    generation=GenerationConfig(
                        max_new_tokens=tokens_cap + 10,
                        max_thinking_tokens=tokens_cap,
                        max_batch_size=256,
                    ),
                    summary_filename=summary_filename,
                ),
                tokenizer=tokenizer,
            )
            cot_evaluator.evaluate_all()
