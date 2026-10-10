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


def run(
    model_name: str,
    relative_out_path: str,
    train_dataset: str,
    save_schedule: list[int],
    max_thinking_tokens: int = 8192,
    skip_missing_checkpoints: bool = False,
    run_evaluation: bool = True,
    lora_training_args: LoRASpecificTrainingArgs | None = None,
):
    MODEL_NAME = Path(__file__).parent.joinpath(f"../../../../artifacts/base_models_v0/{model_name}").as_posix()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    setup_thinking_tokens(tokenizer)

    OUT_PATH = (
        Path(__file__)
        .parent.joinpath("../../../../artifacts/distillation_on_synthetic_traces/mmlu/")
        .joinpath(relative_out_path)
        .as_posix()
    )

    lora_training_args = lora_training_args or LoRASpecificTrainingArgs()
    lora_training_args.train_thinking_token_embeddings = True
    if model_name == "phi4_mini":
        lora_training_args.target_modules = phi4_mini_lora_target_modules

    trainer = LoRATrainer(
        config=LoRATrainerConfig(
            out_path=OUT_PATH,
            model_id=MODEL_NAME,
            train_dataset=CausalDatasetAdapter(
                dataset=MMLUReasoningResponseDataset(
                    config=QADatasetConfig(
                        path=Path(__file__)
                        .parent.joinpath(f"../../../../data/out/splits/random/mmlu/{train_dataset}.parquet")
                        .as_posix(),
                        dataset_id=f"mmlu_{train_dataset}",
                    ),
                    tokenizer=tokenizer,
                )
            ),
            training_args=LoRATrainingArgs(num_train_epochs=save_schedule[-1], per_device_train_batch_size=1),
            lora_training_args=lora_training_args,
            packing=PackingConfig(budget=packing_budget(model_name)),
            save_schedule=save_schedule,
        ),
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.unload()

    if not run_evaluation:
        return

    eval_dataset_id = "mmlu_random_test"
    summary_filename = "summary_reasoning_evals.json"
    if max_thinking_tokens != 8192:
        eval_dataset_id = f"mmlu_random_test_cap{max_thinking_tokens}"
        summary_filename = f"summary_reasoning_evals_cap{max_thinking_tokens}.json"

    cot_evaluator = MultiCheckpointEvaluator(
        config=MultiCheckpointEvaluatorConfig(
            checkpoints_dir=OUT_PATH,
            eval_dataset=QADatasetAdapter(
                dataset=MMLUReasoningResponseDataset(
                    config=QADatasetConfig(
                        path=Path(__file__)
                        .parent.joinpath("../../../../data/out/splits/random/mmlu/test.parquet")
                        .as_posix(),
                        dataset_id=eval_dataset_id,
                    ),
                    tokenizer=tokenizer,
                ),
                add_thinking_start_token=True,
            ),
            generation=GenerationConfig(
                max_new_tokens=max_thinking_tokens + 10, max_thinking_tokens=max_thinking_tokens, max_batch_size=256
            ),
            summary_filename=summary_filename,
            skip_missing_checkpoints=skip_missing_checkpoints,
        ),
        tokenizer=tokenizer,
    )
    cot_evaluator.evaluate_all()
