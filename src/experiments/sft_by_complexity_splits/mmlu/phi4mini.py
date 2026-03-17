from pathlib import Path

from transformers import AutoTokenizer

from core.datasets.causal_dataset_adapter import CausalDatasetAdapter
from core.datasets.mmlu.mmlu_cot_response_dataset import MMLUCoTResponseDataset
from core.datasets.mmlu.mmlu_single_token_response_dataset import MMLUSingleTokenResponseDataset, QADatasetConfig
from core.datasets.qa_dataset_adapter import QADatasetAdapter
from core.evaluation.multi_checkpoint_evaluator import (
    GenerationConfig,
    MultiCheckpointEvaluator,
    MultiCheckpointEvaluatorConfig,
)
from core.training.lora_trainer import LoRATrainer, LoRATrainerConfig, LoRATrainingArgs
from core.utils.logger import logger

MODEL_NAME = "microsoft/Phi-4-mini-instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

paths = [
    Path(__file__)
    .parent.joinpath(f"../../../../artifacts/sft_by_complexity_splits/mmlu/phi4mini/group{group}")
    .as_posix()
    for group in range(6)
]

for group, path in enumerate(paths):
    logger.info(f"Training on group {group}...")

    trainer = LoRATrainer(
        config=LoRATrainerConfig(
            out_path=path,
            model_id=MODEL_NAME,
            train_dataset=CausalDatasetAdapter(
                dataset=MMLUSingleTokenResponseDataset(
                    config=QADatasetConfig(
                        path=Path(__file__)
                        .parent.joinpath(
                            f"../../../../data/out/splits/single_token_entropy/mmlu/phi4mini/group{group}_train.parquet"
                        )
                        .as_posix(),
                        dataset_id=f"mmlu_single_token_response_group{group}_train",
                    ),
                    tokenizer=tokenizer,
                )
            ),
            training_args=LoRATrainingArgs(num_train_epochs=20, per_device_train_batch_size=16),
            save_schedule=[1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20],
        ),
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.unload()

for group, path in enumerate(paths):
    logger.info(f"Single token evals on group {group}...")

    single_token_evaluator = MultiCheckpointEvaluator(
        config=MultiCheckpointEvaluatorConfig(
            checkpoints_dir=path,
            eval_dataset=[
                QADatasetAdapter(
                    dataset=MMLUSingleTokenResponseDataset(
                        config=QADatasetConfig(
                            path=Path(__file__)
                            .parent.joinpath(
                                f"../../../../data/out/splits/single_token_entropy/mmlu/phi4mini/group{j}_test.parquet"
                            )
                            .as_posix(),
                            dataset_id=f"mmlu_single_token_response_group{j}_test",
                        ),
                        tokenizer=tokenizer,
                    )
                )
                for j in range(6)
            ],
            base_model_id=MODEL_NAME,
            generation=GenerationConfig(max_new_tokens=1, max_batch_size=401),
            summary_filename="summary_single_token.json",
        ),
        tokenizer=tokenizer,
    )
    single_token_evaluator.evaluate_all()

for group, path in enumerate(paths):
    logger.info(f"CoT token evals on group {group}...")

    cot_evaluator = MultiCheckpointEvaluator(
        config=MultiCheckpointEvaluatorConfig(
            checkpoints_dir=path,
            eval_dataset=[
                QADatasetAdapter(
                    dataset=MMLUCoTResponseDataset(
                        config=QADatasetConfig(
                            path=Path(__file__)
                            .parent.joinpath(
                                f"../../../../data/out/splits/single_token_entropy/mmlu/phi4mini/group{j}_test.parquet"
                            )
                            .as_posix(),
                            dataset_id=f"mmlu_cot_response_group{j}_test",
                        ),
                        tokenizer=tokenizer,
                    )
                )
                for j in range(6)
            ],
            base_model_id=MODEL_NAME,
            generation=GenerationConfig(max_new_tokens=8192, max_batch_size=401),
            summary_filename="summary_cot.json",
        ),
        tokenizer=tokenizer,
    )
    cot_evaluator.evaluate_all()
