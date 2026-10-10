from pathlib import Path

from transformers import AutoTokenizer

from core.datasets.causal_dataset_adapter import CausalDatasetAdapter
from core.datasets.mmlu.mmlu_reasoning_response_dataset import MMLUReasoningResponseDataset
from core.datasets.mmlu.mmlu_single_token_response_dataset import QADatasetConfig
from core.datasets.qa_dataset_adapter import QADatasetAdapter
from core.evaluation.multi_checkpoint_evaluator import (
    GenerationConfig,
    MultiCheckpointEvaluator,
    MultiCheckpointEvaluatorConfig,
)
from core.training.lora_trainer import LoRASpecificTrainingArgs, LoRATrainer, LoRATrainerConfig, LoRATrainingArgs
from core.training.thinking_tokens import setup_thinking_tokens
from core.utils.logger import logger

MODEL_NAME = Path(__file__).parent.joinpath("../../../../../artifacts/base_models_v0/llama_3b").as_posix()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
setup_thinking_tokens(tokenizer)

paths = [
    Path(__file__)
    .parent.joinpath(f"../../../../artifacts/sft_by_complexity_splits/mmlu/llama_3b_reasoning/group{group}")
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
                dataset=MMLUReasoningResponseDataset(
                    config=QADatasetConfig(
                        path=Path(__file__)
                        .parent.joinpath(
                            f"../../../../data/out/splits/single_token_entropy/mmlu/llama_3b/group{group}_train_corrected_answer.parquet"
                        )
                        .as_posix(),
                        dataset_id=f"mmlu_reasoning_response_group{group}_train",
                    ),
                    tokenizer=tokenizer,
                )
            ),
            training_args=LoRATrainingArgs(num_train_epochs=50, per_device_train_batch_size=8),
            lora_training_args=LoRASpecificTrainingArgs(train_thinking_token_embeddings=True),
            save_schedule=[5, 10, 20, 35, 50],
        ),
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.unload()

for group, path in enumerate(paths):
    logger.info(f"Reasoning token evals on group {group}...")

    cot_evaluator = MultiCheckpointEvaluator(
        config=MultiCheckpointEvaluatorConfig(
            checkpoints_dir=path,
            eval_dataset=[
                QADatasetAdapter(
                    dataset=MMLUReasoningResponseDataset(
                        config=QADatasetConfig(
                            path=Path(__file__)
                            .parent.joinpath(
                                f"../../../../data/out/splits/single_token_entropy/mmlu/llama_3b/group{j}_test.parquet"
                            )
                            .as_posix(),
                            dataset_id=f"mmlu_reasoning_response_group{j}_test",
                        ),
                        tokenizer=tokenizer,
                    ),
                    add_thinking_start_token=True,
                )
                for j in range(6)
            ],
            generation=GenerationConfig(max_new_tokens=8500, max_thinking_tokens=8192, max_batch_size=256),
            summary_filename="summary_reasoning_evals.json",
        ),
        tokenizer=tokenizer,
    )
    cot_evaluator.evaluate_all()
