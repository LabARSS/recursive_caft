from collections.abc import Callable
from pathlib import Path

from transformers import AutoTokenizer

from core.complexity_estimation.complexity_estimator import BaseComplexityEstimator
from core.complexity_estimation.entropy.single_token_entropy_estimator import SingleTokenEntropyEstimator
from core.dataset_samplers.base_sampler import BaseDatasetSampler, BaseDatasetSamplerConfig
from core.datasets.causal_dataset_adapter import CausalDatasetAdapter
from core.datasets.merged_dataset_adapter import MergedDatasetAdapter
from core.datasets.mmlu.mmlu_reasoning_response_dataset import MMLUReasoningResponseDataset
from core.datasets.mmlu.mmlu_single_token_response_dataset import MMLUSingleTokenResponseDataset
from core.datasets.qa_dataset import QADatasetConfig
from core.datasets.qa_dataset_adapter import QADatasetAdapter
from core.evaluation.multi_checkpoint_evaluator import (
    GenerationConfig,
    MultiCheckpointEvaluator,
    MultiCheckpointEvaluatorConfig,
)
from core.training.base_trainer import AbstractDatasetAdapter
from core.training.lora_trainer import (
    LoRASpecificTrainingArgs,
    LoRATrainingArgs,
    phi4_mini_lora_target_modules,
)
from core.training.resampling_trainer import ModelGenerateConfig, ResamplingTrainer, ResamplingTrainerConfig
from core.training.thinking_tokens import setup_thinking_tokens
from core.utils.datasets import add_average_column, merge_mmlu_on_question_id


def run(
    model_name: str,
    relative_out_path: str,
    train_dataset: str,
    train_dataset_adapter: AbstractDatasetAdapter,
    save_schedule: list[int],
    skip_missing_checkpoints: bool = False,
    complexity_estimator_override: BaseComplexityEstimator | None = None,
    resampling_schedule: list[int] | None = None,
    lora_training_args: LoRASpecificTrainingArgs | None = None,
    shuffle: bool = False,
    seed: int = 42,
):
    MODEL_NAME = Path(__file__).parent.joinpath(f"../../../../artifacts/base_models_v0/{model_name}").as_posix()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    setup_thinking_tokens(tokenizer)

    train_dataset_adapter.override_tokenizer(tokenizer)

    OUT_PATH = (
        Path(__file__)
        .parent.joinpath("../../../../artifacts/distillation_by_metrics/mmlu/")
        .joinpath(relative_out_path)
    )

    lora_training_args = lora_training_args or LoRASpecificTrainingArgs()
    lora_training_args.train_thinking_token_embeddings = True
    if model_name == "phi4_mini":
        lora_training_args.target_modules = phi4_mini_lora_target_modules

    TEACHER_ENTROPY_DATASET_PATH = OUT_PATH.joinpath("teacher_entropy.parquet")
    merge_mmlu_on_question_id(
        main_path=Path(__file__).parent.joinpath(f"../../../../data/out/splits/random/mmlu/{train_dataset}.parquet"),
        extra_paths=[
            Path(__file__).parent.joinpath(
                "../../../../data/out/single_token_entropy_normalized/mmlu_llama_70b.parquet"
            ),
            Path(__file__).parent.joinpath(
                "../../../../data/out/single_token_entropy_normalized/mmlu_qwen_72b.parquet"
            ),
        ],
        extra_columns=[
            {"entropy_value": "llama_70b_entropy_value"},
            {"entropy_value": "qwen_72b_entropy_value"},
        ],
        aggregation_function=lambda df: add_average_column(
            df, "llama_70b_entropy_value", "qwen_72b_entropy_value", "teacher_entropy"
        ),
        save_path=TEACHER_ENTROPY_DATASET_PATH,
    )

    trainer = ResamplingTrainer(
        config=ResamplingTrainerConfig(
            training_args=LoRATrainingArgs(
                num_train_epochs=save_schedule[-1], per_device_train_batch_size=2, seed=seed, data_seed=seed
            ),
            lora_training_args=lora_training_args,
            # packing=PackingConfig(budget=packing_budget(model_name)),
            save_schedule=save_schedule,
            resampling_schedule=resampling_schedule,
            shuffle=shuffle,
            out_path=OUT_PATH.as_posix(),
            model_id=MODEL_NAME,
            train_dataset=train_dataset_adapter,
            complexity_evaluation_dataset=QADatasetAdapter(
                dataset=MMLUSingleTokenResponseDataset(
                    config=QADatasetConfig(
                        path=TEACHER_ENTROPY_DATASET_PATH.as_posix(),
                        dataset_id="mmlu_teacher_entropy",
                    ),
                    tokenizer=tokenizer,
                )
            ),
            complexity_estimator=complexity_estimator_override or SingleTokenEntropyEstimator(),
            complexity_estimation_runner_generation_config=ModelGenerateConfig(max_new_tokens=1),
        ),
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.unload()

    for max_thinking_tokens in [4096, 2048]:
        eval_dataset_id = f"mmlu_random_test_cap{max_thinking_tokens}"
        summary_filename = f"summary_reasoning_evals_cap{max_thinking_tokens}.json"

        cot_evaluator = MultiCheckpointEvaluator(
            config=MultiCheckpointEvaluatorConfig(
                checkpoints_dir=OUT_PATH.as_posix(),
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


def get_merged_adapter_with_data_mix(sampler_cls: type[BaseDatasetSampler]) -> MergedDatasetAdapter:
    return get_merged_adapter_with_data_mix_from_factory(
        lambda top_k: sampler_cls(BaseDatasetSamplerConfig(top_k=top_k))
    )


def get_merged_adapter_with_data_mix_from_factory(
    sampler_factory: Callable[[int], BaseDatasetSampler],
) -> MergedDatasetAdapter:
    return MergedDatasetAdapter(
        [
            CausalDatasetAdapter(
                dataset=MMLUReasoningResponseDataset(
                    config=QADatasetConfig(
                        path="should be overridden by SetResamplingPathCallback",
                        dataset_id="train_corrected_answer_deepseek_v4_pro_and_others_head_truncated8192",
                    ),
                    # Will be overridden
                    tokenizer=None,  # type: ignore
                ),
                dataset_sampler=sampler_factory(1024),
            ),
            # Mix in a smaller single-token-answer set selected by the same policy so the model
            # keeps answering with a single letter, keeping per-epoch entropy estimation stable.
            CausalDatasetAdapter(
                dataset=MMLUSingleTokenResponseDataset(
                    config=QADatasetConfig(
                        path="should be overridden by SetResamplingPathCallback",
                        dataset_id="mmlu_single_token_train",
                    ),
                    # Will be overridden
                    tokenizer=None,  # type: ignore
                ),
                dataset_sampler=sampler_factory(256),
            ),
        ]
    )
