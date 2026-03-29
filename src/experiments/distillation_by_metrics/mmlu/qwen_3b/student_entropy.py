from pathlib import Path

from transformers import AutoTokenizer

from core.complexity_estimation.complexity_estimation_runner import ModelGenerateConfig
from core.complexity_estimation.entropy.single_token_entropy_estimator import SingleTokenEntropyEstimator
from core.dataset_samplers.entropy_ratio_sampler import BaseDatasetSamplerConfig
from core.dataset_samplers.student_entropy_sampler import StudentEntropySampler
from core.datasets.causal_dataset_adapter import CausalDatasetAdapter
from core.datasets.mmlu.mmlu_reasoning_response_dataset import MMLUReasoningResponseDataset
from core.datasets.mmlu.mmlu_single_token_response_dataset import MMLUSingleTokenResponseDataset
from core.datasets.qa_dataset import QADatasetConfig
from core.datasets.qa_dataset_adapter import QADatasetAdapter
from core.training.lora_trainer import LoRATrainingArgs
from core.training.resampling_trainer import ResamplingTrainer, ResamplingTrainerConfig

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# TODO: Upd dataset paths
trainer = ResamplingTrainer(
    config=ResamplingTrainerConfig(
        training_args=LoRATrainingArgs(
            num_train_epochs=20,
            per_device_train_batch_size=32,
        ),
        out_path=Path(__file__)
        .parent.joinpath("../../../../artifacts/train_pipeline/mmlu/qwen_3b/student_entropy")
        .as_posix(),
        model_id=MODEL_NAME,
        train_dataset=CausalDatasetAdapter(
            dataset=MMLUReasoningResponseDataset(
                config=QADatasetConfig(
                    path=Path(__file__).parent.joinpath("../../../../../data/source/mmlu_pro_stem.parquet").as_posix(),
                    dataset_id="mmlu_qwen_3b_student_entropy",
                ),
                tokenizer=tokenizer,
            ),
            dataset_sampler=StudentEntropySampler(BaseDatasetSamplerConfig(top_k=1024)),
        ),
        save_schedule=[1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20],
        complexity_evaluation_dataset=QADatasetAdapter(
            dataset=MMLUSingleTokenResponseDataset(
                config=QADatasetConfig(
                    path=Path(__file__).parent.joinpath("../../../../../data/source/mmlu_pro_stem.parquet").as_posix(),
                    dataset_id="mmlu_original",
                ),
                tokenizer=tokenizer,
            )
        ),
        complexity_estimator=SingleTokenEntropyEstimator(),
        complexity_estimation_runner_generation_config=ModelGenerateConfig(max_new_tokens=1),
    ),
    tokenizer=tokenizer,
)
