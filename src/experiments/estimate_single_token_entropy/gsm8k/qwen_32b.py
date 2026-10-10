from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer

from core.complexity_estimation.complexity_estimation_runner import (
    ComplexityEstimationRunner,
    ComplexityEstimationRunnerConfig,
    ModelGenerateConfig,
)
from core.complexity_estimation.entropy.multi_token_entropy_estimator import MultiTokenEntropyEstimator
from core.datasets.gsm8k.gsm8k_direct_response_dataset import GSM8KDirectResponseDataset, QADatasetConfig
from core.datasets.qa_dataset_adapter import QADatasetAdapter
from core.utils.device import DEVICE_MAP

MODEL_NAME = "Qwen/Qwen2.5-32B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, trust_remote_code=True, device_map=DEVICE_MAP, torch_dtype="auto"
)

ComplexityEstimationRunner(
    config=ComplexityEstimationRunnerConfig(
        out_path=str(
            Path(__file__).parent.joinpath("../../../../data/out/single_token_entropy/gsm8k_qwen_32b.parquet")
        ),
        answer_field_name="model_answer",
        answer_correctness_field_name="model_answer_correct",
        generate_config=ModelGenerateConfig(max_new_tokens=10),
        save_every=100,
    ),
    complexity_estimator=MultiTokenEntropyEstimator(),
).estimate(
    QADatasetAdapter(
        GSM8KDirectResponseDataset(
            tokenizer,
            QADatasetConfig(
                path=str(Path(__file__).parent.joinpath("../../../../data/source/gsm8k/gsm8k_train.parquet"))
            ),
        )
    ),
    model,
)
