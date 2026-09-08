from core.dataset_samplers.base_sampler import BaseDatasetSamplerConfig
from core.dataset_samplers.random_sampler import RandomSampler
from core.datasets.causal_dataset_adapter import CausalDatasetAdapter
from core.datasets.mmlu.mmlu_reasoning_response_dataset import (
    MMLUReasoningResponseDataset,
)
from core.datasets.mmlu.mmlu_single_token_response_dataset import QADatasetConfig
from experiments.distillation_by_metrics.mmlu.shared import (
    run,
)

run(
    model_name="qwen_3b",
    relative_out_path="./random/qwen_3b_head_truncated8192_sanity_check",
    train_dataset="train_corrected_answer_deepseek_v4_pro_and_others_head_truncated8192",
    train_dataset_adapter=CausalDatasetAdapter(
        dataset=MMLUReasoningResponseDataset(
            config=QADatasetConfig(
                path="should be overridden by SetResamplingPathCallback",
                dataset_id="train_corrected_answer_deepseek_v4_pro_and_others_head_truncated8192",
            ),
            # Will be overridden
            tokenizer=None,  # type: ignore
        ),
        dataset_sampler=RandomSampler(BaseDatasetSamplerConfig(top_k=9600)),
    ),
    save_schedule=[20, 50, 100, 150, 200],
    resampling_schedule=[0],
    shuffle=True,
)
