from core.dataset_samplers.base_sampler import BaseDatasetSamplerConfig
from core.dataset_samplers.student_entropy_sampler import StudentEntropySampler
from core.datasets.causal_dataset_adapter import CausalDatasetAdapter
from core.datasets.mmlu.mmlu_reasoning_response_dataset import MMLUReasoningResponseDataset
from core.datasets.mmlu.mmlu_single_token_response_dataset import QADatasetConfig
from core.training.lora_trainer import LoRASpecificTrainingArgsLarge
from experiments.distillation_by_metrics.mmlu.shared import run

run(
    model_name="phi4_mini",
    relative_out_path="./student_entropy_once/phi4_mini_head_truncated8192_topk2048_lora_large_shuffle",
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
        dataset_sampler=StudentEntropySampler(BaseDatasetSamplerConfig(top_k=2048)),
    ),
    save_schedule=[5, 10, 25, 40, 50],
    resampling_schedule=[0],
    lora_training_args=LoRASpecificTrainingArgsLarge(),
    shuffle=True,
)
