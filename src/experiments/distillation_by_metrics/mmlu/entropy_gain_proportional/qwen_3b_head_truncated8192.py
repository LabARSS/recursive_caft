from core.complexity_estimation.entropy.single_token_entropy_with_random_estimator import (
    SingleTokenEntropyWithRandomEstimator,
)
from core.dataset_samplers.entropy_gain_proportional_sampler import EntropyGainProportionalSampler
from experiments.distillation_by_metrics.mmlu.shared import get_merged_adapter_with_data_mix, run

run(
    model_name="qwen_3b",
    relative_out_path="./entropy_gain_proportional/qwen_3b_head_truncated8192",
    train_dataset="train_corrected_answer_deepseek_v4_pro_and_others_head_truncated8192",
    train_dataset_adapter=get_merged_adapter_with_data_mix(EntropyGainProportionalSampler),
    save_schedule=[5, 10, 20, 40, 60, 80, 100],
    complexity_estimator_override=SingleTokenEntropyWithRandomEstimator(),
)
