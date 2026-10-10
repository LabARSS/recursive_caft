from core.complexity_estimation.entropy.single_token_entropy_with_random_estimator import (
    SingleTokenEntropyWithRandomEstimator,
)
from core.dataset_samplers.student_entropy_proportional_sampler import StudentEntropyProportionalSampler
from experiments.distillation_by_metrics.mmlu.shared import get_merged_adapter_with_data_mix, run

run(
    model_name="llama_3b",
    relative_out_path="./student_entropy_proportional/llama_3b_head_truncated8192_shuffle",
    train_dataset="train_corrected_answer_deepseek_v4_pro_and_others_head_truncated8192",
    train_dataset_adapter=get_merged_adapter_with_data_mix(StudentEntropyProportionalSampler),
    save_schedule=[20, 50, 100, 150, 200],
    complexity_estimator_override=SingleTokenEntropyWithRandomEstimator(),
    shuffle=True,
)
