from core.dataset_samplers.annealed_proxy_entropy_sampler import (
    AnnealedProxyEntropySampler,
    AnnealedProxyEntropySamplerConfig,
)
from experiments.distillation_by_metrics.mmlu.shared import (
    get_merged_adapter_with_data_mix_from_factory,
    run,
)

INITIAL_PROXY_WEIGHT = 0.8
REFERENCE_EPOCH = 100
REFERENCE_FRACTION = 0.01
SAVE_SCHEDULE = [10, 20, 50, 80, 100, 150, 200]
TRAIN_DATASET = "train_corrected_answer_deepseek_v4_pro_and_others_head_truncated8192"


def run_experiment(model_name: str, seed: int) -> None:
    suffix = "" if seed == 42 else f"_seed{seed}"

    def sampler(top_k: int) -> AnnealedProxyEntropySampler:
        return AnnealedProxyEntropySampler(
            AnnealedProxyEntropySamplerConfig(
                top_k=top_k,
                initial_proxy_weight=INITIAL_PROXY_WEIGHT,
                reference_epoch=REFERENCE_EPOCH,
                reference_fraction=REFERENCE_FRACTION,
                seed=seed,
            )
        )

    run(
        model_name=model_name,
        relative_out_path=f"./combined_entropy_proportional/{model_name}_head_truncated8192{suffix}",
        train_dataset=TRAIN_DATASET,
        train_dataset_adapter=get_merged_adapter_with_data_mix_from_factory(sampler),
        save_schedule=SAVE_SCHEDULE,
        shuffle=True,
        seed=seed,
    )
