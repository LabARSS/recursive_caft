from experiments.distillation_on_synthetic_traces.mmlu.shared import run

run(
    model_name="phi4_mini",
    relative_out_path="./base_reasoner/phi4_mini",
    train_dataset="train_distilled_deepseek_v4_flash_regenerate_incorrect_w_large_head_truncated3072",
    save_schedule=[1],
    max_thinking_tokens=1024,
    run_evaluation=False,
)
