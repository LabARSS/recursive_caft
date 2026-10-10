from experiments.distillation_on_synthetic_traces.mmlu.shared import run

run(
    model_name="phi4_mini",
    relative_out_path="./distill_w_explained/phi4_mini_head_truncated8192",
    train_dataset="train_distilled_w_explained_deepseek_v4_flash_head_truncated8192",
    save_schedule=[2, 5, 10, 15, 20],
    max_thinking_tokens=4096,
)
