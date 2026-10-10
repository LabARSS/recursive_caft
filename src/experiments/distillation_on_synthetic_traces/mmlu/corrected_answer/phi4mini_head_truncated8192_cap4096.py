from experiments.distillation_on_synthetic_traces.mmlu.shared import run

run(
    model_name="phi4_mini",
    relative_out_path="./corrected_answer/phi4_mini_head_truncated8192",
    train_dataset="train_corrected_answer_deepseek_v4_pro_and_others_head_truncated8192",
    save_schedule=[2, 5, 10, 15, 20],
    max_thinking_tokens=4096,
    skip_missing_checkpoints=True,
)
