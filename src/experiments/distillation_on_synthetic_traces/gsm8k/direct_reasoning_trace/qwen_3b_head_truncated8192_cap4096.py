from experiments.distillation_on_synthetic_traces.mmlu.shared import run

run(
    model_name="qwen_3b",
    relative_out_path="./direct_reasoning_trace/qwen_3b_head_truncated8192",
    train_dataset="gsm8k_distilled_deepseek_v4_flash_extend_w_pro_head8192_clean",
    save_schedule=[2, 5, 10, 15, 20],
    max_thinking_tokens=4096,
)
