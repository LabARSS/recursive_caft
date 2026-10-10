from experiments.distillation_on_synthetic_traces.mmlu.shared import run

run(
    model_name="qwen_3b",
    relative_out_path="./direct_reasoning_trace/qwen_3b_head_truncated3072",
    train_dataset="train_distilled_deepseek_v4_flash_regenerate_incorrect_w_large_head_truncated3072",
    save_schedule=[2, 5, 10, 15, 20],
    max_thinking_tokens=2048,
)
