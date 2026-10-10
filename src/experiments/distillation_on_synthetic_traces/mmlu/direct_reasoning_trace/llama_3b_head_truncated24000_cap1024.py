from experiments.distillation_on_synthetic_traces.mmlu.shared import run

run(
    model_name="llama_3b",
    relative_out_path="./direct_reasoning_trace/llama_3b_head_truncated24000",
    train_dataset="train_distilled_deepseek_v4_flash_regenerate_incorrect_w_large_head_truncated24000",
    save_schedule=[1, 3, 5, 10, 15, 20],
    max_thinking_tokens=1024,
)
