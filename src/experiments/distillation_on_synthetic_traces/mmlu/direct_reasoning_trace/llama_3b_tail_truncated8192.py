from experiments.distillation_on_synthetic_traces.mmlu.shared import run

run(
    model_name="llama_3b",
    relative_out_path="./direct_reasoning_trace/llama_3b_tail_truncated8192",
    train_dataset="train_distilled_deepseek_v4_flash_regenerate_incorrect_w_large_tail_truncated8192",
    save_schedule=[2, 5, 10, 15, 20],
)
