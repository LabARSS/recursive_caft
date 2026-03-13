from pathlib import Path

from experiments.distill.train_branches.train_cleaned_b_new import run_cleaned_b_training


def main():
    run_cleaned_b_training(
        model_name="Qwen/Qwen2.5-3B-Instruct",
        train_data_path=Path(__file__).resolve().parents[4]
        / "data"
        / "out"
        / "distillation"
        / "mmlu_synth_gptoss_b_t0_8_cleaned_32b_prompt1.parquet",
        eval_split_model="qwen_3b",
        eval_groups=6,
        per_device_train_batch_size=1,
        effective_train_batch_size=120,
        per_device_eval_batch_size=2,
        eval_accumulation_steps=1,
        num_train_epochs=20,
        learning_rate=1e-4,
        generation_max_new_tokens=768,
        run_tag="runb_cot_prompt1_full20_gpu0_h100_v1",
    )


if __name__ == "__main__":
    main()
