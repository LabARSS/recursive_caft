"""
Full 20-epoch SFT run for cleaned Branch B prompt 1.

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run python src/experiments/distill/train_branches/train_cleaned_b_full20_prompt1.py
"""

from experiments.distill.train_branches.train_cleaned_b_new import run_cleaned_b_training


def main():
    run_cleaned_b_training(
        prompt_id=1,
        eval_split_dir="data/out/splits/single_token_entropy/mmlu/qwen_3b",
        eval_groups=6,
        per_device_train_batch_size=1,
        effective_train_batch_size=120,
        num_train_epochs=20,
        learning_rate=1e-4,
        run_tag="full20_gpu0_eval6_v1",
    )


if __name__ == "__main__":
    main()
