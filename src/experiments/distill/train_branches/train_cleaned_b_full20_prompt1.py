"""
Full 20-epoch SFT run for Branch B prompt 1.

Usage:
    CUDA_VISIBLE_DEVICES=0,2 uv run torchrun --nproc_per_node=2 src/experiments/distill/train_branches/train_cleaned_b_full20_prompt1.py
"""

from core.training.branch_b_training import run_branch_b_training


def main():
    run_branch_b_training(
        prompt_id=1,
        eval_split_dir="data/out/splits/single_token_entropy/mmlu/qwen_3b",
        eval_groups=6,
        per_device_train_batch_size=2,
        num_train_epochs=20,
        run_tag="full20_eval6_v2",
    )


if __name__ == "__main__":
    main()
