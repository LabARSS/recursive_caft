"""
Full 20-epoch SFT run for cleaned Branch B prompt 1 on the LoRA trainer.

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run python src/experiments/distill/train_branches/train_cleaned_b_full20_prompt1.py
"""

from experiments.distill.train_branches.train_cleaned_b_new import run_cleaned_b_training

PROMPT_ID = 1
EVAL_SPLIT_MODEL = "qwen_3b"
EVAL_GROUPS = 6
PER_DEVICE_TRAIN_BATCH_SIZE = 1
EFFECTIVE_TRAIN_BATCH_SIZE = 120
PER_DEVICE_EVAL_BATCH_SIZE = 2
EVAL_ACCUMULATION_STEPS = 1
NUM_TRAIN_EPOCHS = 20
LEARNING_RATE = 1e-4
RUN_TAG = "full20_gpu0_eval6_v1"


def main():
    run_cleaned_b_training(
        prompt_id=PROMPT_ID,
        eval_split_model=EVAL_SPLIT_MODEL,
        eval_groups=EVAL_GROUPS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        effective_train_batch_size=EFFECTIVE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        eval_accumulation_steps=EVAL_ACCUMULATION_STEPS,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        learning_rate=LEARNING_RATE,
        run_tag=RUN_TAG,
    )


if __name__ == "__main__":
    main()
