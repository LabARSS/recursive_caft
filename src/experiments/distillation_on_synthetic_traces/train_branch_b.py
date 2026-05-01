"""Training configuration for Branch B: Q + options + gold -> CoT explanation"""
from pathlib import Path
from core.training.sft_by_complexity_split.sft_by_single_complexity_split import (
    train_sft_by_complexity_split,
    batch_size_config,
)

if __name__ == "__main__":
    train_sft_by_complexity_split(
        out_path=str(Path(__file__).parent / "../../../../data/out/models/branch_b"),
        model_id="Qwen/Qwen2.5-3B-Instruct",
        train_df_path=str(Path(__file__).parent / "../../../../data/out/sft_data/branch_b_train.parquet"),
        test_df_paths=[str(Path(__file__).parent / "../../../../data/out/sft_data/branches_eval.tsv")],
        training_kwargs={
            "num_train_epochs": 3,
            "learning_rate": 1e-5,
            **batch_size_config(global_batch_size=256, per_device_train_batch_size=8),
            "warmup_steps": 50,
            "save_total_limit": None,  # Save all checkpoints for analysis
            "eval_strategy": "epoch",  # Evaluate every epoch
            "logging_strategy": "steps",  # Log frequently
            "logging_steps": 10,
        },
        use_lora=True,
        lora_kwargs={"r": 16, "lora_alpha": 32, "lora_dropout": 0.05, "use_rslora": True},
        distill_branch="B",
    )
