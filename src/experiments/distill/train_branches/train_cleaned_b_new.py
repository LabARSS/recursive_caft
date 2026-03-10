"""
Shared utilities for cleaned Branch B CoT training on the LoRA trainer.
"""

import string
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from core.datasets.causal_dataset import CausalDatasetConfig
from core.datasets.causal_dataset_adapter import CausalDatasetAdapter
from core.datasets.distillation.distillation_branch_b_cot_dataset import DistillationBranchBCoTDataset
from core.datasets.mmlu.mmlu_single_token_response_dataset import MMLUSingleTokenResponseDataset
from core.datasets.qa_dataset import QADatasetConfig
from core.training.lora_trainer import LoRATrainer, LoRATrainerConfig, LoRATrainingArgs, LoRASpecificTrainingArgs

PROJECT_ROOT = Path(__file__).resolve().parents[4]
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"


def _extract_first_option_letter(text: str) -> str | None:
    for ch in text.lower():
        if ch in string.ascii_lowercase:
            return ch
    return None


def build_eval_datasets(tokenizer, split_model: str, groups: int) -> dict[str, CausalDatasetAdapter]:
    split_root = PROJECT_ROOT / f"data/out/splits/single_token_entropy/{split_model}"
    if not split_root.exists():
        raise FileNotFoundError(f"Split folder not found: {split_root}")

    eval_datasets: dict[str, CausalDatasetAdapter] = {}
    for i in range(groups):
        group_path = split_root / f"group{i}_test.parquet"
        if not group_path.exists():
            raise FileNotFoundError(
                f"Expected split file is missing: {group_path}. "
                f"Requested groups={groups}, split_model={split_model}"
            )

        eval_datasets[f"g{i}"] = CausalDatasetAdapter(
            dataset=MMLUSingleTokenResponseDataset(
                tokenizer=tokenizer,
                config=QADatasetConfig(path=str(group_path)),
            )
        )

    return eval_datasets


def compute_single_token_accuracy_factory(tokenizer):
    vocab_size = len(tokenizer)

    def compute_single_token_accuracy(eval_pred) -> dict[str, float]:
        predictions = eval_pred.predictions
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        labels = eval_pred.label_ids
        if predictions is None or labels is None:
            return {"accuracy": 0.0}

        pred_arr = np.asarray(predictions)
        if pred_arr.ndim == 3:
            pred_token_ids = pred_arr.argmax(axis=-1)
        else:
            pred_token_ids = pred_arr
        labels = np.asarray(labels)

        correct = 0
        total = 0
        for sample_pred_ids, sample_labels in zip(pred_token_ids, labels):
            label_positions = np.where(sample_labels != -100)[0]
            if label_positions.size == 0:
                continue

            first_label_pos = int(label_positions[0])
            pred_pos = max(first_label_pos - 1, 0)

            gold_tokens = [
                tok_id
                for tok_id in (int(sample_labels[idx]) for idx in label_positions[:8])
                if 0 <= tok_id < vocab_size
            ]
            pred_tokens = [
                tok_id
                for tok_id in (int(tok) for tok in sample_pred_ids[pred_pos : pred_pos + 8])
                if 0 <= tok_id < vocab_size
            ]
            if not gold_tokens or not pred_tokens:
                continue

            try:
                gold_text = tokenizer.decode(gold_tokens, skip_special_tokens=True)
                pred_text = tokenizer.decode(pred_tokens, skip_special_tokens=True)
            except (OverflowError, ValueError):
                continue

            gold_letter = _extract_first_option_letter(gold_text)
            pred_letter = _extract_first_option_letter(pred_text)

            if gold_letter is None or pred_letter is None:
                continue

            correct += int(gold_letter == pred_letter)
            total += 1

        return {"accuracy": float(correct) / float(total) if total else 0.0}

    return compute_single_token_accuracy


def preprocess_logits_for_metrics(logits: torch.Tensor | tuple[torch.Tensor, ...], labels: torch.Tensor):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def validate_train_prompt_format(train_dataset: DistillationBranchBCoTDataset, train_data_path: Path) -> None:
    first_row = pd.read_parquet(train_data_path).iloc[0].to_dict()

    user_prompt = train_dataset.user_prompt(first_row)
    assistant_response = train_dataset.assistant_response(first_row)
    gold = str(first_row["input"]["gold"]).strip().lower()
    predicted_first_letter = _extract_first_option_letter(assistant_response[:16] if assistant_response else "")

    if "Correct answer:" in user_prompt:
        raise RuntimeError("Leak detected: train user_prompt contains `Correct answer:`")
    if predicted_first_letter != gold:
        raise RuntimeError(
            f"Train target format mismatch: expected first option letter `{gold}`, got `{predicted_first_letter}`"
        )


def run_cleaned_b_training(
    *,
    prompt_id: int,
    eval_split_model: str = "qwen_3b",
    eval_groups: int = 5,
    per_device_train_batch_size: int = 1,
    effective_train_batch_size: int = 120,
    per_device_eval_batch_size: int = 2,
    eval_accumulation_steps: int = 1,
    num_train_epochs: int = 20,
    learning_rate: float = 1e-4,
    run_tag: str = "",
):
    if effective_train_batch_size % per_device_train_batch_size != 0:
        raise ValueError(
            "effective_train_batch_size must be divisible by per_device_train_batch_size "
            f"(got {effective_train_batch_size} and {per_device_train_batch_size})"
        )
    if num_train_epochs < 1:
        raise ValueError(f"num_train_epochs must be >= 1, got {num_train_epochs}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    train_data_path = PROJECT_ROOT / f"data/out/distillation/mmlu_synth_gptoss_b_t0_8_cleaned_32b_prompt{prompt_id}.parquet"
    eval_datasets = build_eval_datasets(
        tokenizer=tokenizer,
        split_model=eval_split_model,
        groups=eval_groups,
    )
    train_dataset = DistillationBranchBCoTDataset(
        tokenizer=tokenizer,
        config=CausalDatasetConfig(path=str(train_data_path)),
    )
    validate_train_prompt_format(train_dataset, train_data_path)

    run_suffix = f"_{run_tag}" if run_tag else ""
    out_path = PROJECT_ROOT / f"artifacts/sft_distill/branch_b_cleaned_prompt{prompt_id}{run_suffix}"
    save_schedule = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20]
    save_schedule = sorted(set(epoch for epoch in save_schedule if epoch <= num_train_epochs) | {num_train_epochs})

    trainer = LoRATrainer(
        config=LoRATrainerConfig(
            out_path=str(out_path),
            model_id=MODEL_NAME,
            train_dataset=CausalDatasetAdapter(dataset=train_dataset),
            eval_dataset=eval_datasets,
            compute_metrics=compute_single_token_accuracy_factory(tokenizer),
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            training_args=LoRATrainingArgs(
                num_train_epochs=num_train_epochs,
                per_device_train_batch_size=per_device_train_batch_size,
                effective_train_batch_size=effective_train_batch_size,
                per_device_eval_batch_size=per_device_eval_batch_size,
                eval_accumulation_steps=eval_accumulation_steps,
                learning_rate=learning_rate,
                warmup_ratio=0.06,
                torch_compile=False,
                eval_strategy="epoch",
                save_strategy="epoch",
                logging_steps=10,
            ),
            lora_training_args=LoRASpecificTrainingArgs(
                r=16,
                alpha=32,
                lora_dropout=0.05,
                use_rslora=True,
            ),
            save_schedule=save_schedule,
        ),
        tokenizer=tokenizer,
    )
    trainer.train()
