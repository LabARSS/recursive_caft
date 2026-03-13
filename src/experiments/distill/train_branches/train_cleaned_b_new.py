import ast
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from datasets import Dataset
from transformers import AutoTokenizer

from core.datasets.causal_dataset import CausalDatasetConfig
from core.datasets.causal_dataset_adapter import CausalDatasetAdapter
from core.datasets.distillation.distillation_branch_b_cot_dataset import DistillationBranchBCoTDataset
from core.datasets.mmlu.mmlu_cot_response_dataset import MMLUCoTResponseDataset
from core.datasets.qa_dataset import QADatasetConfig
from core.prompts.mmlu_cot_answer import answer_marker
from core.training.base_trainer import is_main_process
from core.training.lora_trainer import LoRATrainer, LoRATrainerConfig, LoRATrainingArgs, LoRASpecificTrainingArgs

PROJECT_ROOT = Path(__file__).resolve().parents[4]
FINAL_ANSWER_RE = re.compile(r"\[\[\s*([a-jA-J])\s*\]\]")


class QuestionHoldoutCausalDatasetAdapter(CausalDatasetAdapter):
    def __init__(self, dataset, held_out_question_ids: set[str]):
        super().__init__(dataset)
        self.held_out_question_ids = held_out_question_ids

    def process_dataset(self, path_override: str | None = None) -> Dataset:
        dataset = self._load_ds(path_override)
        dataset = dataset.filter(
            lambda row: str(row["input"]["question_id"]) not in self.held_out_question_ids,
            num_proc=4,
        )
        return dataset.map(
            lambda row: self.process_row(row).model_dump(),
            num_proc=4,
            remove_columns=dataset.column_names,
            load_from_cache_file=False,
        )


def _normalize_text(text: str) -> str:
    normalized = str(text).replace("“", '"').replace("”", '"').replace("’", "'")
    normalized = normalized.strip()
    return re.sub(r"\s+", " ", normalized)


def _normalize_split_options(raw_options) -> list[str]:
    options = raw_options
    if isinstance(options, str):
        options = ast.literal_eval(options)
    return [_normalize_text(option) for option in options if option]


def _normalize_distillation_options(raw_options: dict) -> list[str]:
    values = []
    for _, value in sorted(raw_options.items()):
        if value:
            values.append(_normalize_text(value))
    return values


def _parse_final_answer(text: str) -> str | None:
    matches = FINAL_ANSWER_RE.findall(text)
    if not matches:
        return None
    return matches[-1].lower()


def _load_reasoning_source(train_data_path: Path) -> pd.DataFrame:
    rows = pq.read_table(train_data_path, columns=["input", "output"]).to_pylist()
    records = []
    for row in rows:
        input_row = row["input"]
        output_row = row.get("output") or {}
        records.append(
            {
                "question_id": str(input_row["question_id"]),
                "question_source": input_row["question"],
                "options_source": input_row["options"],
                "gold_source": str(input_row["gold"]).strip().lower(),
                "reasoning": str(output_row.get("thinking") or "").strip(),
            }
        )
    return pd.DataFrame(records)


def _load_eval_split_paths(split_model: str, groups: int) -> dict[str, Path]:
    split_root = PROJECT_ROOT / "data" / "out" / "splits" / "single_token_entropy" / split_model
    if not split_root.exists():
        raise FileNotFoundError(f"Split folder not found: {split_root}")

    split_paths = {}
    for index in range(groups):
        group_name = f"g{index}"
        group_path = split_root / f"group{index}_test.parquet"
        if not group_path.exists():
            raise FileNotFoundError(f"Expected split file is missing: {group_path}")
        split_paths[group_name] = group_path
    return split_paths


def _validate_eval_join(merged: pd.DataFrame, group_name: str) -> None:
    missing_rows = merged.loc[merged["question_source"].isna(), "question_id"].tolist()
    if missing_rows:
        raise RuntimeError(f"{group_name}: missing reasoning rows for question_ids={missing_rows[:5]}")

    bad_question_ids = []
    bad_option_ids = []
    bad_answer_ids = []
    for row in merged.itertuples(index=False):
        if _normalize_text(row.question) != _normalize_text(row.question_source):
            bad_question_ids.append(row.question_id)
        if _normalize_split_options(row.options) != _normalize_distillation_options(row.options_source):
            bad_option_ids.append(row.question_id)
        if str(row.answer).strip().lower() != str(row.gold_source).strip().lower():
            bad_answer_ids.append(row.question_id)

    if bad_question_ids:
        raise RuntimeError(f"{group_name}: question mismatch after join for question_ids={bad_question_ids[:5]}")
    if bad_option_ids:
        raise RuntimeError(f"{group_name}: options mismatch after join for question_ids={bad_option_ids[:5]}")
    if bad_answer_ids:
        raise RuntimeError(f"{group_name}: answer mismatch after join for question_ids={bad_answer_ids[:5]}")


def _prepared_root(split_model: str, run_tag: str) -> Path:
    tag = run_tag if run_tag else "default"
    return PROJECT_ROOT / "artifacts" / "sft_distill" / "prepared" / f"cot_prompt1_{split_model}_{tag}"


def _metadata_path(prepared_root: Path) -> Path:
    return prepared_root / "metadata.json"


def _write_metadata(prepared_root: Path, eval_groups: list[str], eval_question_ids: set[str]) -> None:
    metadata = {
        "eval_groups": eval_groups,
        "eval_question_ids": sorted(eval_question_ids),
    }
    tmp_path = prepared_root / "metadata.json.tmp"
    tmp_path.write_text(json.dumps(metadata, indent=2))
    tmp_path.replace(_metadata_path(prepared_root))


def _wait_for_prepared_files(prepared_root: Path, groups: int, timeout_seconds: int = 600) -> None:
    metadata_path = _metadata_path(prepared_root)
    deadline = time.time() + timeout_seconds
    expected_paths = [prepared_root / f"g{index}_cot_eval.parquet" for index in range(groups)]
    while time.time() < deadline:
        if metadata_path.exists() and all(path.exists() for path in expected_paths):
            return
        time.sleep(1)
    raise TimeoutError(f"Timed out waiting for prepared eval files in {prepared_root}")


def _prepare_eval_files(train_data_path: Path, split_model: str, groups: int, run_tag: str) -> tuple[dict[str, Path], set[str]]:
    prepared_root = _prepared_root(split_model, run_tag)
    prepared_root.mkdir(parents=True, exist_ok=True)
    metadata_path = _metadata_path(prepared_root)

    if is_main_process():
        if metadata_path.exists():
            metadata_path.unlink()

        reasoning_source = _load_reasoning_source(train_data_path)
        split_paths = _load_eval_split_paths(split_model, groups)
        eval_paths = {}
        eval_question_ids: set[str] = set()
        for group_name, group_path in split_paths.items():
            group_df = pd.read_parquet(group_path).copy()
            group_df["question_id"] = group_df["question_id"].astype(str)
            group_df["answer"] = group_df["answer"].astype(str).str.lower()
            eval_question_ids.update(group_df["question_id"].tolist())

            merged = group_df.merge(reasoning_source, on="question_id", how="left", validate="one_to_one")
            _validate_eval_join(merged, group_name)

            out_df = merged[["question_id", "question", "options", "answer", "base_cluster", "reasoning"]].copy()
            out_path = prepared_root / f"{group_name}_cot_eval.parquet"
            out_df.to_parquet(out_path, index=False)
            print(f"Prepared {group_name}: {len(out_df)} rows")
            eval_paths[group_name] = out_path

        _write_metadata(prepared_root, sorted(eval_paths), eval_question_ids)
    else:
        _wait_for_prepared_files(prepared_root, groups)

    metadata = json.loads(metadata_path.read_text())
    eval_paths = {
        group_name: prepared_root / f"{group_name}_cot_eval.parquet"
        for group_name in metadata["eval_groups"]
    }
    eval_question_ids = set(metadata["eval_question_ids"])
    return eval_paths, eval_question_ids


def _build_eval_datasets(tokenizer, train_data_path: Path, split_model: str, groups: int, run_tag: str):
    eval_paths, eval_question_ids = _prepare_eval_files(train_data_path, split_model, groups, run_tag)
    eval_datasets = {}
    for group_name, group_path in eval_paths.items():
        eval_datasets[group_name] = CausalDatasetAdapter(
            dataset=MMLUCoTResponseDataset(
                tokenizer=tokenizer,
                config=QADatasetConfig(path=str(group_path)),
            )
        )
    return eval_datasets, eval_question_ids


def _decode_text(tokenizer, token_ids) -> str:
    vocab_size = len(tokenizer)
    valid_ids = []
    for token_id in token_ids:
        token_id = int(token_id)
        if 0 <= token_id < vocab_size:
            valid_ids.append(token_id)
    if not valid_ids:
        return ""
    return tokenizer.decode(valid_ids, skip_special_tokens=True)


def compute_cot_accuracy_factory(tokenizer):
    def compute_cot_accuracy(eval_pred) -> dict[str, float]:
        predictions = eval_pred.predictions
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        prediction_array = np.asarray(predictions)
        if prediction_array.ndim == 3:
            prediction_array = prediction_array.argmax(axis=-1)

        labels = np.asarray(eval_pred.label_ids)
        correct = 0
        total = 0
        for prediction_ids, label_ids in zip(prediction_array, labels):
            gold_answer = _parse_final_answer(_decode_text(tokenizer, label_ids))
            if gold_answer is None:
                continue
            predicted_answer = _parse_final_answer(_decode_text(tokenizer, prediction_ids))
            total += 1
            if predicted_answer == gold_answer:
                correct += 1

        if total == 0:
            return {"accuracy": 0.0}
        return {"accuracy": correct / total}

    return compute_cot_accuracy


def _validate_train_sample(train_dataset: DistillationBranchBCoTDataset, train_data_path: Path, eval_question_ids: set[str]) -> None:
    rows = pq.read_table(train_data_path, columns=["input", "output"]).to_pylist()
    sample_row = None
    for row in rows:
        question_id = str(row["input"]["question_id"])
        if question_id not in eval_question_ids:
            sample_row = row
            break

    if sample_row is None:
        raise RuntimeError("No train rows left after question holdout")

    user_prompt = train_dataset.user_prompt(sample_row)
    assistant_response = train_dataset.assistant_response(sample_row)
    gold_answer = str(sample_row["input"]["gold"]).strip().lower()

    if "Correct answer:" in user_prompt:
        raise RuntimeError("Leak detected: train user prompt contains `Correct answer:`")
    if answer_marker[0] in user_prompt or answer_marker[1] in user_prompt:
        raise RuntimeError("Leak detected: train user prompt contains final answer marker")
    if _parse_final_answer(assistant_response) != gold_answer:
        raise RuntimeError(f"Train target format mismatch: expected final answer `{gold_answer}`")


def _validate_eval_sample(eval_datasets: dict[str, CausalDatasetAdapter]) -> None:
    first_group_name = sorted(eval_datasets)[0]
    dataset = eval_datasets[first_group_name].dataset
    df = pd.read_parquet(dataset.config.path)
    sample_row = df.iloc[0].to_dict()

    user_prompt = dataset.user_prompt(sample_row)
    assistant_response = dataset.assistant_response(sample_row)
    gold_answer = str(sample_row["answer"]).strip().lower()

    if answer_marker[0] in user_prompt or answer_marker[1] in user_prompt:
        raise RuntimeError(f"Leak detected: eval user prompt contains final answer marker for {first_group_name}")
    if _parse_final_answer(assistant_response) != gold_answer:
        raise RuntimeError(f"Eval target format mismatch in {first_group_name}: expected final answer `{gold_answer}`")


def run_cleaned_b_training(
    *,
    model_name: str,
    train_data_path: str | Path,
    eval_split_model: str = "qwen_3b",
    eval_groups: int = 6,
    per_device_train_batch_size: int = 1,
    effective_train_batch_size: int = 120,
    per_device_eval_batch_size: int = 2,
    eval_accumulation_steps: int = 1,
    num_train_epochs: int = 20,
    learning_rate: float = 1e-4,
    generation_max_new_tokens: int = 768,
    run_tag: str = "",
):
    train_data_path = Path(train_data_path)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    if not train_data_path.exists():
        raise FileNotFoundError(f"Train parquet not found: {train_data_path}")

    eval_datasets, eval_question_ids = _build_eval_datasets(
        tokenizer=tokenizer,
        train_data_path=train_data_path,
        split_model=eval_split_model,
        groups=eval_groups,
        run_tag=run_tag,
    )

    train_dataset = DistillationBranchBCoTDataset(
        tokenizer=tokenizer,
        config=CausalDatasetConfig(path=str(train_data_path)),
    )
    if is_main_process():
        _validate_train_sample(train_dataset, train_data_path, eval_question_ids)
        _validate_eval_sample(eval_datasets)

        total_rows = len(pq.read_table(train_data_path, columns=["input"]).to_pylist())
        train_rows = total_rows - len(eval_question_ids)
        print(f"Train rows after question holdout: {train_rows}/{total_rows}")

    run_suffix = f"_{run_tag}" if run_tag else ""
    out_path = PROJECT_ROOT / "artifacts" / "sft_distill" / f"branch_b_cleaned_prompt1{run_suffix}"
    save_schedule = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20]
    save_schedule = [epoch for epoch in save_schedule if epoch <= num_train_epochs]
    if num_train_epochs not in save_schedule:
        save_schedule.append(num_train_epochs)

    trainer = LoRATrainer(
        config=LoRATrainerConfig(
            out_path=str(out_path),
            model_id=model_name,
            train_dataset=QuestionHoldoutCausalDatasetAdapter(
                dataset=train_dataset,
                held_out_question_ids=eval_question_ids,
            ),
            eval_dataset=eval_datasets,
            compute_metrics=compute_cot_accuracy_factory(tokenizer),
            training_args=LoRATrainingArgs(
                num_train_epochs=num_train_epochs,
                per_device_train_batch_size=per_device_train_batch_size,
                effective_train_batch_size=effective_train_batch_size,
                per_device_eval_batch_size=per_device_eval_batch_size,
                eval_accumulation_steps=eval_accumulation_steps,
                learning_rate=learning_rate,
                warmup_ratio=0.06,
                torch_compile=False,
                gradient_checkpointing=True,
                eval_strategy="epoch",
                save_strategy="epoch",
                logging_steps=10,
                predict_with_generate=True,
                generation_num_beams=1,
                generation_max_new_tokens=generation_max_new_tokens,
                generation_do_sample=False,
                ddp_find_unused_parameters=False,
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
