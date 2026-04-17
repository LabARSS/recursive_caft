"""
Post-training verification for v0 embedding-init checkpoints.

Checks:
1. Pre-existing embedding rows in the v0 checkpoint are byte-identical to the
   raw HF base's rows (once both are loaded into the same dtype). Redundant
   with the in-trainer assertion, but catches any regression introduced by the
   save_pretrained/from_pretrained round-trip.
2. Single-token and CoT MMLU-Pro accuracy on `mmlu_pro_stem.parquet`, run
   against BOTH the v0 dir and the raw HF base. Any single-token accuracy
   delta should be zero under greedy decoding, since pre-existing embedding
   rows are untouched and the input contains no <think>/</think> tokens.

Writes `eval_summary.json` into the v0 dir and returns the summary dict.
"""

import json
from pathlib import Path

import torch
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.datasets.mmlu.mmlu_cot_response_dataset import MMLUCoTResponseDataset
from core.datasets.mmlu.mmlu_single_token_response_dataset import MMLUSingleTokenResponseDataset
from core.datasets.qa_dataset import QADatasetConfig
from core.datasets.qa_dataset_adapter import QADatasetAdapter
from core.evaluation.evaluator import Evaluator, EvaluatorConfig, GenerationConfig
from core.training.thinking_tokens import setup_thinking_tokens
from core.utils.logger import logger


class V0EmbeddingEvalConfig(BaseModel):
    v0_dir: str
    base_model_id: str
    mmlu_test_parquet: str
    eval_out_path: str


def run_v0_embedding_eval(config: V0EmbeddingEvalConfig) -> dict:
    v0_dir = Path(config.v0_dir)
    summary: dict = {"v0_dir": str(v0_dir), "base_model_id": config.base_model_id}

    summary["byte_identical_pre_existing_rows"] = _check_pre_existing_rows_match(
        v0_dir=v0_dir, base_model_id=config.base_model_id
    )

    summary["mmlu_single_token"] = _run_mmlu_eval(
        v0_dir=v0_dir,
        base_model_id=config.base_model_id,
        mmlu_test_parquet=config.mmlu_test_parquet,
        eval_out_path=config.eval_out_path,
        mode="single_token",
    )
    summary["mmlu_cot"] = _run_mmlu_eval(
        v0_dir=v0_dir,
        base_model_id=config.base_model_id,
        mmlu_test_parquet=config.mmlu_test_parquet,
        eval_out_path=config.eval_out_path,
        mode="cot",
    )

    summary_path = v0_dir / "eval_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Wrote eval summary to {summary_path}")

    return summary


def _check_pre_existing_rows_match(v0_dir: Path, base_model_id: str) -> dict:
    v0_tok = AutoTokenizer.from_pretrained(v0_dir.as_posix(), trust_remote_code=True)
    base_tok = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    num_new = len(v0_tok) - len(base_tok)
    assert num_new > 0, f"v0 vocab ({len(v0_tok)}) is not larger than base ({len(base_tok)})"

    v0_model = AutoModelForCausalLM.from_pretrained(v0_dir.as_posix(), torch_dtype=torch.bfloat16)
    base_model = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=torch.bfloat16)

    v0_rows = v0_model.get_input_embeddings().weight.detach().cpu()[:-num_new]
    base_rows = base_model.get_input_embeddings().weight.detach().cpu()

    identical = torch.equal(v0_rows, base_rows)
    max_abs_delta = (v0_rows - base_rows).abs().max().item() if not identical else 0.0

    del v0_model, base_model

    assert identical, (
        f"Pre-existing embedding rows differ between v0 and base (max abs delta {max_abs_delta}). "
        "Row-scoped backprop or save/load round-trip corrupted the embedding table."
    )
    return {"identical": True, "num_new_rows": num_new, "max_abs_delta": max_abs_delta}


def _run_mmlu_eval(
    v0_dir: Path,
    base_model_id: str,
    mmlu_test_parquet: str,
    eval_out_path: str,
    mode: str,
) -> dict:
    dataset_cls = MMLUSingleTokenResponseDataset if mode == "single_token" else MMLUCoTResponseDataset
    max_new_tokens = 1 if mode == "single_token" else 8192

    return {
        "v0": _eval_one(
            model_path=v0_dir.as_posix(),
            mmlu_test_parquet=mmlu_test_parquet,
            eval_out_path=f"{eval_out_path}/v0/{mode}",
            dataset_cls=dataset_cls,
            dataset_id=f"mmlu_pro_{mode}_v0",
            max_new_tokens=max_new_tokens,
            load_tokenizer_from=v0_dir.as_posix(),
        ),
        "base": _eval_one(
            model_path=base_model_id,
            mmlu_test_parquet=mmlu_test_parquet,
            eval_out_path=f"{eval_out_path}/base/{mode}",
            dataset_cls=dataset_cls,
            dataset_id=f"mmlu_pro_{mode}_base",
            max_new_tokens=max_new_tokens,
            load_tokenizer_from=base_model_id,
        ),
    }


def _eval_one(
    model_path: str,
    mmlu_test_parquet: str,
    eval_out_path: str,
    dataset_cls: type,
    dataset_id: str,
    max_new_tokens: int,
    load_tokenizer_from: str,
) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(load_tokenizer_from, trust_remote_code=True)
    setup_thinking_tokens(tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = dataset_cls(
        tokenizer=tokenizer,
        config=QADatasetConfig(path=mmlu_test_parquet, dataset_id=dataset_id),
    )
    adapter = QADatasetAdapter(dataset=dataset)

    evaluator = Evaluator(
        config=EvaluatorConfig(
            model_path=model_path,
            eval_dataset=adapter,
            out_path=eval_out_path,
            generation=GenerationConfig(max_new_tokens=max_new_tokens, max_batch_size=401),
        ),
        tokenizer=tokenizer,
    )
    [result] = evaluator.evaluate()
    return {
        "accuracy": result.accuracy,
        "total": result.total,
        "correct": result.correct,
        "num_truncated": result.num_truncated,
    }
