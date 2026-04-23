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

import pandas as pd
import torch
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.datasets.mmlu.mmlu_cot_response_dataset import MMLUCoTResponseDataset
from core.datasets.mmlu.mmlu_single_token_response_dataset import MMLUSingleTokenResponseDataset
from core.datasets.qa_dataset import QADatasetConfig
from core.datasets.qa_dataset_adapter import QADatasetAdapter
from core.evaluation.evaluator import Evaluator, EvaluatorConfig, GenerationConfig
from core.training.thinking_tokens import new_token_ids, setup_thinking_tokens
from core.utils.logger import logger


class V0EmbeddingEvalConfig(BaseModel):
    v0_dir: str
    base_model_id: str
    mmlu_test_parquet: str
    eval_out_path: str
    # Subsample size for the MMLU evals. None means use the full parquet.
    # Evals on 12k+ rows take too long in practice; 1000 gives ±1.5% 95% CI.
    mmlu_sample_size: int | None = 1000
    mmlu_sample_seed: int = 42


def run_v0_embedding_eval(config: V0EmbeddingEvalConfig) -> dict:
    v0_dir = Path(config.v0_dir)
    summary: dict = {"v0_dir": str(v0_dir), "base_model_id": config.base_model_id}

    summary["byte_identical_pre_existing_rows"] = _check_pre_existing_rows_match(
        v0_dir=v0_dir, base_model_id=config.base_model_id
    )

    mmlu_parquet = _prepare_mmlu_sample(
        src=Path(config.mmlu_test_parquet),
        eval_out_path=Path(config.eval_out_path),
        sample_size=config.mmlu_sample_size,
        seed=config.mmlu_sample_seed,
    )
    summary["mmlu_sample"] = {
        "path": str(mmlu_parquet),
        "size": config.mmlu_sample_size,
        "seed": config.mmlu_sample_seed,
    }

    summary["mmlu_single_token"] = _run_mmlu_eval(
        v0_dir=v0_dir,
        base_model_id=config.base_model_id,
        mmlu_test_parquet=mmlu_parquet.as_posix(),
        eval_out_path=config.eval_out_path,
        mode="single_token",
    )
    summary["mmlu_cot"] = _run_mmlu_eval(
        v0_dir=v0_dir,
        base_model_id=config.base_model_id,
        mmlu_test_parquet=mmlu_parquet.as_posix(),
        eval_out_path=config.eval_out_path,
        mode="cot",
    )

    summary_path = v0_dir / "eval_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Wrote eval summary to {summary_path}")

    return summary


def _prepare_mmlu_sample(
    src: Path,
    eval_out_path: Path,
    sample_size: int | None,
    seed: int,
) -> Path:
    if sample_size is None:
        return src

    eval_out_path.mkdir(parents=True, exist_ok=True)
    dst = eval_out_path / f"mmlu_sample_n{sample_size}_s{seed}.parquet"
    if dst.exists():
        logger.info(f"Using cached MMLU sample at {dst}")
        return dst

    df = pd.read_parquet(src)
    if sample_size >= len(df):
        logger.info(f"Sample size {sample_size} >= full parquet ({len(df)}); using full file")
        return src

    sampled = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)
    sampled.to_parquet(dst, index=False)
    logger.info(f"Wrote MMLU sample ({sample_size} rows, seed={seed}) to {dst}")
    return dst


def _check_pre_existing_rows_match(v0_dir: Path, base_model_id: str) -> dict:
    v0_tok = AutoTokenizer.from_pretrained(v0_dir.as_posix(), trust_remote_code=True)
    setup_thinking_tokens(v0_tok)
    new_ids = new_token_ids(v0_tok)

    v0_model = AutoModelForCausalLM.from_pretrained(v0_dir.as_posix(), torch_dtype=torch.bfloat16)
    base_model = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=torch.bfloat16)

    v0_weight = v0_model.get_input_embeddings().weight.detach().cpu()
    base_weight = base_model.get_input_embeddings().weight.detach().cpu()

    # For tight-vocab models v0 has +num_added rows vs base; for padded-vocab
    # models (Phi-4-mini) the row count matches. Mask out the new-token ids and
    # compare the remainder — valid in both cases.
    v0_rows, base_rows = v0_weight.shape[0], base_weight.shape[0]
    num_new = v0_rows - base_rows
    assert num_new in (0, len(new_ids)), (
        f"Unexpected v0 vs base embedding row delta: v0={v0_rows}, base={base_rows}, "
        f"expected delta 0 or {len(new_ids)}"
    )

    v0_keep = torch.ones(v0_rows, dtype=torch.bool)
    v0_keep[torch.tensor(new_ids)] = False
    v0_existing = v0_weight[v0_keep]

    if num_new == 0:
        base_keep = torch.ones(base_rows, dtype=torch.bool)
        base_keep[torch.tensor(new_ids)] = False
        base_existing = base_weight[base_keep]
    else:
        base_existing = base_weight

    identical = torch.equal(v0_existing, base_existing)
    max_abs_delta = (v0_existing - base_existing).abs().max().item() if not identical else 0.0

    del v0_model, base_model

    assert identical, (
        f"Pre-existing embedding rows differ between v0 and base (max abs delta {max_abs_delta}). "
        "Row-scoped backprop or save/load round-trip corrupted the embedding table."
    )
    return {"identical": True, "new_ids": new_ids, "row_delta": num_new, "max_abs_delta": max_abs_delta}


def _run_mmlu_eval(
    v0_dir: Path,
    base_model_id: str,
    mmlu_test_parquet: str,
    eval_out_path: str,
    mode: str,
) -> dict:
    dataset_cls = MMLUSingleTokenResponseDataset if mode == "single_token" else MMLUCoTResponseDataset
    # CoT reasoning traces rarely exceed 1-2k tokens; 8192 just slows the eval
    # to a crawl without improving accuracy signal.
    max_new_tokens = 1 if mode == "single_token" else 2048

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
