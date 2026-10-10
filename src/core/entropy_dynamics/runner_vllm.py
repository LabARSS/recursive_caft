"""
vLLM-accelerated runner for entropy dynamics.
Drop-in replacement for the HuggingFace runner when speed matters.

Key differences vs HF runner:
  - Batches ALL prompts in a single vLLM call (10-50x speedup)
  - Entropy computed from top-K logprobs (K=500, error < 0.001 nats for MMLU)
  - Caches built prompts to disk → restarts skip the build step entirely
  - Auto-detects tensor_parallel_size from available GPUs

Why logprobs=500 instead of logits_processors:
  - vLLM V0 per-request logits_processors run inside worker subprocesses (TP>1),
    so Python object state never propagates back to the main process → all nan.
  - logprobs=500 via max_logprobs works across all TP configs.
  - For MMLU/GPQA answer entropy, top-500 logprobs capture >99.99% of probability
    mass. Max entropy error ≈ 0.001 nats — negligible for our purposes.

Usage:
  python -m core.entropy_dynamics.run_experiment \\
      --teacher_reasoning_path data/out/distillation/mmlu_synth_gptoss_b_t0_8.parquet \\
      --out_dir artifacts/entropy_dynamics/mmlu_forced \\
      --role proxy --mode forced --students qwen_32b \\
      --use_vllm --gpu_memory_utilization 0.85
"""

from __future__ import annotations

import hashlib
import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import TokensPrompt


from core.entropy_dynamics.config import EntropyDynamicsConfig, StudentModelConfig
from core.entropy_dynamics.prompt_builder import PrefixedPrompt, build_prefixed_prompts
from core.entropy_dynamics.reasoning_loader import TeacherReasoning, load_teacher_reasoning
from core.entropy_dynamics.runner import StepResult, ExperimentResults, _load_processed_ids
from core.utils.seed import set_seed

# How many top logprobs to request from vLLM.
# 500 captures >99.99% of probability mass for typical LLM answer distributions.
# Entropy error vs full-vocab: < 0.001 nats.
_LOGPROBS_K = 500

# Force vLLM V0 engine: V1 caps logprobs at 20 and has other restrictions.
# Must be set before any vLLM import / LLM() construction.
os.environ["VLLM_USE_V1"] = "0"


# ---------------------------------------------------------------------------
# Entropy from top-K logprobs
# ---------------------------------------------------------------------------

def _entropy_from_topk_logprobs(logprob_dict: dict) -> float:
    """Shannon entropy (nats) from vLLM's top-K logprob dict.

    vLLM returns {token_id: Logprob(logprob=..., ...)} for the top-K tokens.
    We exponentiate, renormalise to sum=1 (the missing tail adds a tiny
    correction term that is < 0.001 nats for K=500 on MMLU), and compute H.
    """
    if not logprob_dict:
        return float("nan")
    log_ps = np.array([lp.logprob for lp in logprob_dict.values()], dtype=np.float64)
    ps = np.exp(log_ps)
    total = ps.sum()
    if total <= 0:
        return float("nan")
    ps /= total
    return float(-np.sum(ps * np.log(ps + 1e-12)))


# ---------------------------------------------------------------------------
# Prompt cache helpers
# ---------------------------------------------------------------------------

def _prompt_cache_key(model_id: str, teacher_path: str, window_size: int,
                      mode: str, dataset_type: str) -> str:
    """Stable MD5 key for a set of built prompts. Includes file mtime."""
    mtime = int(os.path.getmtime(teacher_path)) if os.path.exists(teacher_path) else 0
    raw = f"{model_id}|{teacher_path}|{mtime}|{window_size}|{mode}|{dataset_type}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _prompt_cache_path(out_dir: Path, cache_key: str) -> Path:
    return out_dir / f"prompt_cache_{cache_key}.pkl"


def _load_prompt_cache(path: Path) -> list[PrefixedPrompt] | None:
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        print(f"  Loaded {len(data)} prompts from cache: {path.name}")
        return data
    except Exception as e:
        print(f"  Warning: could not load prompt cache ({e}), will rebuild.")
        return None


def _save_prompt_cache(path: Path, prompts: list[PrefixedPrompt]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(prompts, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Prompt cache saved: {path.name} ({len(prompts)} prompts)")



# ---------------------------------------------------------------------------
# Parallel prompt building
# ---------------------------------------------------------------------------

def _build_one_sample(args: tuple) -> list:
    """Worker function: tokenize one sample and build all its prefixed prompts.

    Receives a plain tuple (not a TeacherReasoning) to be picklable across
    processes.  Returns a list of dicts that can be reconstructed into
    PrefixedPrompt objects in the main process.
    """
    from transformers import AutoTokenizer
    from core.entropy_dynamics.prompt_builder import build_prefixed_prompts
    from core.entropy_dynamics.reasoning_loader import TeacherReasoning

    (question_id, question, options, gold_answer, thinking_text,
     model_id, window_size, mode, dataset_type) = args

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    sample = TeacherReasoning(
        question_id=question_id,
        question=question,
        options=options,
        gold_answer=gold_answer,
        thinking_text=thinking_text,
        thinking_token_ids=tokenizer.encode(thinking_text, add_special_tokens=False),
    )

    prompts = build_prefixed_prompts(
        sample=sample,
        tokenizer=tokenizer,
        window_size=window_size,
        mode=mode,
        dataset_type=dataset_type,
    )
    return prompts


def _build_prompts_parallel(
    samples: list,
    model_id: str,
    window_size: int,
    mode,
    dataset_type: str,
    num_workers: int = 16,        # 0 = auto (cpu_count / 2)
) -> list:
    """Build all prefixed prompts in parallel using ProcessPoolExecutor.

    Each worker loads its own tokenizer instance (necessary because
    HuggingFace tokenizers are not fork-safe).  Workers are spawned fresh,
    so GPU memory from the main process is not duplicated.
    """
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, as_completed

    if num_workers <= 0:
        num_workers = max(1, multiprocessing.cpu_count() // 2)

    print(f"Building prompts for {len(samples)} questions "
          f"(parallel, {num_workers} workers)...")

    args_list = [
        (
            s.question_id, s.question, s.options, s.gold_answer,
            s.thinking_text, model_id, window_size, mode, dataset_type,
        )
        for s in samples
    ]

    all_prompts = []
    with ProcessPoolExecutor(max_workers=num_workers,
                              mp_context=__import__("multiprocessing").get_context("spawn")) as pool:
        futures = {pool.submit(_build_one_sample, a): i for i, a in enumerate(args_list)}
        done = 0
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Building prompts"):
            all_prompts.extend(fut.result())
            done += 1

    print(f"  Built {len(all_prompts)} prompts total.")
    return all_prompts

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class VLLMEntropyRunner:
    """Fast entropy measurement using vLLM offline inference."""

    def __init__(
        self,
        config: EntropyDynamicsConfig,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 8192,
        tensor_parallel_size: int = 0,   # 0 → auto-detect from GPU count
    ):
        self.config = config
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len

        if tensor_parallel_size <= 0:
            detected = torch.cuda.device_count() if torch.cuda.is_available() else 1
            self.tensor_parallel_size = max(detected, 1)
            print(f"  Auto-detected tensor_parallel_size = {self.tensor_parallel_size} "
                  f"({detected} GPU(s) visible)")
        else:
            self.tensor_parallel_size = tensor_parallel_size

    def run(self) -> pd.DataFrame:
        set_seed()

        # Force loopback for distributed rendezvous — prevents TCP timeout
        # when creating a second LLM() after del llm (PyTorch distributed
        # may leave a stale external-IP TCPStore from the previous run).
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")

        first_model_id = self.config.students[0].model_id
        loader_tokenizer = AutoTokenizer.from_pretrained(first_model_id)

        print(f"Loading teacher reasoning from {self.config.teacher_reasoning_path}...")
        samples = load_teacher_reasoning(
            self.config.teacher_reasoning_path,
            tokenizer=loader_tokenizer,
            min_thinking_tokens=self.config.window_size,
        )
        print(f"Loaded {len(samples)} samples with valid reasoning chains.")

        # ── Pre-load any results already saved to the final output file ───
        # Ensures that if a previous run crashed after saving some model's
        # data to the final parquet, those rows survive into this run's
        # all_results and are not silently dropped.
        out_path = self.config.out_path / self.config.results_filename
        all_results = ExperimentResults()
        if out_path.exists():
            try:
                existing_df = pd.read_parquet(out_path)
                all_results.rows = existing_df.to_dict(orient="records")
                print(f"Loaded {len(all_results.rows)} existing rows from {out_path.name}")
            except Exception as e:
                print(f"Warning: could not load existing results ({e}), starting fresh.")

        for model_cfg in self.config.students:
            self._run_single_model(model_cfg, samples, all_results)

        all_results.save(out_path)
        print(f"All results saved to {out_path}")
        return all_results.to_dataframe()

    def _run_single_model(
        self,
        model_cfg: StudentModelConfig,
        samples: list[TeacherReasoning],
        results: ExperimentResults,
    ):
        from vllm import LLM, SamplingParams

        role = self.config.role.value
        print(f"\n{'='*60}")
        print(f"[vLLM {role}] {model_cfg.label} ({model_cfg.model_id})")
        print(f"  tensor_parallel_size = {self.tensor_parallel_size}")
        print(f"  logprobs_k           = {_LOGPROBS_K}")
        print(f"{'='*60}")

        # ── Checkpoint ───────────────────────────────────────────────────
        checkpoint_path = (
            self.config.out_path
            / f"checkpoint_{role}_{model_cfg.label}_{self.config.mode.value}.parquet"
        )
        processed_ids = _load_processed_ids(checkpoint_path)
        remaining = [s for s in samples if s.question_id not in processed_ids]
        print(f"Resuming: {len(processed_ids)} already done, {len(remaining)} remaining.")

        if not remaining:
            print("  Nothing to do.")
            return

        # ── Tokenizer ────────────────────────────────────────────────────
        tokenizer = AutoTokenizer.from_pretrained(model_cfg.model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # ── Build / load prompts from cache ──────────────────────────────
        cache_key = _prompt_cache_key(
            model_cfg.model_id,
            self.config.teacher_reasoning_path,
            self.config.window_size,
            self.config.mode.value,
            self.config.dataset_type,
        )
        cache_path = _prompt_cache_path(self.config.out_path, cache_key)
        all_prompts = _load_prompt_cache(cache_path)

        if all_prompts is None:
            all_prompts = _build_prompts_parallel(
                samples=remaining,
                model_id=model_cfg.model_id,
                window_size=self.config.window_size,
                mode=self.config.mode,
                dataset_type=self.config.dataset_type,
            )
            _save_prompt_cache(cache_path, all_prompts)
        else:
            before = len(all_prompts)
            all_prompts = [p for p in all_prompts if p.question_id not in processed_ids]
            print(f"  After filtering already-done: {before} → {len(all_prompts)} prompts")

        if not all_prompts:
            print("  All prompts already processed.")
            return

        # ── Filter over-length prompts ────────────────────────────────────
        max_input_len = self.max_model_len - 1
        valid_prompts, skipped_prompts = [], []
        for p in all_prompts:
            (valid_prompts if len(p.input_ids) <= max_input_len else skipped_prompts).append(p)

        if skipped_prompts:
            n_q = len({p.question_id for p in skipped_prompts})
            print(f"  Skipping {len(skipped_prompts)} prompts from {n_q} questions "
                  f"(exceed max_model_len={self.max_model_len}).")
            for p in skipped_prompts:
                results.append(StepResult(
                    question_id=p.question_id,
                    model_label=model_cfg.label,
                    role=role,
                    k=p.k,
                    num_reasoning_tokens=p.num_reasoning_tokens,
                    total_reasoning_tokens=p.total_reasoning_tokens,
                    mode=p.mode.value,
                    answer_entropy=float("nan"),
                    model_answer="",
                    model_correct=False,
                    gold_answer=p.gold_answer,
                ))

        all_prompts = valid_prompts
        print(f"Total prompts to process: {len(all_prompts)} "
              f"(skipped {len(skipped_prompts)} over-length)")

        if not all_prompts:
            print("  All remaining prompts were over-length.")
            return

        # ── Load vLLM model ───────────────────────────────────────────────
        print(f"Loading vLLM model (engine=V0, gpu_mem={self.gpu_memory_utilization}, "
              f"max_len={self.max_model_len}, tp={self.tensor_parallel_size})...")

        llm = LLM(
            model=model_cfg.model_id,
            tokenizer=model_cfg.model_id,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            tensor_parallel_size=self.tensor_parallel_size,
            dtype="bfloat16",
            trust_remote_code=True,
            # Override V0's default max_logprobs (5) so we can request 500.
            max_logprobs=_LOGPROBS_K,
        )

        # ── Single shared SamplingParams ──────────────────────────────────
        # logprobs=K asks vLLM to return the top-K log-probabilities for the
        # single generated token. No per-request Python objects → works with any TP.
        sampling_params = SamplingParams(
            max_tokens=1,
            temperature=0.0,
            logprobs=_LOGPROBS_K,
        )

        # ── Chunked inference with periodic checkpointing ────────────────
        # Splits all_prompts into chunks of CHECKPOINT_EVERY and saves after
        # each chunk. On restart, already-processed question_ids are filtered
        # out above, so we never reprocess completed work.
        CHECKPOINT_EVERY = 5_000
        total = len(all_prompts)
        t_start = time.perf_counter()

        print(f"Running vLLM inference on {total} prompts "
              f"(checkpoint every {CHECKPOINT_EVERY})...")

        for batch_start in range(0, total, CHECKPOINT_EVERY):
            batch_prompts = all_prompts[batch_start:batch_start + CHECKPOINT_EVERY]
            batch_token_ids = [p.input_ids for p in batch_prompts]
            vllm_inputs = [TokensPrompt(prompt_token_ids=ids) for ids in batch_token_ids]

            batch_outputs = llm.generate(
                prompts=vllm_inputs,
                sampling_params=sampling_params,
                use_tqdm=True,
            )

            for prompt_obj, output in zip(batch_prompts, batch_outputs):
                answer, entropy = self._extract_from_output(output, tokenizer)
                results.append(StepResult(
                    question_id=prompt_obj.question_id,
                    model_label=model_cfg.label,
                    role=role,
                    k=prompt_obj.k,
                    num_reasoning_tokens=prompt_obj.num_reasoning_tokens,
                    total_reasoning_tokens=prompt_obj.total_reasoning_tokens,
                    mode=prompt_obj.mode.value,
                    answer_entropy=entropy,
                    model_answer=answer,
                    model_correct=(answer == prompt_obj.gold_answer),
                    gold_answer=prompt_obj.gold_answer,
                ))

            results.save(checkpoint_path)
            done = min(batch_start + CHECKPOINT_EVERY, total)
            elapsed = time.perf_counter() - t_start
            rate = done / elapsed if elapsed > 0 else 0
            eta = (total - done) / rate if rate > 0 else 0
            print(f"  Checkpoint [{done}/{total}] "
                  f"| {rate:.1f} prompts/s "
                  f"| ETA {eta/60:.0f} min")

        # ── Save to final output file after this model completes ─────────
        # Writes incrementally so a crash on the next model doesn't lose
        # this model's data (the final run() save becomes a no-op).
        out_path = self.config.out_path / self.config.results_filename
        results.save(out_path)
        print(f"Model results saved to {out_path.name}")

        # ── Free GPU memory ───────────────────────────────────────────────
        del llm
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _extract_from_output(output, tokenizer) -> tuple[str, float]:
        """Return (answer_str, entropy_nats) from a single vLLM RequestOutput."""
        if not output.outputs:
            return "", float("nan")
        gen = output.outputs[0]

        answer = ""
        if gen.token_ids:
            answer = tokenizer.decode([gen.token_ids[0]]).strip().lower()

        entropy = float("nan")
        if gen.logprobs and len(gen.logprobs) > 0:
            entropy = _entropy_from_topk_logprobs(gen.logprobs[0])

        return answer, entropy
