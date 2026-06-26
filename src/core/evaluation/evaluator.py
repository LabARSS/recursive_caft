import gc
import hashlib
import json
import os
import pickle
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd
import psutil
import torch
from pydantic import BaseModel
from pydraconf import PydraConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

import core.utils.runtime_trace  # noqa: F401  # install faulthandler/excepthook/signal handlers
from core.datasets.qa_dataset import QADataset
from core.datasets.qa_dataset_adapter import QADatasetAdapter
from core.evaluation.phased_batch_generator import BatchGenerator, _malloc_trim
from core.utils.device import DEVICE_MAP
from core.utils.logger import logger
from core.utils.memory_limit import read_cgroup_mem_limit
from core.utils.subprocess_supervision import supervise_unit

# BatchGenerator.generate() prefills every prompt upfront and stages each KV
# cache on CPU RAM. For 12k+ MMLU-Pro prompts on a 200k-vocab model like
# Phi-4-mini that's ~200GB of staged KV and the container gets OOM-killed.
# Chunking the prompt list bounds peak CPU RAM to one chunk's staging queue.

# A bit more than 1/6 of MMLU test
CHUNK_SIZE = 410


# --- Per-unit crash isolation ------------------------------------------------
# The eval GPU is flaky and dies with a native SIGSEGV (EXIT=139) mid-run (see
# the corrected-AER RxErr storm on 0000:61:00.0). Evaluator.evaluate() runs each
# dataset's _evaluate_single in its own fresh subprocess (_eval_worker.py) and
# restarts it on a nonzero exit — a fresh process per unit gives a full resource
# reset (CUDA context, caching allocator, host RAM) even on clean exit, and
# isolates crashes; resume rides the per-chunk/per-phase checkpoints. The parent
# does no GPU work, so it survives the crashes. Disable with EVAL_SUPERVISE=0.
# The supervision loop / process-group teardown lives in core.utils.subprocess_supervision.
_WORKER_PATH = str(Path(__file__).with_name("_eval_worker.py"))


class GenerationConfig(BaseModel):
    max_new_tokens: int
    max_thinking_tokens: int | None = None
    max_batch_size: int
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    attn_implementation: str | None = "flash_attention_2"
    # Once the cumulative RAM footprint of staged KV cache exceeds this many
    # GB, overflow slots spill to disk instead of RAM.
    kv_cache_offload_threshold_gb: float = 200.0
    # Parent directory for spilled KV files. None → "_kv_spill" under the
    # dataset out dir. Keep this on local NVMe — a network mount is slow.
    kv_cache_spill_dir: str | None = None
    # Stage KV into pinned (page-locked) host memory so CPU<->GPU transfers use
    # the DMA fast path (~8-14x faster restore than pageable). Assumes the host
    # has enough RAM to hold the RAM-resident staged KV page-locked; set False on
    # RAM-constrained hosts (also auto-falls-back to pageable if pinning fails).
    kv_cache_pin_memory: bool = True
    # Outer length-stage boundary (absolute generated-token count). Each stage
    # decodes the pooled survivors only up to the next multiple of this value,
    # then carries still-running sequences (token ids, re-prefilled next stage)
    # so the long tail is batched across chunks instead of decoded once per chunk
    # at a near-empty batch. >= max_new_tokens reproduces single-pass generation.
    # Best as a multiple of BatchGenerator._PHASE_STEP (512) for clean phase grids.
    stage_new_tokens: int = 3072


class EvaluatorConfig(PydraConfig):
    model_path: str
    eval_dataset: QADatasetAdapter | list[QADatasetAdapter]
    out_path: str | None = None
    generation: GenerationConfig


class EvaluationResult(BaseModel):
    accuracy: float
    total: int
    correct: int
    num_truncated: int = 0


class Evaluator:
    def __init__(self, config: EvaluatorConfig, tokenizer: PreTrainedTokenizer | None = None):
        self.config = config
        self.tokenizer = tokenizer

    @property
    def _datasets(self) -> list[QADatasetAdapter]:
        if isinstance(self.config.eval_dataset, list):
            return self.config.eval_dataset
        return [self.config.eval_dataset]

    def evaluate(self) -> list[EvaluationResult]:
        cached_results: list[EvaluationResult | None] = [self._load_cached_result(ds) for ds in self._datasets]
        if all(r is not None for r in cached_results):
            return cached_results  # type: ignore[return-value]

        # Escape hatch / non-GPU debugging: run everything in this process.
        if os.environ.get("EVAL_SUPERVISE") == "0":
            return self._evaluate_in_process(cached_results)

        # Default: one fresh subprocess per uncached dataset (full resource reset
        # even on clean exit), each restarted on crash. Pickle config+tokenizer
        # once; the worker rebuilds the Evaluator, loads the model, and runs
        # _evaluate_single, which writes results.json — recovered here via the cache.
        fd, spec_path = tempfile.mkstemp(prefix="eval_spec_", suffix=".pkl")
        results: list[EvaluationResult] = []
        try:
            with os.fdopen(fd, "wb") as f:
                pickle.dump((self.config, self.tokenizer), f, protocol=pickle.HIGHEST_PROTOCOL)
            for ds_idx, (ds, cached) in enumerate(
                tqdm(zip(self._datasets, cached_results), total=len(self._datasets), desc="Datasets")
            ):
                if cached is not None:
                    results.append(cached)
                    continue
                self._run_unit_in_child(spec_path, ds_idx)
                result = self._load_cached_result(ds)
                if result is None:
                    raise RuntimeError(
                        f"eval worker for dataset {ds_idx} exited 0 but wrote no results at "
                        f"{self._eval_results_path_for(ds)}"
                    )
                results.append(result)
        finally:
            try:
                os.unlink(spec_path)
            except OSError:
                pass
        return results

    def _evaluate_in_process(self, cached_results: list[EvaluationResult | None]) -> list[EvaluationResult]:
        """Run every dataset's _evaluate_single in this process (no subprocesses).

        Used for EVAL_SUPERVISE=0 (local / non-GPU debugging). The crash-isolation
        path is evaluate() spawning one worker per dataset instead.
        """
        model, tokenizer = self._load_model()
        model.eval()
        logger.info(
            f"[cfg] attn_implementation={getattr(model.config, '_attn_implementation', '?')} "
            f"dtype={getattr(model, 'dtype', '?')}"
        )
        results: list[EvaluationResult] = []
        for ds, cached in tqdm(zip(self._datasets, cached_results), total=len(self._datasets), desc="Datasets"):
            if cached is not None:
                results.append(cached)
            else:
                results.append(self._evaluate_single(ds, model, tokenizer))
        return results

    def _run_unit_in_child(self, spec_path: str, ds_idx: int) -> None:
        """Run one _evaluate_single (dataset ds_idx) in a fresh, supervised subprocess.

        Delegates the restart-on-crash / circuit-breaker / interrupt handling to
        supervise_unit; resume rides the chunk/phase checkpoints the worker writes.
        EVAL_SUPERVISE=0 in the child guards against any accidental nested spawn.
        """
        cmd = [sys.executable, _WORKER_PATH, spec_path, str(ds_idx)]
        child_env = {**os.environ, "EVAL_SUPERVISE": "0"}
        supervise_unit(
            cmd,
            child_env,
            label=f"dataset={ds_idx}",
            min_healthy_s=float(os.environ.get("EVAL_MIN_HEALTHY_S", "120")),
            max_fast=int(os.environ.get("EVAL_MAX_FAST_FAILURES", "3")),
            max_attempts=int(os.environ.get("EVAL_MAX_UNIT_ATTEMPTS", "50")),
            mem_watchdog_frac=float(os.environ.get("EVAL_MEM_WATCHDOG_FRAC", "0.92")),
            mem_poll_interval_s=float(os.environ.get("EVAL_MEM_WATCHDOG_INTERVAL_S", "2")),
            max_mem_kills=int(os.environ.get("EVAL_MEM_MAX_KILLS", "3")),
        )

    def _evaluate_single(self, eval_dataset: QADatasetAdapter, model, tokenizer) -> EvaluationResult:
        ds = eval_dataset.process_dataset()

        prompts = [row["input_ids"] for row in ds]
        total = len(prompts)
        logger.info(
            f"Evaluating {total} samples with model from {self.config.model_path} for dataset {eval_dataset.dataset.dataset_id}..."
        )
        mem_limit = read_cgroup_mem_limit()
        rss_gb = psutil.Process().memory_info().rss / 1e9
        logger.info(
            f"Container RAM limit: {f'{mem_limit / 1e9:.0f}GB' if mem_limit is not None else 'none detected (host RAM)'}"
            f" (current RSS {rss_gb:.1f}GB)"
        )

        qa_dataset: QADataset = eval_dataset.dataset

        thinking_end_token_id: int | None = None
        if self.config.generation.max_thinking_tokens is not None:
            resolved = getattr(tokenizer, "thinking_end_token_id", None)
            assert isinstance(resolved, int) and resolved >= 0, (
                "max_thinking_tokens set but tokenizer has no thinking_end_token_id; "
                "call setup_thinking_tokens(tokenizer) in the experiment script."
            )
            thinking_end_token_id = resolved

        chunks_dir = self._chunks_dir_for(eval_dataset)
        chunks_dir.mkdir(parents=True, exist_ok=True)

        stage_size = self.config.generation.stage_new_tokens
        max_new = self.config.generation.max_new_tokens
        if stage_size % BatchGenerator._PHASE_STEP != 0:
            logger.warning(
                f"stage_new_tokens={stage_size} is not a multiple of phase step "
                f"{BatchGenerator._PHASE_STEP}; stage boundaries won't align with the "
                f"phase grid (correct, but mildly less efficient)."
            )

        if total > 0:
            for i, prompt in enumerate(prompts[:3]):
                logger.info(f"Example prompt {i}: {tokenizer.decode(prompt, skip_special_tokens=False)}")

        # --- Outer length-stage loop ---
        # Each stage decodes the pooled survivors only up to the next `stage_size`
        # boundary, then carries still-running sequences (token ids, re-prefilled
        # next stage) so the long tail is batched across chunks rather than decoded
        # once per chunk at a near-empty batch. stage_size >= max_new => a single
        # stage, identical to single-pass generation.
        #
        # Resume is driven entirely by the per-(stage, sub_chunk) result files: a
        # restarted worker replays this loop from stage 0, cache-hits completed
        # sub_chunks (re-deriving the survivor flow deterministically under greedy
        # decoding), and runs the first uncomputed sub_chunk. Per-phase crash
        # granularity inside a sub_chunk rides the generator's own checkpoint.
        all_results: list[dict] = []
        correct = 0
        num_truncated = 0

        pending: list[dict] = [
            {
                "row_id": ds[i]["row_id"],
                "ds_index": i,
                "prompt_ids": prompts[i],
                "prior_generated": [],
                "thinking_budget_exhausted": False,
            }
            for i in range(total)
        ]

        stage = 0
        while pending:
            carry_len = min((stage + 1) * stage_size, max_new)
            # Final stage (carry_len == max_new): pass None so sequences truncate at
            # the hard budget instead of carrying (there is no next stage to pool to).
            carry_arg = None if carry_len >= max_new else carry_len
            stage_dir = chunks_dir / f"stage_{stage:02d}"
            stage_dir.mkdir(parents=True, exist_ok=True)

            num_sub = (len(pending) + CHUNK_SIZE - 1) // CHUNK_SIZE
            logger.info(
                f"Stage {stage + 1}: {len(pending)} sequences, carry_len={carry_len} "
                f"({'final' if carry_arg is None else 'pooling'}), {num_sub} sub-chunk(s)"
            )

            survivors: list[dict] = []
            for sub_idx in range(num_sub):
                sub = pending[sub_idx * CHUNK_SIZE : (sub_idx + 1) * CHUNK_SIZE]
                result_path = stage_dir / f"subchunk_{sub_idx:04d}.result.json"

                cached = self._load_subchunk_result(result_path)
                if cached is not None:
                    finished_rows, sub_survivors = cached
                    logger.info(
                        f"Stage {stage + 1} sub-chunk {sub_idx + 1}/{num_sub}: loaded "
                        f"{len(finished_rows)} finished + {len(sub_survivors)} survivors from cache"
                    )
                else:
                    logger.info(
                        f"Stage {stage + 1} sub-chunk {sub_idx + 1}/{num_sub}: {len(sub)} sequences"
                    )
                    finished_rows, sub_survivors = self._run_subchunk(
                        sub=sub,
                        stage_dir=stage_dir,
                        sub_idx=sub_idx,
                        carry_at_new_tokens=carry_arg,
                        ds=ds,
                        qa_dataset=qa_dataset,
                        model=model,
                        tokenizer=tokenizer,
                        thinking_end_token_id=thinking_end_token_id,
                        eval_dataset=eval_dataset,
                    )

                all_results.extend(finished_rows)
                correct += sum(1 for r in finished_rows if r["is_correct"])
                num_truncated += sum(1 for r in finished_rows if r["is_truncated"])
                survivors.extend(sub_survivors)

            pending = survivors
            stage += 1

        if num_truncated > 0:
            pct = num_truncated / total * 100
            logger.warning(
                f"Generation reached max_new_tokens ({self.config.generation.max_new_tokens}) "
                f"for {num_truncated}/{total} sequences ({pct:.1f}%)"
            )

        # Restore original dataset order, then drop the internal ds_index helper key
        # so responses.parquet keeps its original schema.
        all_results.sort(key=lambda r: r["ds_index"])
        for r in all_results:
            r.pop("ds_index", None)

        accuracy = correct / total if total > 0 else 0.0
        result = EvaluationResult(accuracy=accuracy, total=total, correct=correct, num_truncated=num_truncated)

        logger.info(f"Evaluation complete: accuracy={accuracy:.4f} ({correct}/{total})")

        self._save_results(eval_dataset, result, all_results)
        self._cleanup_chunks(eval_dataset)

        return result

    def _run_subchunk(
        self,
        sub: list[dict],
        stage_dir: Path,
        sub_idx: int,
        carry_at_new_tokens: int | None,
        ds,
        qa_dataset: QADataset,
        model,
        tokenizer,
        thinking_end_token_id: int | None,
        eval_dataset: QADatasetAdapter,
    ) -> tuple[list[dict], list[dict]]:
        """Generate one sub-chunk up to the stage's carry boundary.

        Returns (finished_rows, survivors): finished rows are decoded + verified;
        survivors carry their token ids (prompt + generated-so-far) to the next
        length-stage. The (finished, survivors) split is persisted atomically so a
        restarted worker can cache-hit this sub_chunk instead of re-decoding it.
        """
        result_path = stage_dir / f"subchunk_{sub_idx:04d}.result.json"
        ckpt_path = stage_dir / f"subchunk_{sub_idx:04d}.ckpt.json"

        spill_parent = self.config.generation.kv_cache_spill_dir or str(self._out_path_for(eval_dataset) / "_kv_spill")
        generator = BatchGenerator(
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=self.config.generation.max_new_tokens,
            max_batch_size=self.config.generation.max_batch_size,
            temperature=self.config.generation.temperature,
            top_p=self.config.generation.top_p,
            top_k=self.config.generation.top_k,
            max_thinking_tokens=self.config.generation.max_thinking_tokens,
            thinking_end_token_id=thinking_end_token_id,
            kv_cache_offload_threshold_gb=self.config.generation.kv_cache_offload_threshold_gb,
            kv_cache_spill_dir=spill_parent,
            kv_cache_pin_memory=self.config.generation.kv_cache_pin_memory,
        )

        gen_result = generator.generate(
            [r["prompt_ids"] for r in sub],
            checkpoint_path=str(ckpt_path),
            carry_at_new_tokens=carry_at_new_tokens,
            resume_generated=[(r["prior_generated"] or None) for r in sub],
            resume_thinking_exhausted=[r["thinking_budget_exhausted"] for r in sub],
        )

        finished_rows: list[dict] = []
        sub_survivors: list[dict] = []
        for offset, gen_ids in enumerate(gen_result.sequences):
            src = sub[offset]
            if gen_result.unfinished[offset]:
                sub_survivors.append(
                    {
                        "row_id": src["row_id"],
                        "ds_index": src["ds_index"],
                        "prompt_ids": src["prompt_ids"],
                        "prior_generated": gen_ids,
                        "thinking_budget_exhausted": bool(gen_result.thinking_budget_exhausted[offset]),
                    }
                )
                continue

            ds_row = ds[src["ds_index"]]
            response = tokenizer.decode(gen_ids, skip_special_tokens=False).strip()
            try:
                parsed_answer, is_correct = qa_dataset.verify_assistant_response(ds_row, response)
            except Exception as ex:
                logger.warning(f"Error verifying row {ds_row['row_id']}: {ex}")
                parsed_answer = response
                is_correct = False

            finished_rows.append(
                {
                    "row_id": ds_row["row_id"],
                    "ds_index": src["ds_index"],
                    "response": response,
                    "parsed_answer": parsed_answer,
                    "is_correct": is_correct,
                    "is_truncated": bool(gen_result.truncated[offset]),
                    "is_thinking_budget_exhausted": bool(gen_result.thinking_budget_exhausted[offset]),
                }
            )

        self._save_subchunk_result(result_path, finished_rows, sub_survivors)
        # Sub-chunk is durably saved; its per-phase resume checkpoint is no longer needed.
        ckpt_path.unlink(missing_ok=True)

        # Release this sub-chunk's CPU-staged KV before the next prefills its own.
        del generator, gen_result
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # gc.collect() drops Python refs; the chunks then sit in glibc's per-arena
        # freelists and never return to the OS. stage_row_to_cpu produces many small
        # allocs per slot, so arenas fragment heavily over a run.
        _malloc_trim()
        rss_gb = psutil.Process().memory_info().rss / 1e9
        logger.info(f"[mem] post-subchunk baseline: rss={rss_gb:.2f}GB")

        return finished_rows, sub_survivors

    def _config_hash(self) -> str:
        gen = self.config.generation
        raw = "|".join(
            str(x)
            for x in (
                self.config.model_path,
                gen.max_new_tokens,
                gen.max_thinking_tokens,
                gen.temperature,
                gen.top_p,
                gen.top_k,
                CHUNK_SIZE,
                gen.stage_new_tokens,
            )
        )
        return hashlib.md5(raw.encode()).hexdigest()[:12]

    def _chunks_dir_for(self, eval_dataset: QADatasetAdapter) -> Path:
        return self._out_path_for(eval_dataset) / f"_chunks_{self._config_hash()}"

    @staticmethod
    def _load_subchunk_result(path: Path) -> tuple[list[dict], list[dict]] | None:
        """Load a cached (finished_rows, survivors) sub-chunk result, or None.

        A corrupt/partial file is treated as absent (the sub-chunk reruns), mirroring
        _load_chunk's old tolerance. Under greedy decoding a rerun reproduces the same
        finished/survivor split, so later stages' caches stay consistent.
        """
        if not path.exists():
            return None
        try:
            with open(path) as f:
                data = json.load(f)
            return data["finished"], data["survivors"]
        except (json.JSONDecodeError, KeyError, TypeError) as ex:
            logger.warning(f"Corrupt sub-chunk result at {path} ({ex}); will regenerate")
            return None

    def _save_subchunk_result(self, path: Path, finished_rows: list[dict], survivors: list[dict]) -> None:
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump({"finished": finished_rows, "survivors": survivors}, f)
        os.replace(tmp, path)

    def _cleanup_chunks(self, eval_dataset: QADatasetAdapter) -> None:
        out_dir = self._out_path_for(eval_dataset)
        if not out_dir.exists():
            return
        for entry in out_dir.iterdir():
            # _kv_spill holds per-chunk KV spill subdirs (each store removes its
            # own; the empty parent is swept here).
            if entry.is_dir() and (entry.name.startswith("_chunks_") or entry.name == "_kv_spill"):
                shutil.rmtree(entry, ignore_errors=True)

    def _load_cached_result(self, eval_dataset: QADatasetAdapter) -> EvaluationResult | None:
        results_path = self._eval_results_path_for(eval_dataset)
        if not results_path.exists():
            return None
        with open(results_path) as f:
            data = json.load(f)
        logger.info(f"Found cached results at {results_path}, skipping evaluation")
        return EvaluationResult(
            accuracy=data["accuracy"],
            total=data["total"],
            correct=data["correct"],
            num_truncated=data.get("num_truncated", 0),
        )

    def _out_path_for(self, eval_dataset: QADatasetAdapter) -> Path:
        dataset_id = eval_dataset.dataset.dataset_id
        if self.config.out_path:
            return Path(self.config.out_path) / dataset_id

        model_path = Path(self.config.model_path)
        if not model_path.is_dir():
            raise ValueError(f"out_path must be set when model_path is not a local directory: {self.config.model_path}")
        return model_path / "evals" / dataset_id

    def _eval_results_path_for(self, eval_dataset: QADatasetAdapter) -> Path:
        return self._out_path_for(eval_dataset) / "results.json"

    def _load_model(self):
        model_path = Path(self.config.model_path)

        if model_path.is_dir():
            adapter_config = model_path / "adapter_config.json"
            if adapter_config.exists():
                return self._load_lora_model(model_path, adapter_config)

        logger.info(f"Loading model from {self.config.model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            device_map=DEVICE_MAP,
            torch_dtype=torch.bfloat16,
            attn_implementation=self.config.generation.attn_implementation,
        )
        if not self.tokenizer:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
        return model, self.tokenizer

    def _load_lora_model(self, model_path: Path, adapter_config: Path):
        from peft import PeftModel

        from core.training.callbacks.save_thinking_token_rows import ROWS_FILENAME

        with open(adapter_config) as f:
            config = json.load(f)

        base_model_id = config.get("base_model_name_or_path")
        if not base_model_id:
            raise ValueError(f"adapter_config.json at {adapter_config} missing 'base_model_name_or_path'")

        logger.info(f"Loading LoRA model: base={base_model_id}, adapter={model_path}")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            device_map=DEVICE_MAP,
            torch_dtype=torch.bfloat16,
            attn_implementation=self.config.generation.attn_implementation,
        )

        rows_path = model_path / ROWS_FILENAME
        if rows_path.exists():
            payload = torch.load(rows_path, map_location="cpu", weights_only=True)
            new_ids = payload["new_ids"]
            in_rows = payload["input_rows"]
            in_w = base_model.get_input_embeddings().weight
            with torch.no_grad():
                ids_t = torch.tensor(new_ids, dtype=torch.long, device=in_w.device)
                in_w.data[ids_t] = in_rows.to(dtype=in_w.dtype, device=in_w.device)
                if "output_rows" in payload:
                    out_layer = base_model.get_output_embeddings()
                    assert out_layer is not None, (
                        f"{rows_path} has output_rows but base model has no output embedding layer"
                    )
                    out_w = out_layer.weight
                    out_w.data[ids_t] = payload["output_rows"].to(dtype=out_w.dtype, device=out_w.device)
            logger.info(f"Loaded {len(new_ids)} thinking-token rows from {rows_path}")

        model = PeftModel.from_pretrained(base_model, str(model_path))
        if not self.tokenizer:
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        return model, self.tokenizer

    def _save_results(self, eval_dataset: QADatasetAdapter, result: EvaluationResult, all_results: list[dict]) -> None:
        out_path = self._out_path_for(eval_dataset)
        out_path.mkdir(parents=True, exist_ok=True)

        # Save summary
        summary_path = self._eval_results_path_for(eval_dataset)
        summary = {
            "accuracy": result.accuracy,
            "total": result.total,
            "correct": result.correct,
            "num_truncated": result.num_truncated,
            "model_path": self.config.model_path,
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Summary saved to {summary_path}")

        # Save per-row results
        results_path = out_path / "responses.parquet"
        pd.DataFrame(all_results).to_parquet(results_path, index=False)
        logger.info(f"Per-row results saved to {results_path}")
