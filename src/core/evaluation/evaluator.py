import gc
import hashlib
import json
import os
import shutil
from pathlib import Path

import pandas as pd
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

# BatchGenerator.generate() prefills every prompt upfront and stages each KV
# cache on CPU RAM. For 12k+ MMLU-Pro prompts on a 200k-vocab model like
# Phi-4-mini that's ~200GB of staged KV and the container gets OOM-killed.
# Chunking the prompt list bounds peak CPU RAM to one chunk's staging queue.
CHUNK_SIZE = 256


class GenerationConfig(BaseModel):
    max_new_tokens: int
    max_thinking_tokens: int | None = None
    max_batch_size: int
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    torch_compile: bool = True
    attn_implementation: str | None = "flash_attention_2"
    # Once the cumulative RAM footprint of staged KV cache exceeds this many
    # GB, overflow slots spill to disk instead of RAM.
    kv_cache_offload_threshold_gb: float = 120.0
    # Parent directory for spilled KV files. None → "_kv_spill" under the
    # dataset out dir. Keep this on local NVMe — a network mount is slow.
    kv_cache_spill_dir: str | None = None


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

        model, tokenizer = self._load_model()
        model.eval()

        if self.config.generation.torch_compile:
            if not torch.cuda.is_available():
                logger.warning("torch_compile=True but CUDA not available — skipping compilation.")
            else:
                # Inductor's default cache lives at /tmp/torchinductor_<user>. In
                # Docker, /tmp is usually tmpfs (RAM-backed) with a small size cap
                # set at container start — a few GB at most. After several
                # compile-heavy runs the cache fills the tmpfs and the next write
                # fails with `OSError: [Errno 28] No space left on device`.
                # Redirect to artifacts/_inductor_cache under the repo so the
                # cache lives next to other run outputs and is trivial to inspect
                # or wipe. Skipped if the user already set TORCHINDUCTOR_CACHE_DIR.
                if "TORCHINDUCTOR_CACHE_DIR" not in os.environ:
                    _repo_root = Path(__file__).resolve().parents[3]
                    _cache_dir = str(_repo_root / "artifacts" / "_inductor_cache")
                    os.makedirs(_cache_dir, exist_ok=True)
                    os.environ["TORCHINDUCTOR_CACHE_DIR"] = _cache_dir
                    logger.info(f"[trace] TORCHINDUCTOR_CACHE_DIR={_cache_dir}")

                logger.info("Compiling model with torch.compile... First forward call will be slow.")
                torch.set_float32_matmul_precision("high")
                torch._dynamo.config.cache_size_limit = 2048
                torch._dynamo.config.suppress_errors = False
                torch._dynamo.config.verbose = False
                # Default inductor worker pool is min(32, cpu_count). Each worker is a
                # forked Python with torch loaded; under COW divergence the pool can
                # consume tens of GB. Cap to 4 — recompiles here are infrequent and
                # the parallelism wasn't buying much.
                import torch._inductor.config as _inductor_config

                _inductor_config.compile_threads = 4
                logger.info(
                    f"[trace] torch.compile config: cache_size_limit={torch._dynamo.config.cache_size_limit} "
                    f"compile_threads={_inductor_config.compile_threads}"
                )
                model = torch.compile(model)

        results: list[EvaluationResult] = []
        for ds, cached in tqdm(zip(self._datasets, cached_results), total=len(self._datasets), desc="Datasets"):
            if cached is not None:
                results.append(cached)
            else:
                results.append(self._evaluate_single(ds, model, tokenizer))

        return results

    def _evaluate_single(self, eval_dataset: QADatasetAdapter, model, tokenizer) -> EvaluationResult:
        ds = eval_dataset.process_dataset()

        prompts = [row["input_ids"] for row in ds]
        total = len(prompts)
        logger.info(
            f"Evaluating {total} samples with model from {self.config.model_path} for dataset {eval_dataset.dataset.dataset_id}..."
        )

        qa_dataset: QADataset = eval_dataset.dataset

        correct = 0
        num_truncated = 0
        all_results: list[dict] = []

        thinking_end_token_id: int | None = None
        if self.config.generation.max_thinking_tokens is not None:
            resolved = getattr(tokenizer, "thinking_end_token_id", None)
            assert isinstance(resolved, int) and resolved >= 0, (
                "max_thinking_tokens set but tokenizer has no thinking_end_token_id; "
                "call setup_thinking_tokens(tokenizer) in the experiment script."
            )
            thinking_end_token_id = resolved

        num_chunks = (total + CHUNK_SIZE - 1) // CHUNK_SIZE
        chunks_dir = self._chunks_dir_for(eval_dataset)
        chunks_dir.mkdir(parents=True, exist_ok=True)

        for chunk_idx in range(num_chunks):
            start = chunk_idx * CHUNK_SIZE
            end = min(start + CHUNK_SIZE, total)
            chunk_path = self._chunk_path(chunks_dir, chunk_idx)

            cached_rows = self._load_chunk(chunk_path)
            if cached_rows is not None:
                logger.info(f"Chunk {chunk_idx + 1}/{num_chunks}: loaded {len(cached_rows)} rows from {chunk_path}")
                all_results.extend(cached_rows)
                correct += sum(1 for r in cached_rows if r["is_correct"])
                num_truncated += sum(1 for r in cached_rows if r["is_truncated"])
                continue

            chunk_prompts = prompts[start:end]
            logger.info(f"Chunk {chunk_idx + 1}/{num_chunks}: prompts [{start}:{end}] ({len(chunk_prompts)} samples)")

            if chunk_idx == 0:
                for i, prompt in enumerate(chunk_prompts[:3]):
                    logger.info(f"Example prompt {i}: {tokenizer.decode(prompt, skip_special_tokens=False)}")

            spill_parent = self.config.generation.kv_cache_spill_dir or str(
                self._out_path_for(eval_dataset) / "_kv_spill"
            )
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
            )

            gen_result = generator.generate(chunk_prompts)

            chunk_rows: list[dict] = []
            for offset, gen_ids in enumerate(gen_result.sequences):
                row = ds[start + offset]
                response = tokenizer.decode(gen_ids, skip_special_tokens=False).strip()
                is_truncated = gen_result.truncated[offset]
                is_thinking_budget_exhausted = gen_result.thinking_budget_exhausted[offset]

                try:
                    parsed_answer, is_correct = qa_dataset.verify_assistant_response(row, response)
                except Exception as ex:
                    logger.warning(f"Error verifying row {row['row_id']}: {ex}")
                    parsed_answer = response
                    is_correct = False

                chunk_rows.append(
                    {
                        "row_id": row["row_id"],
                        "response": response,
                        "parsed_answer": parsed_answer,
                        "is_correct": is_correct,
                        "is_truncated": is_truncated,
                        "is_thinking_budget_exhausted": is_thinking_budget_exhausted,
                    }
                )

            self._save_chunk_atomic(chunk_path, chunk_rows)
            all_results.extend(chunk_rows)
            correct += sum(1 for r in chunk_rows if r["is_correct"])
            num_truncated += gen_result.num_truncated

            # Release the chunk's CPU-staged KV caches before the next chunk
            # prefills its own. Without this, peak CPU RAM keeps climbing.
            del generator, gen_result
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # gc.collect() drops Python refs; the chunks then sit in glibc's
            # per-arena freelists and never return to the OS. The .cpu() in
            # stage_row_to_cpu produces 2*num_layers small allocs per slot, so
            # arenas fragment heavily over a run.
            _malloc_trim()
            # Wipe the dynamo compile cache between chunks. Cached entries from
            # earlier chunks see different (bs, seq_width, max_cache_len) tuples
            # and count toward cache_size_limit without serving the next chunk.
            # Without this, by ~chunk 7 the cache thrashes and decode collapses
            # into a recompile-evict-recompile loop.
            torch._dynamo.reset()

        if num_truncated > 0:
            pct = num_truncated / total * 100
            logger.warning(
                f"Generation reached max_new_tokens ({self.config.generation.max_new_tokens}) "
                f"for {num_truncated}/{total} sequences ({pct:.1f}%)"
            )

        accuracy = correct / total if total > 0 else 0.0
        result = EvaluationResult(accuracy=accuracy, total=total, correct=correct, num_truncated=num_truncated)

        logger.info(f"Evaluation complete: accuracy={accuracy:.4f} ({correct}/{total})")

        self._save_results(eval_dataset, result, all_results)
        self._cleanup_chunks(eval_dataset)

        return result

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
            )
        )
        return hashlib.md5(raw.encode()).hexdigest()[:12]

    def _chunks_dir_for(self, eval_dataset: QADatasetAdapter) -> Path:
        return self._out_path_for(eval_dataset) / f"_chunks_{self._config_hash()}"

    def _chunk_path(self, chunks_dir: Path, chunk_idx: int) -> Path:
        return chunks_dir / f"chunk_{chunk_idx:04d}.json"

    def _load_chunk(self, path: Path) -> list[dict] | None:
        if not path.exists():
            return None
        try:
            with open(path) as f:
                return json.load(f)
        except json.JSONDecodeError as ex:
            logger.warning(f"Corrupt chunk cache at {path} ({ex}); will regenerate")
            return None

    def _save_chunk_atomic(self, path: Path, rows: list[dict]) -> None:
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(rows, f)
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
