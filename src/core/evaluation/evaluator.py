import gc
import json
from pathlib import Path

import pandas as pd
import torch
from pydantic import BaseModel
from pydraconf import PydraConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

from core.datasets.qa_dataset import QADataset
from core.datasets.qa_dataset_adapter import QADatasetAdapter
from core.evaluation.phased_batch_generator import BatchGenerator
from core.utils.device import DEVICE_MAP
from core.utils.logger import logger

# BatchGenerator.generate() prefills every prompt upfront and stages each KV
# cache on CPU RAM. For 12k+ MMLU-Pro prompts on a 200k-vocab model like
# Phi-4-mini that's ~200GB of staged KV and the container gets OOM-killed.
# Chunking the prompt list bounds peak CPU RAM to one chunk's staging queue.
CHUNK_SIZE = 1024


class GenerationConfig(BaseModel):
    max_new_tokens: int
    max_batch_size: int
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    torch_compile: bool = True
    attn_implementation: str | None = "flash_attention_2"


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
                logger.info("Compiling model with torch.compile... First forward call will be slow.")
                torch.set_float32_matmul_precision("high")
                torch._dynamo.config.cache_size_limit = 128
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

        num_chunks = (total + CHUNK_SIZE - 1) // CHUNK_SIZE
        for chunk_idx in range(num_chunks):
            start = chunk_idx * CHUNK_SIZE
            end = min(start + CHUNK_SIZE, total)
            chunk_prompts = prompts[start:end]

            logger.info(
                f"Chunk {chunk_idx + 1}/{num_chunks}: prompts [{start}:{end}] "
                f"({len(chunk_prompts)} samples)"
            )

            generator = BatchGenerator(
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=self.config.generation.max_new_tokens,
                max_batch_size=self.config.generation.max_batch_size,
                temperature=self.config.generation.temperature,
                top_p=self.config.generation.top_p,
                top_k=self.config.generation.top_k,
            )

            gen_result = generator.generate(chunk_prompts)
            num_truncated += gen_result.num_truncated

            for offset, gen_ids in enumerate(gen_result.sequences):
                row = ds[start + offset]
                response = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

                try:
                    parsed_answer, is_correct = qa_dataset.verify_assistant_response(row, response)
                except Exception as ex:
                    logger.warning(f"Error verifying row {row['row_id']}: {ex}")
                    parsed_answer = response
                    is_correct = False

                if is_correct:
                    correct += 1

                all_results.append(
                    {
                        "row_id": row["row_id"],
                        "response": response,
                        "parsed_answer": parsed_answer,
                        "is_correct": is_correct,
                    }
                )

            # Release the chunk's CPU-staged KV caches before the next chunk
            # prefills its own. Without this, peak CPU RAM keeps climbing.
            del generator, gen_result
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if num_truncated > 0:
            pct = num_truncated / total * 100
            logger.warning(
                f"Generation reached max_new_tokens ({self.config.generation.max_new_tokens}) "
                f"for {num_truncated}/{total} sequences ({pct:.1f}%)"
            )

        accuracy = correct / total if total > 0 else 0.0
        result = EvaluationResult(
            accuracy=accuracy, total=total, correct=correct, num_truncated=num_truncated
        )

        logger.info(f"Evaluation complete: accuracy={accuracy:.4f} ({correct}/{total})")

        self._save_results(eval_dataset, result, all_results)

        return result

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
