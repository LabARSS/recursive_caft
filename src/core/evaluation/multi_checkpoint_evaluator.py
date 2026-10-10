import gc
import json
from pathlib import Path

import psutil
import torch
from pydraconf import PydraConfig
from transformers import PreTrainedTokenizer

from core.datasets.qa_dataset_adapter import QADatasetAdapter
from core.evaluation.evaluator import EvaluationResult, Evaluator, EvaluatorConfig, GenerationConfig
from core.utils.logger import logger


def _mem_snapshot() -> str:
    rss_gb = psutil.Process().memory_info().rss / 1e9
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        alloc = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        return (
            f"rss={rss_gb:.2f}GB cuda_free={free / 1e9:.2f}GB cuda_total={total / 1e9:.2f}GB "
            f"cuda_alloc={alloc / 1e9:.2f}GB cuda_reserved={reserved / 1e9:.2f}GB"
        )
    return f"rss={rss_gb:.2f}GB cuda=unavailable"


class MultiCheckpointEvaluatorConfig(PydraConfig):
    checkpoints_dir: str
    skip_missing_checkpoints: bool = False
    eval_dataset: QADatasetAdapter | list[QADatasetAdapter]
    base_model_id: str | None = None
    out_path: str | None = None
    summary_filename: str = "summary.json"
    generation: GenerationConfig


class MultiCheckpointEvaluator:
    """Evaluates all checkpoints in a directory sequentially.

    For each checkpoint, loads the model, runs evaluation, then unloads and
    frees GPU memory before proceeding to the next.
    """

    def __init__(self, config: MultiCheckpointEvaluatorConfig, tokenizer: PreTrainedTokenizer | None = None):
        self.config = config
        self.tokenizer = tokenizer

    def _normalize_datasets(self) -> list[QADatasetAdapter]:
        if isinstance(self.config.eval_dataset, list):
            return self.config.eval_dataset
        return [self.config.eval_dataset]

    def evaluate_all(self) -> list[tuple[str, list[EvaluationResult], float | None]]:
        checkpoints_dir = Path(self.config.checkpoints_dir)
        if not checkpoints_dir.is_dir():
            raise NotADirectoryError(f"{checkpoints_dir} is not a directory")

        checkpoint_dirs = sorted(
            checkpoints_dir.glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[1]),
        )

        if not checkpoint_dirs:
            logger.warning(f"No checkpoint-* dirs found in {checkpoints_dir}")
            return []

        logger.info(f"Found {len(checkpoint_dirs)} checkpoints in {checkpoints_dir}")

        results: list[tuple[str, list[EvaluationResult], float | None]] = []

        if self.config.base_model_id:
            logger.info(f"Evaluating base model {self.config.base_model_id} as epoch 0...")
            base_out_path = str(self._out_path / "base_model" / "evals")
            base_config = EvaluatorConfig(
                model_path=self.config.base_model_id,
                eval_dataset=self.config.eval_dataset,
                out_path=base_out_path,
                generation=self.config.generation,
            )
            base_results = Evaluator(base_config, self.tokenizer).evaluate()
            results.append((self.config.base_model_id, base_results, 0.0))

            for r in base_results:
                logger.info(f"base_model: accuracy={r.accuracy:.4f} ({r.correct}/{r.total})")
                if r.num_truncated > 0:
                    pct = r.num_truncated / r.total * 100
                    logger.warning(
                        f"base_model: {r.num_truncated}/{r.total} ({pct:.1f}%) sequences reached max_new_tokens"
                    )

            self._free_vram()

        for ckpt_dir in checkpoint_dirs:
            ckpt_name = ckpt_dir.name
            ckpt_out_path = str(self._out_path / ckpt_name / "evals")

            logger.trace(f"[trace] starting checkpoint={ckpt_name} {_mem_snapshot()}")
            logger.info(f"Evaluating {ckpt_name}...")

            if self.config.skip_missing_checkpoints:
                if not (ckpt_dir / "trainer_state.json").exists():
                    logger.warning(f"Skipping {ckpt_name} because no model file found")
                    continue

            config = EvaluatorConfig(
                model_path=str(ckpt_dir),
                eval_dataset=self.config.eval_dataset,
                out_path=ckpt_out_path,
                generation=self.config.generation,
            )

            try:
                eval_results = Evaluator(config, self.tokenizer).evaluate()
            except Exception:
                from loguru import logger as _raw_logger

                _raw_logger.opt(exception=True).error(f"[trace] checkpoint failed: {ckpt_name}")
                _raw_logger.complete()
                raise
            epoch = self._read_epoch(ckpt_dir)
            results.append((ckpt_name, eval_results, epoch))

            for r in eval_results:
                logger.info(f"{ckpt_name}: accuracy={r.accuracy:.4f} ({r.correct}/{r.total})")
                if r.num_truncated > 0:
                    pct = r.num_truncated / r.total * 100
                    logger.warning(
                        f"{ckpt_name}: {r.num_truncated}/{r.total} ({pct:.1f}%) sequences reached max_new_tokens"
                    )

            self._free_vram()
            logger.trace(f"[trace] done checkpoint={ckpt_name} {_mem_snapshot()}")

        self._save_summary(results)

        self._free_vram()

        return results

    def _read_epoch(self, ckpt_dir: Path) -> float | None:
        state_file = ckpt_dir / "trainer_state.json"
        if not state_file.exists():
            return None
        with open(state_file) as f:
            state = json.load(f)
        return state.get("epoch")

    def _save_summary(self, results: list[tuple[str, list[EvaluationResult], float | None]]) -> None:
        self._out_path.mkdir(parents=True, exist_ok=True)

        datasets = self._normalize_datasets()

        summary: dict[str, list[dict]] = {}
        for ds_idx, ds_adapter in enumerate(datasets):
            dataset_id = ds_adapter.dataset.dataset_id
            summary[dataset_id] = [
                {
                    "epoch": epoch,
                    "checkpoint": ckpt_name,
                    "accuracy": eval_results[ds_idx].accuracy,
                    "total": eval_results[ds_idx].total,
                    "correct": eval_results[ds_idx].correct,
                    "num_truncated": eval_results[ds_idx].num_truncated,
                }
                for ckpt_name, eval_results, epoch in results
            ]

        summary_path = self._out_path / self.config.summary_filename
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Summary saved to {summary_path}")

    @property
    def _out_path(self) -> Path:
        if self.config.out_path:
            return Path(self.config.out_path)

        return Path(self.config.checkpoints_dir)

    def _free_vram(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.mps.is_available():
            torch.mps.empty_cache()
