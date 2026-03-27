from typing import override

from core.complexity_estimation.complexity_estimation_runner import (
    BaseComplexityEstimator,
    ComplexityEstimationRunner,
    ComplexityEstimationRunnerConfig,
    ModelGenerateConfig,
    QADatasetAdapter,
)
from core.training.lora_trainer import LoRATrainer, LoRATrainerConfig
from core.utils.logger import logger


class ResamplingTrainerConfig(LoRATrainerConfig):
    complexity_evaluation_dataset: QADatasetAdapter
    complexity_estimator: BaseComplexityEstimator


class ResamplingTrainer(LoRATrainer[ResamplingTrainerConfig]):
    def _estimate_complexity_for_epoch(self, epoch: int):
        logger.info(f"Estimating complexity for epoch {epoch + 1}...")

        ComplexityEstimationRunner(
            config=ComplexityEstimationRunnerConfig(
                out_path=self._path_for_epoch(epoch).as_posix(),
                answer_field_name="estimation_phase_answer",
                answer_correctness_field_name="estimation_phase_answer_correctness",
                generate_config=ModelGenerateConfig(max_new_tokens=1),
            ),
            complexity_estimator=self.config.complexity_estimator,
        ).estimate(dataset_adapter=self.config.complexity_evaluation_dataset, model=self._trainer.model)

    @override
    def _prepare_data(self): ...

    @override
    def _build_trainer(self, train_ds): ...
