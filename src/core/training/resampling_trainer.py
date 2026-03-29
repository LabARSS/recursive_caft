from pathlib import Path
from typing import override

from torch.utils.data import IterableDataset
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from transformers.modeling_utils import PreTrainedModel

from core.complexity_estimation.complexity_estimation_runner import (
    BaseComplexityEstimator,
    ComplexityEstimationRunner,
    ComplexityEstimationRunnerConfig,
    ModelGenerateConfig,
    QADatasetAdapter,
)
from core.datasets.abstract_dataset_adapter import AbstractDatasetAdapter
from core.training.lora_trainer import LoRATrainer, LoRATrainerConfig
from core.utils.logger import logger


class ResamplingDatasetAdapter(IterableDataset):
    def __init__(self, dataset: AbstractDatasetAdapter):
        self.dataset = dataset

    def __iter__(self):
        dataset = self.dataset.process_dataset()
        yield from dataset


class EstimateComplexityCallback(TrainerCallback):
    def __init__(
        self,
        complexity_evaluation_dataset: QADatasetAdapter,
        complexity_estimator: BaseComplexityEstimator,
        complexity_estimation_runner_generation_config: ModelGenerateConfig,
        out_path: Path,
    ) -> None:
        super().__init__()

        self._complexity_evaluation_dataset = complexity_evaluation_dataset
        self._complexity_estimator = complexity_estimator
        self._complexity_estimation_runner_generation_config = complexity_estimation_runner_generation_config
        self._out_path = out_path

    @override
    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs) -> None:
        logger.info(f"Estimating complexity for epoch {state.epoch}...")

        model: PreTrainedModel = kwargs["model"]

        ComplexityEstimationRunner(
            config=ComplexityEstimationRunnerConfig(
                out_path=self._out_path.as_posix(),
                answer_field_name="estimation_phase_answer",
                answer_correctness_field_name="estimation_phase_answer_correctness",
                generate_config=self._complexity_estimation_runner_generation_config,
            ),
            complexity_estimator=self._complexity_estimator,
        ).estimate(dataset_adapter=self._complexity_evaluation_dataset, model=model)


class ResamplingTrainerConfig(LoRATrainerConfig):
    complexity_evaluation_dataset: QADatasetAdapter
    complexity_estimator: BaseComplexityEstimator
    complexity_estimation_runner_generation_config: ModelGenerateConfig


class ResamplingTrainer(LoRATrainer[ResamplingTrainerConfig]):
    @override
    def _prepare_data(self): ...

    @override
    def _build_trainer(self, train_ds):
        trainer = super()._build_trainer(train_ds)

        trainer.add_callback(
            EstimateComplexityCallback(
                complexity_evaluation_dataset=self.config.complexity_evaluation_dataset,
                complexity_estimator=self.config.complexity_estimator,
                complexity_estimation_runner_generation_config=self.config.complexity_estimation_runner_generation_config,
                out_path=Path(__file__).parent.joinpath("TODO"),
            )
        )

        return trainer
