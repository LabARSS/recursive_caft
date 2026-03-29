from pathlib import Path
from typing import override

from torch.utils.data import IterableDataset
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer

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


class ResamplingDataset(IterableDataset):
    def __init__(self, dataset: AbstractDatasetAdapter, tokenizer: PreTrainedTokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self._logged_samples = False

        self.dataset_path: str | None = None

    def __iter__(self):
        # Calling process_dataset to re-sample dataset after EstimateComplexityCallback runs
        dataset = self.dataset.process_dataset(path_override=self.dataset_path)

        if not self._logged_samples:
            first_sample = dataset[0]
            logger.info("Dataset samples")
            logger.info("Train")
            logger.info(f"Input: {self.tokenizer.decode(first_sample['input_ids'])}")
            labels = [tok for tok in first_sample["labels"] if tok != -100]
            logger.info(f"Labels: {self.tokenizer.decode(labels)}")
            self._logged_samples = True

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
        assert state.epoch is not None

        logger.info(f"Estimating complexity for epoch {state.epoch}...")

        model: PreTrainedModel = kwargs["model"]

        ComplexityEstimationRunner(
            config=ComplexityEstimationRunnerConfig(
                out_path=self.out_path_for_epoch(int(state.epoch)).as_posix(),
                answer_field_name="estimation_phase_answer",
                answer_correctness_field_name="estimation_phase_answer_correctness",
                generate_config=self._complexity_estimation_runner_generation_config,
            ),
            complexity_estimator=self._complexity_estimator,
        ).estimate(dataset_adapter=self._complexity_evaluation_dataset, model=model)

    def out_path_for_epoch(self, epoch: int) -> Path:
        return (
            self._out_path
            / str(epoch)
            / "complexity_estimation"
            / f"{self._complexity_evaluation_dataset.dataset.config.id}.parquet"
        )


class SetResamplingPathCallback(TrainerCallback):
    def __init__(
        self,
        estimation_complexity_callback: EstimateComplexityCallback,
        resampling_ds: ResamplingDataset,
    ) -> None:
        super().__init__()

        self.estimation_complexity_callback = estimation_complexity_callback
        self.resampling_ds = resampling_ds

    @override
    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs) -> None:
        assert state.epoch is not None

        self.resampling_ds.dataset_path = self.estimation_complexity_callback.out_path_for_epoch(
            int(state.epoch)
        ).as_posix()


class ResamplingTrainerConfig(LoRATrainerConfig):
    complexity_evaluation_dataset: QADatasetAdapter
    complexity_estimator: BaseComplexityEstimator
    complexity_estimation_runner_generation_config: ModelGenerateConfig


class ResamplingTrainer(LoRATrainer[ResamplingTrainerConfig]):
    @override
    def _prepare_data(self):
        train_ds = ResamplingDataset(self.config.train_dataset, self.tokenizer)
        return train_ds

    @override
    def _build_trainer(self, train_ds):
        trainer = super()._build_trainer(train_ds)

        self._estimate_complexity_callback = EstimateComplexityCallback(
            complexity_evaluation_dataset=self.config.complexity_evaluation_dataset,
            complexity_estimator=self.config.complexity_estimator,
            complexity_estimation_runner_generation_config=self.config.complexity_estimation_runner_generation_config,
            out_path=self._data_path,
        )
        trainer.add_callback(self._estimate_complexity_callback)
        trainer.add_callback(
            SetResamplingPathCallback(
                estimation_complexity_callback=self._estimate_complexity_callback, resampling_ds=train_ds
            )
        )

        return trainer

    def _path_for_epoch(self, epoch: int) -> Path:
        return self._data_path / str(epoch)

    @property
    def _data_path(self) -> Path:
        return Path(self.config.out_path) / "resampling_trainer_data"
