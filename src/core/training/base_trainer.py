import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any

from datasets import Dataset
from pydantic import BaseModel
from pydraconf import PydraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    PreTrainedTokenizerBase,
    Seq2SeqTrainingArguments,
)
from transformers.trainer_seq2seq import Seq2SeqTrainer

from core.datasets.abstract_dataset_adapter import AbstractDatasetAdapter
from core.training.callbacks.save_by_schedule import SaveByScheduleCallback
from core.utils.last_checkpoint_dir import get_last_checkpoint_dir
from core.utils.seed import set_seed

logger = logging.getLogger(__name__)


def get_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def get_world_size() -> int:
    return max(1, int(os.environ.get("WORLD_SIZE", "1")))


def is_main_process() -> bool:
    return get_rank() == 0


class BaseTrainingArgs(BaseModel):
    num_train_epochs: int
    effective_train_batch_size: int = 256
    per_device_train_batch_size: int
    per_device_eval_batch_size: int = 8
    eval_accumulation_steps: int | None = 1
    predict_with_generate: bool = False
    generation_num_beams: int | None = None
    generation_max_length: int | None = None
    generation_max_new_tokens: int | None = None
    generation_do_sample: bool = False
    ddp_find_unused_parameters: bool | None = False

    learning_rate: float = 2e-5
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    optim: str = "adamw_torch"
    gradient_checkpointing: bool = True
    bf16: bool = True
    report_to: str = "none"
    seed: int = 42
    data_seed: int = 42
    torch_compile: bool = True
    eval_strategy: str = "no"
    save_strategy: str = "epoch"
    logging_steps: int = 10
    logging_first_step: bool = True


class BaseTrainerConfig[TTrainingArgs: BaseTrainingArgs = BaseTrainingArgs](PydraConfig):
    out_path: str
    model_id: str
    train_dataset: AbstractDatasetAdapter
    eval_dataset: AbstractDatasetAdapter | dict[str, AbstractDatasetAdapter] | None = None
    compute_metrics: Any | None = None
    training_args: TTrainingArgs
    save_schedule: list[int] | None = None


class BaseTrainer[TConfig: BaseTrainerConfig[Any] = BaseTrainerConfig]:
    def __init__(self, config: TConfig, tokenizer: PreTrainedTokenizerBase | None = None):
        self.config = config
        self._tokenizer: PreTrainedTokenizerBase | None = tokenizer
        self._model = None

    def train(self):
        if not self._directory_is_empty(self.config.out_path, self.config.training_args.num_train_epochs):
            if is_main_process():
                logger.error(f"BaseTrainerConfig.train -> out_path not empty: {self.config.out_path}")
            return None

        set_seed()
        if is_main_process():
            logger.info(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)

        train_ds, eval_ds = self._prepare_data()
        self._run_training(train_ds, eval_ds)
        return get_last_checkpoint_dir(self.config.out_path)

    @property
    def tokenizer(self):
        if not self._tokenizer:
            self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)

        assert isinstance(self._tokenizer, PreTrainedTokenizerBase), (
            "Tokenizer must be a PreTrainedTokenizerBase, but got {}".format(type(self._tokenizer))
        )

        if self._tokenizer.pad_token is None:
            if is_main_process():
                logger.warning("Tokenizer has no pad token, setting it to eos token")
            self._tokenizer.pad_token = self._tokenizer.eos_token

        return self._tokenizer

    @property
    def model(self):
        if not self._model:
            self._model = AutoModelForCausalLM.from_pretrained(self.config.model_id)

        assert self._model is not None, "Model should be initialized"
        return self._model

    @property
    def data_collator(self):
        return DataCollatorForTokenClassification(
            tokenizer=self.tokenizer,
            padding=True,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )

    @property
    def training_args(self):
        args = self.config.training_args.model_dump(
            exclude={
                "effective_train_batch_size",
                "per_device_train_batch_size",
                "generation_max_new_tokens",
                "generation_do_sample",
            }
        )
        return Seq2SeqTrainingArguments(
            **args,
            **self._batch_size_config(
                self.config.training_args.effective_train_batch_size,
                self.config.training_args.per_device_train_batch_size,
            ),
            output_dir=self.config.out_path,
        )

    def _prepare_data(self):
        train_ds = self.config.train_dataset.process_dataset()
        eval_ds = self._prepare_eval_data()

        if is_main_process():
            logger.info("Dataset samples")
            self._log_dataset_sample("Train", train_ds)
            if isinstance(eval_ds, dict):
                for name, dataset in eval_ds.items():
                    self._log_dataset_sample(f"Eval[{name}]", dataset)
            elif eval_ds is not None:
                self._log_dataset_sample("Eval", eval_ds)

        return train_ds, eval_ds

    def _prepare_eval_data(self) -> Dataset | dict[str, Dataset] | None:
        if self.config.eval_dataset is None:
            return None
        if isinstance(self.config.eval_dataset, dict):
            eval_datasets = {}
            for name, dataset_adapter in self.config.eval_dataset.items():
                eval_datasets[name] = dataset_adapter.process_dataset()
            return eval_datasets
        return self.config.eval_dataset.process_dataset()

    def _log_dataset_sample(self, dataset_name: str, dataset: Dataset) -> None:
        if len(dataset) == 0:
            logger.warning(f"{dataset_name} dataset is empty")
            return

        sample = dataset[0]
        input_ids = [int(token_id) for token_id in sample.get("input_ids", [])]
        label_ids = [int(token_id) for token_id in sample.get("labels", []) if int(token_id) >= 0]

        logger.info(dataset_name)
        if input_ids:
            logger.info(f"Input: {self._decode_preview(input_ids)}")
        if label_ids:
            logger.info(f"Labels: {self._decode_preview(label_ids)}")

    def _decode_preview(self, token_ids: list[int], max_chars: int = 4000) -> str:
        decoded = self.tokenizer.decode(token_ids)
        if len(decoded) <= max_chars:
            return decoded
        return decoded[:max_chars] + "\n...<truncated>..."

    def _run_training(self, train_ds: Dataset, eval_ds: Dataset | dict[str, Dataset] | None):
        if self.config.training_args.generation_num_beams is not None:
            self.model.generation_config.num_beams = self.config.training_args.generation_num_beams
        if self.config.training_args.generation_max_length is not None:
            self.model.generation_config.max_length = self.config.training_args.generation_max_length
        if self.config.training_args.generation_max_new_tokens is not None:
            self.model.generation_config.max_new_tokens = self.config.training_args.generation_max_new_tokens
        self.model.generation_config.do_sample = self.config.training_args.generation_do_sample

        trainer_kwargs = {
            "model": self.model,
            "args": self.training_args,
            "train_dataset": train_ds,
            "data_collator": self.data_collator,
            "processing_class": self.tokenizer,
        }
        if eval_ds is not None:
            trainer_kwargs["eval_dataset"] = eval_ds
        if self.config.compute_metrics is not None:
            trainer_kwargs["compute_metrics"] = self.config.compute_metrics

        trainer = Seq2SeqTrainer(**trainer_kwargs)

        if self.config.save_schedule is not None and is_main_process():
            trainer.add_callback(SaveByScheduleCallback(schedule=self.config.save_schedule))

        latest_checkpoint = self._latest_checkpoint_dir(self.config.out_path)
        if latest_checkpoint is None:
            trainer.train()
        else:
            trainer.train(resume_from_checkpoint=latest_checkpoint)

    def _directory_is_empty(self, directory: str, expected_epochs: int) -> bool:
        path = Path(directory)
        if not path.exists():
            return True
        if not path.is_dir():
            raise Exception("Not a directory!")

        checkpoint_dirs = list(path.glob("checkpoint-*"))
        if not checkpoint_dirs:
            return True

        checkpoint_dirs.sort(key=lambda item: int(item.name.split("-")[1]))
        last_checkpoint = checkpoint_dirs[-1]
        state_file = last_checkpoint / "trainer_state.json"
        if not state_file.exists():
            return True

        with open(state_file, "r") as file:
            state = json.load(file)

        return int(state.get("epoch", 0)) != expected_epochs

    def _batch_size_config(self, effective_batch_size: int, per_device_train_batch_size: int):
        world_size = get_world_size()
        denominator = per_device_train_batch_size * world_size
        assert effective_batch_size % denominator == 0, (
            f"Effective batch size {effective_batch_size} is not divisible by per device batch size {per_device_train_batch_size} * world size {world_size}"
        )
        gradient_accumulation_steps = effective_batch_size // denominator
        return {
            "per_device_train_batch_size": per_device_train_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
        }

    def _latest_checkpoint_dir(self, directory: str) -> str | None:
        path = Path(directory)
        if not path.exists() or not path.is_dir():
            return None

        checkpoint_dirs = [item for item in path.glob("checkpoint-*") if item.is_dir()]
        if not checkpoint_dirs:
            return None

        checkpoint_dirs.sort(key=lambda item: int(item.name.split("-")[1]))
        return str(checkpoint_dirs[-1])
