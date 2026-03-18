import gc
import json
import subprocess
from pathlib import Path
from typing import Any

import torch

from pydantic import BaseModel
from pydraconf import PydraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    PreTrainedTokenizer,
    Seq2SeqTrainingArguments,
)
from transformers.trainer_seq2seq import Seq2SeqTrainer

from core.datasets.abstract_dataset_adapter import AbstractDatasetAdapter
from core.training.callbacks.save_by_schedule import SaveByScheduleCallback
from core.utils.last_checkpoint_dir import get_last_checkpoint_dir
from core.utils.logger import logger
from core.utils.seed import set_seed


class BaseTrainingArgs(BaseModel):
    num_train_epochs: int
    effective_train_batch_size: int = 256
    per_device_train_batch_size: int

    # Sane defaults for SFT fine-tuning
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
    save_strategy: str = "epoch"
    logging_steps: int = 10
    logging_first_step: bool = True


class BaseTrainerConfig[TTrainingArgs: BaseTrainingArgs = BaseTrainingArgs](PydraConfig):
    out_path: str
    model_id: str
    train_dataset: AbstractDatasetAdapter
    training_args: TTrainingArgs
    save_schedule: list[int] | None = None


class BaseTrainer[TConfig: BaseTrainerConfig[Any] = BaseTrainerConfig]:
    def __init__(self, config: TConfig, tokenizer: PreTrainedTokenizer | None = None):
        self.config = config
        self._tokenizer: PreTrainedTokenizer | None = tokenizer
        self._model: AutoModelForCausalLM | None = None

    def train(self):
        if not self._directory_is_empty(self.config.out_path, self.config.training_args.num_train_epochs):
            logger.error("BaseTrainerConfig.train -> out_path not empty", self.config.out_path)
            return None

        set_seed()

        logger.info(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)

        train_ds = self._prepare_data()
        self._run_training(train_ds)

        return get_last_checkpoint_dir(self.config.out_path)

    @property
    def tokenizer(self):
        if not self._tokenizer:
            self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)

        assert self._tokenizer is not None, "Tokenizer should be initialized"

        if self._tokenizer.pad_token is None:
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
            tokenizer=self.tokenizer, padding=True, pad_to_multiple_of=8, return_tensors="pt"
        )

    @property
    def training_args(self):
        return Seq2SeqTrainingArguments(
            **self.config.training_args.model_dump(
                exclude={"effective_train_batch_size", "per_device_train_batch_size", "gradient_accumulation_steps"}
            ),
            **self._batch_size_config(
                self.config.training_args.effective_train_batch_size,
                self.config.training_args.per_device_train_batch_size,
            ),
            output_dir=self.config.out_path,
        )

    def _prepare_data(self):
        train_ds = self.config.train_dataset.process_dataset()
        logger.info("Dataset samples")
        logger.info("Train")
        logger.info(f"Input: {self.tokenizer.decode(train_ds[0]['input_ids'])}")
        labels = [tok for tok in train_ds[0]["labels"] if tok != -100]
        logger.info(f"Labels: {self.tokenizer.decode(labels)}")

        return train_ds

    def _run_training(self, train_ds):
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_ds,
            data_collator=self.data_collator,
            processing_class=self.tokenizer,
        )

        if self.config.save_schedule is not None:
            trainer.add_callback(SaveByScheduleCallback(schedule=self.config.save_schedule))

        has_checkpoint = get_last_checkpoint_dir(self.config.out_path) is not None
        logger.info(f"Has checkpoint: {has_checkpoint}")
        trainer.train(resume_from_checkpoint=has_checkpoint)

    def unload(self):
        self._model = None
        self._tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _directory_is_empty(self, directory: str, expected_epochs: int) -> bool:
        p = Path(directory)
        if not p.exists():
            return True
        if not p.is_dir():
            raise Exception("Not a directory!")

        checkpoint_dirs = list(p.glob("checkpoint-*"))
        if not checkpoint_dirs:
            return True

        checkpoint_dirs.sort(key=lambda x: int(x.name.split("-")[1]))
        last_checkpoint = checkpoint_dirs[-1] if checkpoint_dirs else None

        if last_checkpoint:
            state_file = last_checkpoint / "trainer_state.json"
            if state_file.exists():
                with open(state_file, "r") as f:
                    state = json.load(f)
                    if int(state.get("epoch", 0)) == expected_epochs:
                        return False

        return True

    def _batch_size_config(self, effective_batch_size: int, per_device_train_batch_size: int):
        gradient_accumulation_steps = effective_batch_size // per_device_train_batch_size
        assert effective_batch_size % per_device_train_batch_size == 0, (
            f"Effective batch size {effective_batch_size} is not divisible by per device batch size {per_device_train_batch_size}"
        )
        return {
            "per_device_train_batch_size": per_device_train_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
        }
