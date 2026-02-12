import json
import subprocess
from pathlib import Path

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

from core.datasets.base_dataset_adapter import BaseDatasetAdapter
from core.training.callbacks.eval_every_n_epoch import EvalEveryNEpochsCallback
from core.training.callbacks.save_and_log_weights import SaveOnEpochEndAndLogWeightsCallback
from core.training.callbacks.save_every_n_epoch import SaveEveryNEpochsCallback
from core.utils.last_checkpoint_dir import get_last_checkpoint_dir
from core.utils.logger import logger
from core.utils.seed import set_seed


class TrainingArgs(BaseModel):
    epochs: int


class BaseTrainerConfig(PydraConfig):
    out_path: str
    model_id: str
    train_dataset: BaseDatasetAdapter
    test_datasets: list[BaseDatasetAdapter]
    training_args: TrainingArgs


class BaseTrainer:
    def __init__(self, config: BaseTrainerConfig):
        self.config = config
        self._tokenizer: PreTrainedTokenizer | None = None
        self._model: AutoModelForCausalLM | None = None

    def train(self):
        if not self._directory_is_empty(self.config.out_path, self.config.training_args.epochs):
            logger.error("BaseTrainerConfig.train -> out_path not empty", self.config.out_path)
            return None

        set_seed()

        logger.info(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)

        train_ds, test_combined_ds_dict = self._prepare_data()
        self._run_training(train_ds, test_combined_ds_dict)

        return get_last_checkpoint_dir(self.config.out_path)

    @property
    def tokenizer(self):
        if not self._tokenizer:
            self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)

        if self._tokenizer.pad_token is None:
            logger.warning("Tokenizer has no pad token, setting it to eos token")
            self._tokenizer.pad_token = self._tokenizer.eos_token

        return self._tokenizer

    @property
    def model(self):
        if not self._model:
            self._model = AutoModelForCausalLM.from_pretrained(self.config.model_id)
        return self._model

    @property
    def data_collator(self):
        return DataCollatorForTokenClassification(
            tokenizer=self.tokenizer, padding=True, pad_to_multiple_of=8, return_tensors="pt"
        )

    @property
    def training_args(self):
        return Seq2SeqTrainingArguments(**self.config.training_args.model_dump())

    def _prepare_data(self):
        train_ds = self.config.train_dataset.process_dataset(self.tokenizer)
        test_combined_ds_dict = {
            dataset_adapter.id: dataset_adapter.process_dataset(self.tokenizer)
            for dataset_adapter in self.config.test_datasets
        }
        logger.info("Dataset samples")
        logger.info("Train")
        logger.info(f"Input: {self.tokenizer.decode(train_ds[0]['input_ids'])}")
        logger.info(f"Labels: {self.tokenizer.decode(train_ds[0]['labels'])}")
        for ds_id, test_ds in test_combined_ds_dict.items():
            logger.info(f"Test dataset: {ds_id}")
            logger.info(f"Test input: {self.tokenizer.decode(test_ds[0]['input_ids'])}")
            logger.info(f"Test labels: {self.tokenizer.decode(test_ds[0]['labels'])}")

        return train_ds, test_combined_ds_dict

    def _run_training(self, train_ds, test_combined_ds_dict):
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_ds,
            eval_dataset=test_combined_ds_dict,
            data_collator=self.data_collator,
            compute_metrics=self._compute_metrics,
            processing_class=self.tokenizer,
        )

        trainer.add_callback(
            SaveOnEpochEndAndLogWeightsCallback(
                output_dir=self.config.out_path,
                save_full_model_for_non_lora=False,
            )
        )

        trainer.add_callback(SaveEveryNEpochsCallback(n_epochs=4))
        trainer.add_callback(EvalEveryNEpochsCallback(schedule=[(1, 5, 1), (6, 10, 2), (11, 30, 5)]))

        trainer.train(resume_from_checkpoint=True)

    def _compute_metrics(self, eval_pred):
        # TODO
        pass

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
