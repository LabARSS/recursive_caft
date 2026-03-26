from pydraconf import PydraConfig

from core.complexity_estimation.complexity_estimation_runner import ComplexityEstimationRunner
from core.training.lora_trainer import LoRATrainer, LoRATrainerConfig, LoRATrainingArgs
from core.utils.logger import logger


class ResamplingTrainerConfig(PydraConfig):
    training_args: LoRATrainingArgs


class ResamplingTrainer:
    def __init__(self, config: ResamplingTrainerConfig, tokenizer):
        self.config = config
        self.tokenizer = tokenizer

    def train(self):
        for epoch in range(self.config.training_args.num_train_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.config.training_args.num_train_epochs}...")

            # TODO: build tmp dataset with teacher entropy

            logger.info("Estimating complexity...")
            ComplexityEstimationRunner().estimate()

            logger.info("Training...")
            trainer = LoRATrainer(
                config=LoRATrainerConfig(
                    out_path=f"{self.config.out_path}/epoch_{epoch + 1}",
                    model_id=self.config.model_id,
                    train_dataset=self.config.train_dataset,
                    training_args=self.config.training_args,
                    lora_training_args=self.config.lora_training_args,
                ),
                tokenizer=self.tokenizer,
            )
            trainer.train()
            trainer.unload()
