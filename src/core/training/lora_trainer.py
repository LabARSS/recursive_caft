from typing import Literal, override

from peft import LoraConfig, TaskType, get_peft_model
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM
from transformers.modeling_utils import PreTrainedModel

from core.training.base_trainer import BaseTrainer, BaseTrainerConfig, BaseTrainingArgs


class LoRATrainingArgs(BaseTrainingArgs):
    # Sane overrides for LoRA SFT fine-tuning
    effective_train_batch_size: int = 64
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.06
    weight_decay: float = 0.0


class LoRASpecificTrainingArgs(BaseModel):
    r: int = 8
    alpha: int = 16
    lora_dropout: float = 0.1
    init_lora_weights: Literal[
        "gaussian", "eva", "olora", "pissa", "pissa_niter_[number of iters]", "corda", "loftq", "orthogonal"
    ] = "gaussian"
    target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    bias: Literal["none", "all", "lora_only"] = "none"
    task_type: TaskType = TaskType.CAUSAL_LM
    use_rslora: bool = False


class LoRATrainerConfig(BaseTrainerConfig[LoRATrainingArgs]):
    lora_training_args: LoRASpecificTrainingArgs = Field(default_factory=LoRASpecificTrainingArgs)


class LoRATrainer[TConfig: LoRATrainerConfig = LoRATrainerConfig](BaseTrainer[TConfig]):
    @property
    @override
    def model(self) -> PreTrainedModel:
        if not self._model:
            model = AutoModelForCausalLM.from_pretrained(self.config.model_id)
            peft_config = LoraConfig(
                r=self.config.lora_training_args.r,
                lora_alpha=self.config.lora_training_args.alpha,
                lora_dropout=self.config.lora_training_args.lora_dropout,
                init_lora_weights=self.config.lora_training_args.init_lora_weights,
                target_modules=self.config.lora_training_args.target_modules,
                bias=self.config.lora_training_args.bias,
                task_type=self.config.lora_training_args.task_type,
                use_rslora=self.config.lora_training_args.use_rslora,
            )
            self._model = get_peft_model(model, peft_config)

        assert self._model is not None
        return self._model
