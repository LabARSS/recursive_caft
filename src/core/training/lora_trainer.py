from typing import Literal, override

from peft import LoraConfig, TaskType, get_peft_model
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM
from transformers.modeling_utils import PreTrainedModel

from core.training.base_trainer import BaseTrainer, BaseTrainerConfig, BaseTrainingArgs
from core.training.callbacks.save_thinking_token_rows import SaveThinkingTokenRowsCallback
from core.training.row_scoped_embedding_training import (
    RowScopedSnapshot,
    assert_no_row_drift,
    install_row_scoped_grad,
)
from core.training.thinking_tokens import new_token_ids


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
    # When True, the <think>/</think> rows of embed_tokens (and lm_head if
    # untied) are unfrozen during the LoRA stage with a row-scoped grad mask.
    # The two updated rows are persisted alongside each checkpoint and must
    # be reloaded at eval time. Only meaningful when the experiment's tokenizer
    # has been through setup_thinking_tokens(...).
    train_thinking_token_embeddings: bool = False


class LoRASpecificTrainingArgsMedium(LoRASpecificTrainingArgs):
    r: int = 32
    alpha: int = 64
    lora_dropout: float = 0.05


class LoRASpecificTrainingArgsLarge(LoRASpecificTrainingArgs):
    r: int = 64
    alpha: int = 128
    lora_dropout: float = 0.05


# Phi-4-mini is Phi3ForCausalLM, which FUSES the attention and MLP input
# projections into qkv_proj / gate_up_proj. The default target_modules list
# (q/k/v/gate/up_proj) is Llama/Qwen naming and matches NONE of these, so PEFT
# would silently adapt only o_proj + down_proj (it doesn't error because those
# two do match). That cripples the adapter and is why phi4 trained far worse.
phi4_mini_lora_target_modules = ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]


class LoRATrainerConfig(BaseTrainerConfig[LoRATrainingArgs]):
    lora_training_args: LoRASpecificTrainingArgs = Field(default_factory=LoRASpecificTrainingArgs)


class LoRATrainer[TConfig: LoRATrainerConfig = LoRATrainerConfig](BaseTrainer[TConfig]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._snapshot: RowScopedSnapshot | None = None

    @property
    @override
    def model(self) -> PreTrainedModel:
        if not self._model:
            model = AutoModelForCausalLM.from_pretrained(self.config.model_id, **self._model_load_kwargs())
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

            if self.config.lora_training_args.train_thinking_token_embeddings:
                new_ids = new_token_ids(self.tokenizer)
                self._snapshot = install_row_scoped_grad(self._model, new_ids)
                if self.config.training_args.gradient_checkpointing:
                    # PEFT enables this for LoRA-trainable modules, but we are
                    # piggybacking on a non-LoRA-managed parameter (embed_tokens).
                    # Be explicit so the grad path through the embedding is kept.
                    self._model.enable_input_require_grads()

        assert self._model is not None
        return self._model

    @override
    def _build_trainer(self, train_ds):
        trainer = super()._build_trainer(train_ds)
        if self.config.lora_training_args.train_thinking_token_embeddings:
            assert self._snapshot is not None
            trainer.add_callback(SaveThinkingTokenRowsCallback(new_ids=self._snapshot.new_ids))
        return trainer

    @override
    def _run_training(self, trainer):
        super()._run_training(trainer)
        if self.config.lora_training_args.train_thinking_token_embeddings:
            assert self._snapshot is not None
            assert_no_row_drift(trainer.model, self._snapshot)
