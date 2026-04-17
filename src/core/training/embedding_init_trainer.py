"""
Trainer that registers <think>/</think> as new special tokens, mean-inits the
new embedding rows, freezes the entire backbone, and fine-tunes ONLY the
input embedding (and lm_head, which is tied for our target models) for a short
low-LR pass. The result is saved as a plain HF checkpoint for downstream
experiments to consume via AutoModelForCausalLM.from_pretrained(<v0 dir>).

Not a LoRA trainer: LoRA cannot reach the embedding matrix without the
modules_to_save footgun (PEFT issues #1750, #2777 around tied weights).
"""

from pathlib import Path
from typing import override

from transformers import AutoModelForCausalLM
from transformers.modeling_utils import PreTrainedModel

from core.training.base_trainer import BaseTrainer, BaseTrainerConfig, BaseTrainingArgs
from core.training.thinking_tokens import setup_thinking_tokens
from core.utils.logger import logger


class EmbeddingInitTrainingArgs(BaseTrainingArgs):
    learning_rate: float = 1e-5
    num_train_epochs: int = 1
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    gradient_checkpointing: bool = True
    torch_compile: bool = False
    save_strategy: str = "no"


class EmbeddingInitTrainerConfig(BaseTrainerConfig[EmbeddingInitTrainingArgs]):
    final_save_dir: str


class EmbeddingInitTrainer(BaseTrainer[EmbeddingInitTrainerConfig]):
    @property
    @override
    def model(self) -> PreTrainedModel:
        if not self._model:
            model = AutoModelForCausalLM.from_pretrained(self.config.model_id)
            _, num_added = setup_thinking_tokens(self.tokenizer, model)
            logger.info(
                f"Added {num_added} special tokens; vocab={len(self.tokenizer)}; "
                f"tie_word_embeddings={model.config.tie_word_embeddings}"
            )

            in_ptr = model.get_input_embeddings().weight.data_ptr()
            out_layer = model.get_output_embeddings()
            out_ptr = out_layer.weight.data_ptr() if out_layer is not None else None
            logger.info(f"input_emb ptr == output_emb ptr: {in_ptr == out_ptr}")

            think_ids = self.tokenizer.encode("<think>", add_special_tokens=False)
            close_ids = self.tokenizer.encode("</think>", add_special_tokens=False)
            assert len(think_ids) == 1, f"<think> tokenizes to {think_ids}, expected single id"
            assert len(close_ids) == 1, f"</think> tokenizes to {close_ids}, expected single id"

            for p in model.parameters():
                p.requires_grad = False
            model.get_input_embeddings().weight.requires_grad = True
            if out_layer is not None:
                out_layer.weight.requires_grad = True

            if self.config.training_args.gradient_checkpointing:
                model.enable_input_require_grads()

            self._model = model

        assert self._model is not None
        return self._model

    @override
    def _run_training(self, trainer):
        trainer.train()
        save_dir = Path(self.config.final_save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        trainer.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        logger.info(f"Saved v0 base to {save_dir}")
