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

import torch
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._num_new: int = 0
        self._base_rows_snapshot: torch.Tensor | None = None

    @property
    @override
    def model(self) -> PreTrainedModel:
        if not self._model:
            model = AutoModelForCausalLM.from_pretrained(self.config.model_id)
            _, num_added = setup_thinking_tokens(self.tokenizer, model)
            assert num_added > 0, (
                f"Tokenizer for {self.config.model_id} already has <think>/</think>; "
                "v0 training is only meaningful when new tokens are actually added."
            )
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
            in_w = model.get_input_embeddings().weight
            in_w.requires_grad = True
            if out_layer is not None:
                out_layer.weight.requires_grad = True

            # Row-scoped backprop: zero grads for every pre-existing row so only
            # the new-token rows are updated. Tied embeddings share the tensor,
            # so a single hook on the input covers lm_head too.
            def _mask_grad(grad: torch.Tensor) -> torch.Tensor:
                masked = grad.clone()
                masked[:-num_added] = 0
                return masked

            in_w.register_hook(_mask_grad)
            if not model.config.tie_word_embeddings and out_layer is not None:
                out_layer.weight.register_hook(_mask_grad)

            self._base_rows_snapshot = in_w.detach()[:-num_added].cpu().clone()
            self._num_new = num_added

            if self.config.training_args.gradient_checkpointing:
                model.enable_input_require_grads()

            self._model = model

        assert self._model is not None
        return self._model

    @override
    def _run_training(self, trainer):
        trainer.train()

        assert self._base_rows_snapshot is not None and self._num_new > 0
        after = trainer.model.get_input_embeddings().weight.detach()[: -self._num_new].cpu()
        assert torch.equal(after, self._base_rows_snapshot), (
            "Pre-existing embedding rows drifted during v0 training — gradient mask failed"
        )

        save_dir = Path(self.config.final_save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        trainer.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        logger.info(f"Saved v0 base to {save_dir}")
