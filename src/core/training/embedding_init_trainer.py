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
from core.training.thinking_tokens import new_token_ids, setup_thinking_tokens
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
        self._new_ids: list[int] = []
        self._base_rows_snapshot: torch.Tensor | None = None

    @property
    @override
    def model(self) -> PreTrainedModel:
        if not self._model:
            model = AutoModelForCausalLM.from_pretrained(self.config.model_id)

            # Idempotent: registers tokens if the upstream script didn't already,
            # resizes the model only if the tokenizer has outgrown the embedding
            # table (skipped for padded-vocab models like Phi-4-mini), and
            # mean-inits the rows at the new-token ids (not necessarily the tail).
            setup_thinking_tokens(self.tokenizer, model)
            new_ids = new_token_ids(self.tokenizer)
            think_id, close_id = new_ids

            logger.info(
                f"new_ids={new_ids}; tokenizer_len={len(self.tokenizer)}; "
                f"embedding_rows={model.get_input_embeddings().weight.shape[0]}; "
                f"tie_word_embeddings={model.config.tie_word_embeddings}"
            )

            in_ptr = model.get_input_embeddings().weight.data_ptr()
            out_layer = model.get_output_embeddings()
            out_ptr = out_layer.weight.data_ptr() if out_layer is not None else None
            logger.info(f"input_emb ptr == output_emb ptr: {in_ptr == out_ptr}")

            think_tok = self.tokenizer.encode("<think>", add_special_tokens=False)
            close_tok = self.tokenizer.encode("</think>", add_special_tokens=False)
            assert think_tok == [think_id], f"<think> tokenizes to {think_tok}, expected [{think_id}]"
            assert close_tok == [close_id], f"</think> tokenizes to {close_tok}, expected [{close_id}]"

            for p in model.parameters():
                p.requires_grad = False
            in_w = model.get_input_embeddings().weight
            in_w.requires_grad = True
            if out_layer is not None:
                out_layer.weight.requires_grad = True

            # Row-scoped backprop: zero grads for every row NOT in new_ids so only
            # the new-token rows are updated. Tied embeddings share the tensor, so
            # a single hook on the input covers lm_head too.
            # Done in-place to avoid allocating a second [vocab, hidden] tensor
            # per backward (meaningful on padded-vocab models like Phi-4-mini).
            vocab = in_w.shape[0]
            keep_col = torch.zeros(vocab, 1, dtype=torch.bool)
            keep_col[torch.tensor(new_ids)] = True

            def _mask_grad(grad: torch.Tensor) -> torch.Tensor:
                grad.mul_(keep_col.to(dtype=grad.dtype, device=grad.device))
                return grad

            in_w.register_hook(_mask_grad)
            if not model.config.tie_word_embeddings and out_layer is not None:
                out_layer.weight.register_hook(_mask_grad)

            # Snapshot everything EXCEPT the new rows, so the post-train assert
            # can verify row-scoped backprop held regardless of vocab layout.
            existing_mask = torch.ones(vocab, dtype=torch.bool)
            existing_mask[torch.tensor(new_ids)] = False
            self._base_rows_snapshot = in_w.detach()[existing_mask].cpu().clone()
            self._new_ids = new_ids

            if self.config.training_args.gradient_checkpointing:
                model.enable_input_require_grads()

            self._model = model

        assert self._model is not None
        return self._model

    @override
    def _run_training(self, trainer):
        trainer.train()

        assert self._base_rows_snapshot is not None and self._new_ids
        weight = trainer.model.get_input_embeddings().weight.detach()
        vocab = weight.shape[0]
        keep_mask = torch.zeros(vocab, dtype=torch.bool)
        keep_mask[torch.tensor(self._new_ids)] = True
        existing = weight[~keep_mask].cpu()
        assert torch.equal(existing, self._base_rows_snapshot), (
            "Pre-existing embedding rows drifted during v0 training — gradient mask failed"
        )

        save_dir = Path(self.config.final_save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        trainer.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        logger.info(f"Saved v0 base to {save_dir}")
