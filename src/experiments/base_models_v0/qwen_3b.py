from pathlib import Path

from transformers import AutoTokenizer

from core.datasets.causal_dataset_adapter import CausalDatasetAdapter
from core.datasets.mmlu.mmlu_reasoning_response_dataset import MMLUReasoningResponseDataset
from core.datasets.qa_dataset import QADatasetConfig
from core.training.embedding_init_trainer import (
    EmbeddingInitTrainer,
    EmbeddingInitTrainerConfig,
    EmbeddingInitTrainingArgs,
)
from core.training.thinking_tokens import setup_thinking_tokens
from core.utils.logger import logger
from experiments.base_models_v0._shared import flatten_distillation_parquet

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
MODEL_NICK = "qwen_3b"
# Cap the reasoning trace so all three v0 bases train on identical data
# (Phi-4-mini's 200k vocab forces the cap; Qwen/Llama inherit it for parity).
MAX_THINKING_CHARS = 12000

REPO_ROOT = Path(__file__).resolve().parents[3]
FINAL_SAVE_DIR = REPO_ROOT / "artifacts" / "base_models_v0" / MODEL_NICK
STAGING_DIR = FINAL_SAVE_DIR.with_name(MODEL_NICK + "_staging")
SOURCE_PARQUET = REPO_ROOT / "data" / "out" / "distillation" / "mmlu_synth_qwen3_a_t0_8.parquet"
FLAT_PARQUET = (
    REPO_ROOT
    / "artifacts"
    / "base_models_v0"
    / "_flattened"
    / f"mmlu_synth_qwen3_a_t0_8_think_le{MAX_THINKING_CHARS}.parquet"
)


def main():
    if (FINAL_SAVE_DIR / "config.json").exists():
        logger.info(f"v0 base already at {FINAL_SAVE_DIR}, skipping. Delete the dir to force a rerun.")
        return

    flat_path = flatten_distillation_parquet(
        SOURCE_PARQUET, FLAT_PARQUET, max_thinking_chars=MAX_THINKING_CHARS
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    setup_thinking_tokens(tokenizer)

    trainer = EmbeddingInitTrainer(
        config=EmbeddingInitTrainerConfig(
            out_path=STAGING_DIR.as_posix(),
            final_save_dir=FINAL_SAVE_DIR.as_posix(),
            model_id=MODEL_NAME,
            train_dataset=CausalDatasetAdapter(
                dataset=MMLUReasoningResponseDataset(
                    config=QADatasetConfig(
                        path=flat_path.as_posix(),
                        dataset_id=f"v0_embedding_init_{MODEL_NICK}",
                    ),
                    tokenizer=tokenizer,
                )
            ),
            training_args=EmbeddingInitTrainingArgs(
                num_train_epochs=1,
                per_device_train_batch_size=2,
            ),
        ),
        tokenizer=tokenizer,
    )
    trainer.train()


if __name__ == "__main__":
    main()
