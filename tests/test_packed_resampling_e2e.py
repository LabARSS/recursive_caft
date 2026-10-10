"""End-to-end CPU smoke test for the packed resampling trainer.

Drives a real ResamplingTrainer + HF Trainer loop (transformers 4.52.3) with packing
enabled, on a tiny randomly-initialized Qwen2 — no GPU, no flash-attn (cross-document
attention leakage inside packs is acceptable here; this tests the *mechanics*):

- every epoch yields exactly len(dataset) rows, so epoch boundaries land on exact
  integers and the complexity-estimation / save-schedule callbacks key correctly;
- null packs never produce NaN losses (the per-chunk num_items_in_batch guard);
- resume from a scheduled checkpoint continues with the same step arithmetic.

Slower than the unit tests (trains 3 tiny epochs + per-epoch estimation) but still
seconds-scale.
"""

import json
import math
from pathlib import Path
from types import SimpleNamespace

from transformers import Qwen2Config, Qwen2ForCausalLM

from core.complexity_estimation.complexity_estimation_runner import ModelGenerateConfig
from core.complexity_estimation.entropy.single_token_entropy_estimator import SingleTokenEntropyEstimator
from core.dataset_samplers.base_sampler import BaseDatasetSamplerConfig
from core.dataset_samplers.entropy_gain_sampler import EntropyGainSampler
from core.datasets.causal_dataset_adapter import CausalDatasetAdapter
from core.datasets.mmlu.mmlu_reasoning_response_dataset import MMLUReasoningResponseDataset
from core.datasets.mmlu.mmlu_single_token_response_dataset import MMLUSingleTokenResponseDataset
from core.datasets.qa_dataset import QADatasetConfig
from core.datasets.qa_dataset_adapter import QADatasetAdapter
from core.training.base_trainer import BaseTrainer, PackingConfig
from core.training.lora_trainer import LoRATrainingArgs
from core.training.resampling_trainer import PackedResamplingDataset, ResamplingTrainer, ResamplingTrainerConfig

TOP_K = 8
EFFECTIVE_BATCH = 4  # == accumulation steps at per_device=1; updates/epoch = 8/4 = 2
UPDATES_PER_EPOCH = TOP_K // EFFECTIVE_BATCH
N_DOCS = 6  # < TOP_K: the null padding must absorb the shortfall


def _tiny_model_dir(tmp_path: Path, tokenizer) -> str:
    # Vocab must cover the thinking tokens setup_thinking_tokens appended past the base vocab.
    vocab_size = math.ceil(len(tokenizer) / 64) * 64
    config = Qwen2Config(
        vocab_size=vocab_size,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=4096,
        tie_word_embeddings=True,
        pad_token_id=tokenizer.pad_token_id,
    )
    model_dir = tmp_path / "tiny_qwen"
    Qwen2ForCausalLM(config).save_pretrained(model_dir)
    return model_dir.as_posix()


def _make_trainer(tokenizer, model_dir: str, out_path: Path, teacher_parquet: Path, num_epochs: int, schedule):
    train_adapter = CausalDatasetAdapter(
        dataset=MMLUReasoningResponseDataset(
            config=QADatasetConfig(path="sentinel-overridden-per-epoch", dataset_id="train"),
            tokenizer=tokenizer,
        ),
        # drop_non_positive=False: a random model's entropy estimates are unreliable, and this
        # test is about trainer mechanics, not selection quality.
        dataset_sampler=EntropyGainSampler(BaseDatasetSamplerConfig(top_k=TOP_K, drop_non_positive=False)),
    )
    return ResamplingTrainer(
        config=ResamplingTrainerConfig(
            training_args=LoRATrainingArgs(
                num_train_epochs=num_epochs,
                per_device_train_batch_size=1,
                effective_train_batch_size=EFFECTIVE_BATCH,
                bf16=False,
                gradient_checkpointing=False,
                torch_compile=False,
                logging_steps=1,
            ),
            packing=PackingConfig(budget=_budget(tokenizer, teacher_parquet)),
            save_schedule=schedule,
            out_path=out_path.as_posix(),
            model_id=model_dir,
            train_dataset=train_adapter,
            complexity_evaluation_dataset=QADatasetAdapter(
                dataset=MMLUSingleTokenResponseDataset(
                    config=QADatasetConfig(path=teacher_parquet.as_posix(), dataset_id="eval"),
                    tokenizer=tokenizer,
                )
            ),
            complexity_estimator=SingleTokenEntropyEstimator(),
            complexity_estimation_runner_generation_config=ModelGenerateConfig(max_new_tokens=1),
            max_failed_estimation_fraction=1.0,  # a random model rarely answers with a letter
        ),
        tokenizer=tokenizer,
    )


def _budget(tokenizer, teacher_parquet: Path) -> int:
    # The per-epoch estimation parquet reuses the teacher rows' text, so their tokenized
    # lengths bound every epoch's docs. 2x the longest doc lets ~2 docs merge per pack,
    # exercising real packs AND null padding.
    probe = CausalDatasetAdapter(
        dataset=MMLUReasoningResponseDataset(
            config=QADatasetConfig(path=teacher_parquet.as_posix(), dataset_id="probe"), tokenizer=tokenizer
        ),
        dataset_sampler=None,
    ).process_dataset()
    return 2 * max(len(ids) for ids in probe["input_ids"])


def _state(checkpoint_dir) -> dict:
    return json.loads((Path(checkpoint_dir) / "trainer_state.json").read_text())


def test_packed_resampling_end_to_end(thinking_tokenizer, tmp_path, make_resampling_df, monkeypatch):
    monkeypatch.setenv("COMPLEXITY_SUPERVISE", "0")  # in-process estimation, no GPU subprocess
    # No nvidia-smi on this machine; the log line is irrelevant to the mechanics under test.
    monkeypatch.setattr(
        "core.training.base_trainer.subprocess.run", lambda *_args, **_kwargs: SimpleNamespace(stdout="no gpu")
    )
    # No flash-attn on CPU. Packs lose per-document attention isolation; fine for mechanics.
    monkeypatch.setattr(BaseTrainer, "_model_load_kwargs", lambda _self: {})

    teacher_parquet = tmp_path / "teacher_entropy.parquet"
    make_resampling_df(N_DOCS).to_parquet(teacher_parquet)
    model_dir = _tiny_model_dir(tmp_path, thinking_tokenizer)
    out_path = tmp_path / "run"

    # --- run 1: epochs 0-1, checkpoint at epoch 2 ---
    trainer1 = _make_trainer(thinking_tokenizer, model_dir, out_path, teacher_parquet, num_epochs=2, schedule=[2])
    assert isinstance(trainer1._prepare_data(), PackedResamplingDataset)
    checkpoint = trainer1.train()
    trainer1.unload()

    assert checkpoint is not None
    state = _state(checkpoint)
    assert state["global_step"] == 2 * UPDATES_PER_EPOCH
    assert state["epoch"] == 2.0  # exact integer: the epoch yielded exactly len(dataset) rows

    # A true resume never revisits epochs 0-1, so their estimation parquets must survive
    # run 2 byte-untouched (a from-scratch retrain would re-estimate and rewrite them —
    # and still end at the same global_step, which is why this is the discriminator).
    est_dir = out_path / "resampling_trainer_data"
    done_parquets = [est_dir / str(e) / "complexity_estimation" / "eval.parquet" for e in range(2)]
    mtimes = {p: p.stat().st_mtime_ns for p in done_parquets}

    # --- run 2: resume from the epoch-2 checkpoint, train epoch 2 (of 3) ---
    trainer2 = _make_trainer(thinking_tokenizer, model_dir, out_path, teacher_parquet, num_epochs=3, schedule=[2, 3])
    checkpoint = trainer2.train()
    trainer2.unload()

    assert checkpoint is not None
    state = _state(checkpoint)
    # Resume arithmetic: epochs_trained = global_step // updates_per_epoch found epoch 2, and
    # exactly one more epoch of updates ran.
    assert state["global_step"] == 3 * UPDATES_PER_EPOCH
    assert state["epoch"] == 3.0
    assert all(p.stat().st_mtime_ns == mtimes[p] for p in done_parquets), (
        "run 2 re-estimated already-trained epochs: it retrained from scratch instead of resuming"
    )

    # Every optimizer step logged a finite loss: null packs contributed zero items, never NaN.
    losses = [entry["loss"] for entry in state["log_history"] if "loss" in entry]
    assert len(losses) >= 3 * UPDATES_PER_EPOCH
    assert all(math.isfinite(loss) for loss in losses), f"non-finite training loss: {losses}"
    # state.epoch advanced in exact 1/updates_per_epoch increments throughout.
    epochs = [entry["epoch"] for entry in state["log_history"] if "epoch" in entry]
    assert all(
        math.isclose(epoch * UPDATES_PER_EPOCH, round(epoch * UPDATES_PER_EPOCH), abs_tol=1e-9) for epoch in epochs
    )

    # Per-epoch complexity estimation ran (or was correctly reused) for every trained epoch.
    for epoch in range(3):
        parquet = out_path / "resampling_trainer_data" / str(epoch) / "complexity_estimation" / "eval.parquet"
        assert parquet.exists(), f"missing estimation parquet for epoch {epoch}"

    # Both checkpoints of the schedule exist (epoch 2 from run 1 survived the resume).
    checkpoints = sorted(p.name for p in out_path.glob("checkpoint-*"))
    assert checkpoints == [f"checkpoint-{2 * UPDATES_PER_EPOCH}", f"checkpoint-{3 * UPDATES_PER_EPOCH}"]
