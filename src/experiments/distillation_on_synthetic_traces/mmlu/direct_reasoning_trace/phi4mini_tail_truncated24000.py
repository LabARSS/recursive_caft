from pathlib import Path

from transformers import AutoTokenizer

from core.datasets.causal_dataset_adapter import CausalDatasetAdapter
from core.datasets.mmlu.mmlu_reasoning_response_dataset import MMLUReasoningResponseDataset
from core.datasets.qa_dataset import QADatasetConfig
from core.datasets.qa_dataset_adapter import QADatasetAdapter
from core.evaluation.multi_checkpoint_evaluator import (
    GenerationConfig,
    MultiCheckpointEvaluator,
    MultiCheckpointEvaluatorConfig,
)
from core.training.base_trainer import PackingConfig
from core.training.lora_trainer import (
    LoRASpecificTrainingArgs,
    LoRATrainer,
    LoRATrainerConfig,
    LoRATrainingArgs,
    phi4_mini_lora_target_modules,
)
from core.training.packing_budgets import packing_budget
from core.training.thinking_tokens import setup_thinking_tokens

MODEL_NAME = Path(__file__).parent.joinpath("../../../../../artifacts/base_models_v0/phi4_mini").as_posix()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
setup_thinking_tokens(tokenizer)

OUT_PATH = (
    Path(__file__)
    .parent.joinpath(
        "../../../../../artifacts/distillation_on_synthetic_traces/mmlu/direct_reasoning_trace/phi4_mini_tail_truncated24000"
    )
    .as_posix()
)

trainer = LoRATrainer(
    config=LoRATrainerConfig(
        out_path=OUT_PATH,
        model_id=MODEL_NAME,
        train_dataset=CausalDatasetAdapter(
            dataset=MMLUReasoningResponseDataset(
                config=QADatasetConfig(
                    path=Path(__file__)
                    .parent.joinpath(
                        "../../../../../data/out/splits/random/mmlu/train_distilled_deepseek_v4_flash_regenerate_incorrect_w_large_tail_truncated24000.parquet"
                    )
                    .as_posix(),
                    dataset_id="train_distilled_deepseek_v4_flash_regenerate_incorrect_w_large_tail_truncated24000",
                ),
                tokenizer=tokenizer,
            )
        ),
        training_args=LoRATrainingArgs(num_train_epochs=20, per_device_train_batch_size=2),
        lora_training_args=LoRASpecificTrainingArgs(
            train_thinking_token_embeddings=True, target_modules=phi4_mini_lora_target_modules
        ),
        packing=PackingConfig(budget=packing_budget("phi4_mini")),
        save_schedule=[1, 3, 5, 10, 15, 20],
    ),
    tokenizer=tokenizer,
)
trainer.train()
trainer.unload()

cot_evaluator = MultiCheckpointEvaluator(
    config=MultiCheckpointEvaluatorConfig(
        checkpoints_dir=OUT_PATH,
        eval_dataset=QADatasetAdapter(
            dataset=MMLUReasoningResponseDataset(
                config=QADatasetConfig(
                    path=Path(__file__)
                    .parent.joinpath("../../../../../data/out/splits/random/mmlu/test.parquet")
                    .as_posix(),
                    dataset_id="mmlu_random_test",
                ),
                tokenizer=tokenizer,
            ),
            add_thinking_start_token=True,
        ),
        generation=GenerationConfig(max_new_tokens=8500, max_thinking_tokens=8192, max_batch_size=256),
        summary_filename="summary_reasoning_evals.json",
    ),
    tokenizer=tokenizer,
)
cot_evaluator.evaluate_all()
