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
from core.training.lora_trainer import (
    LoRASpecificTrainingArgs,
    LoRATrainer,
    LoRATrainerConfig,
    LoRATrainingArgs,
)
from core.training.thinking_tokens import setup_thinking_tokens

MODEL_NAME = Path(__file__).parent.joinpath("../../../../../artifacts/base_models_v0/llama_3b").as_posix()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
setup_thinking_tokens(tokenizer)

OUT_PATH = (
    Path(__file__)
    .parent.joinpath(
        "../../../../../artifacts/distillation_on_synthetic_traces/mmlu/direct_reasoning_trace_correct_only/llama_3b_head_truncated8192"
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
                        "../../../../../data/out/splits/random/mmlu/train_correct_only_distilled_deepseek_v4_flash_regenerate_incorrect_w_large.parquet"
                    )
                    .as_posix(),
                    dataset_id="mmlu_train_correct_only_distilled_deepseek_v4_flash_regenerate_incorrect_w_large",
                ),
                tokenizer=tokenizer,
            )
        ),
        training_args=LoRATrainingArgs(num_train_epochs=20, per_device_train_batch_size=8),
        lora_training_args=LoRASpecificTrainingArgs(train_thinking_token_embeddings=True),
        save_schedule=[1, 2, 3, 5, 7, 10, 15, 20],
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
