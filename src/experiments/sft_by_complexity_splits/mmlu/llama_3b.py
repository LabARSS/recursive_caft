from pathlib import Path

from transformers import AutoTokenizer

from core.datasets.mmlu.mmlu_single_token_response_dataset import MMLUSingleTokenResponseDataset, QADatasetConfig
from core.datasets.qa_dataset_adapter import QADatasetAdapter
from core.training.lora_trainer import LoRATrainer, LoRATrainerConfig, LoRATrainingArgs

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

trainer = LoRATrainer(
    config=LoRATrainerConfig(
        out_path=Path(__file__).parent.joinpath("../../../../artifacts/sft_by_complexity_splits/llama_3b").as_posix(),
        model_id=MODEL_NAME,
        train_dataset=QADatasetAdapter(
            dataset=MMLUSingleTokenResponseDataset(
                config=QADatasetConfig(
                    path=Path(__file__)
                    .parent.joinpath("../../../../data/out/splits/single_token_entropy/qwen_3b/group0_train.parquet")
                    .as_posix()
                ),
                tokenizer=tokenizer,
            )
        ),
        training_args=LoRATrainingArgs(num_train_epochs=20, per_device_train_batch_size=32),
        save_schedule=[1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20],
    ),
    tokenizer=tokenizer,
)
trainer.train()
