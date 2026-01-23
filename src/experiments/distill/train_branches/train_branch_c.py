import logging
import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Config
BASE_MODEL = "unsloth/Qwen2.5-3B-Instruct"
DATA_FILE = "data/out/sft_data/branch_c_sft.parquet"
OUTPUT_DIR = "data/out/models/branch_c"
MAX_SEQ_LENGTH = 2048
BATCH_SIZE = 4
GRAD_ACCUM = 4
EPOCHS = 3
LR = 2e-4

logging.info(f"Loading model: {BASE_MODEL}")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=torch.bfloat16,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0.0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

logging.info(f"Loading data: {DATA_FILE}")
dataset = load_dataset("parquet", data_files=DATA_FILE, split="train")
logging.info(f"Loaded {len(dataset)} samples")

def formatting_func(example):
    return tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="messages",
    formatting_func=formatting_func,
    max_seq_length=MAX_SEQ_LENGTH,
    args=TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        warmup_steps=50,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=42,
    ),
)

logging.info("Starting training...")
trainer.train()

logging.info(f"Saving model to {OUTPUT_DIR}")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
logging.info("Done!")
