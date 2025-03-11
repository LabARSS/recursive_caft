import re
import torch
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    AutoConfig
)
import wandb
import os
import pandas as pd
from datasets import Dataset
from torch.utils.data import DataLoader
import argparse
import sys
import platform
from datetime import datetime
import numpy as np
from tqdm import tqdm
import logging
import time  # Добавлен импорт модуля time
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from ast import literal_eval
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

ANSWER_MARKER = ["[[", "]]"]
ANSWER_PATTERN = re.compile(r'.*?\[\[\s*(\d+)\s*\]\].*?', re.DOTALL)

def cot_sys_prompt(subject: str | None = None):
    if subject is not None:
        sys_msg = f"The following are multiple choice questions about {subject}."
    else:
        sys_msg = "The following are multiple choice questions."
    sys_msg += f" Analyze question and answer with only one number in {ANSWER_MARKER[0]}number{ANSWER_MARKER[1]} format."
    return sys_msg


def calculate_accuracy(logits, labels, tokenizer):
    predictions = torch.argmax(logits, dim=-1)
    shifted_labels = labels[..., 1:].contiguous()
    shifted_predictions = predictions[..., :-1].contiguous()
    mask = shifted_labels != -100
    mask = mask & (shifted_labels != tokenizer.eos_token_id)
    correct = (shifted_predictions == shifted_labels) & mask
    return correct.float().sum() / mask.float().sum()



def evaluate_qa(model, df, tokenizer, desc="Validating", batch_size=1):
    model.eval()
    total_correct = 0
    total_errors = 0
    processed_samples = 0
    pbar = tqdm(total=len(df), desc=desc, leave=False)

    # Добавлено: счетчик для логирования примеров
    example_count = 0  
    
    # Создаем батчи
    indices = list(range(len(df)))
    for i in range(0, len(df), batch_size):
        batch_indices = indices[i:i+batch_size]
        batch_rows = [df.iloc[idx] for idx in batch_indices]
        batch_results = []

        try:
            # Подготовка батча
            prompts = []
            correct_answers = []
            raw_questions = []  # Добавлено: для логирования
            
            for row in batch_rows:
                try:
                    correct_answer = str(int(row['distill_answer'])).strip()
                    options = literal_eval(row['options'])
                    num_choices = len(options)
                    
                    if not (1 <= int(correct_answer) <= num_choices):
                        raise ValueError(f"Invalid answer {correct_answer}")
                        
                    prompt = format_prompt_qa(row, include_answer=False)
                    prompts.append(prompt)
                    correct_answers.append(correct_answer)
                    raw_questions.append(row['question'])  # Добавлено
                    
                except Exception as e:
                    batch_results.append(("error", str(e)))
                    continue

            # Токенизация батча
            inputs = tokenizer(
                prompts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=800
            ).to(model.device)

            # Добавлено: логирование первого примера в батче
            if example_count < 1:  # Записать первые 1 примера
                logging.info("\n" + "="*50 + f"\nEXAMPLE INPUT {example_count+1}:")
                logging.info(f"RAW QUESTION:\n{raw_questions[0]}")
                logging.info(f"FULL PROMPT:\n{prompts[0]}")
                logging.info(f"TOKENIZED INPUT:\n{inputs.input_ids[0]}")
                example_count += 1

            # Генерация
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_new_tokens=1500,  # Увеличим для демонстрации
                    num_return_sequences=1,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id
                )

            # Добавлено: логирование вывода
            responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            if example_count <= 2:
                logging.info("\n" + "="*50 + "\nMODEL OUTPUTS:")
                for idx, (resp, correct) in enumerate(zip(responses, correct_answers)):
                    if idx >= 3: break
                    logging.info(f"Response {idx+1}:")
                    logging.info(f"GENERATED:\n{resp}")
                    logging.info(f"EXPECTED: [[{correct}]]\n")
                example_count +=1
                    
            for response, correct in zip(responses, correct_answers):
                try:
                    matches = ANSWER_PATTERN.findall(response)
                    generated = matches[-1] if matches else None
                    is_correct = generated == correct
                    batch_results.append(("success", is_correct))
                    
                    # Логирование
                    logging.debug(f"Generated: {generated} | Correct: {correct}")
                    
                except Exception as e:
                    batch_results.append(("error", str(e)))

        except Exception as e:
            batch_results = [("error", str(e))]*len(batch_rows)

        # Обработка результатов
        for result in batch_results:
            pbar.update(1)
            if result[0] == "success":
                total_correct += result[1]
                processed_samples += 1
            else:
                total_errors += 1
                logging.error(f"Error: {result[1]}")

        pbar.set_postfix({
            "acc": f"{(total_correct/processed_samples)*100:.1f}" if processed_samples else "0%",
            "errors": total_errors
        })

    pbar.close()
    return total_correct / processed_samples if processed_samples else 0

def format_prompt_qa(row, include_answer=True):
    try:
        options = literal_eval(row['options'])
        num_choices = len(options)
        choice_numbers = [str(i+1) for i in range(num_choices)]
        formatted_options = '\n'.join([f"{num}. {opt}" for num, opt in zip(choice_numbers, options)])
        
        sys_msg = cot_sys_prompt()
        prompt = f"""<|im_start|>system
        {sys_msg}<|im_end|>
        <|im_start|>user
        Question: {row['question']}
        Options:
        {formatted_options}
        Please analyze and provide the answer with only one number in [[number]].<|im_end|>
        <|im_start|>assistant"""

        if include_answer:
            # Добавляем полный ответ с CoT из distill_response
            prompt += f"\n{ANSWER_MARKER[0]}{str(int(row['distill_answer']))}{ANSWER_MARKER[1]}<|im_end|>"
        
        return prompt
        
    except Exception as e:
        logging.error(f"Error formatting prompt: {e}")
        return None

        

    except (SyntaxError, ValueError, KeyError, TypeError) as e:
        # Логирование ошибки при необходимости
        return None

def prepare_datasets(train_file, val_file, test_file, tokenizer):
    np.random.seed(42)
    torch.manual_seed(42)

    def load_and_filter_df(file_path):
        df = pd.read_csv(file_path, sep="\t")
        df = df[df['distill_answer'].notna()]
        #df = preprocess_options(df)
        formatted = df.apply(format_prompt_qa, axis=1)
        valid_mask = formatted.notnull()
        print(f"Пропущено вопросов при подготовке ({os.path.basename(file_path)}): {len(df)-valid_mask.sum()}")
        return df[valid_mask], formatted[valid_mask]

    # Загрузка и фильтрация данных

    train_df, train_formatted = load_and_filter_df(train_file)
    val_df, val_formatted = load_and_filter_df(val_file)
    test_df, test_formatted = load_and_filter_df(test_file)

    def create_dataset(formatted_series, original_df):
        dataset = Dataset.from_pandas(pd.DataFrame({'text': formatted_series}), preserve_index=False)
        return dataset.map(

            lambda ex: tokenizer(
                ex["text"],
                truncation=True,
                max_length=800,
                padding=False,
                return_attention_mask=True,
                add_special_tokens=False
            ),
            remove_columns=["text"],
            num_proc=4
        ), original_df

    train_dataset, train_df = create_dataset(train_formatted, train_df)
    val_dataset, val_df = create_dataset(val_formatted, val_df)
    test_dataset, test_df = create_dataset(test_formatted, test_df)

    return train_dataset, val_dataset, test_dataset, train_df, val_df, test_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="Qwen2.5-3B-Instruct")
    parser.add_argument("--train_path", default="easy_window_train.tsv")
    parser.add_argument("--valid_path", default="easy_window_valid.tsv")
    parser.add_argument("--test_path", default="combined_window_test.tsv")
    parser.add_argument("--test_balanced_path", default="combined_window_test.tsv")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_dir", default="/mnt/data202/PERSONAL/VYAZHEV/PROJECT_1/full_reasoning_split/reasoning_easy_learning/")

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    log_file = os.path.join(args.save_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w"),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logging.info("=== Environment Info ===")
    logging.info(f"Python: {platform.python_version()}")
    logging.info(f"PyTorch: {torch.__version__}")
    logging.info("Starting training...")

    # Инициализация модели

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        use_cache=False
    )

    model.gradient_checkpointing_enable()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Подготовка данных

    train_dataset, val_dataset, test_dataset, train_df, val_df, test_df = prepare_datasets(
        args.train_path,
        args.valid_path,
        args.test_path,
        tokenizer
    )
    
    test_balanced_df = pd.read_csv(args.test_balanced_path, sep="\t")

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=10
    )

    optimizer = AdamW(model.parameters(), lr=args.lr, fused=True)

    scaler = GradScaler(enabled=False)

    # Обучение
    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0.0
        total_train_acc = 0.0
        total_samples = 0

        pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch_idx, batch in enumerate(train_loader):
            batch = {k: v.cuda() for k, v in batch.items()}
            
            if batch_idx == 0:
                sample = tokenizer.decode(batch['input_ids'][0], skip_special_tokens=True)
                logging.info(f"\nTraining Sample:\n{sample}")
            
            with autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(**batch)
                loss = outputs.loss / args.gradient_accumulation
                accuracy = calculate_accuracy(outputs.logits, batch['input_ids'], tokenizer)

            scaler.scale(loss).backward()

            if (batch_idx + 1) % args.gradient_accumulation == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            batch_size = batch['input_ids'].size(0)
            total_train_loss += loss.item() * batch_size
            total_train_acc += accuracy.item() * batch_size
            total_samples += batch_size

            pbar.update(1)

            pbar.set_postfix({
                "loss": f"{(total_train_loss / total_samples) * args.gradient_accumulation:.3f}",
                "acc": f"{(total_train_acc / total_samples) * 100:.2f}%",
                "lr": f"{optimizer.param_groups[0]['lr']:.1e}"
            })

            if (batch_idx + 1) % args.log_interval == 0:
                logging.info(f"Epoch {epoch+1}, Batch {batch_idx+1}: Loss = {(total_train_loss / total_samples)* args.gradient_accumulation:.3f}, Acc = {(total_train_acc / total_samples) * 100:.2f}%")
                total_train_loss = 0.0
                total_train_acc = 0.0
                total_samples = 0

        pbar.close()
        logging.info(f"Epoch {epoch+1} completed.")

        # Валидация
        val_metrics = evaluate_qa(model, val_df, tokenizer, desc='Validation QA')
        logging.info(f"Validation QA Accuracy: {val_metrics * 100:.2f}%")
        
       # test_metrics = evaluate_qa(model, test_df, tokenizer, desc='TEST QA')
       # logging.info(f"TEST QA Accuracy: {test_metrics * 100:.2f}%")
        
        test_metrics = evaluate_qa(model, test_balanced_df, tokenizer, desc='TEST BALANCED QA')
        logging.info(f"TEST BALANCED QA Accuracy: {test_metrics * 100:.2f}%")
       

    epoch_dir = os.path.join(args.save_dir, f"epoch_{epoch+1}")
    os.makedirs(epoch_dir, exist_ok=True)
    model.save_pretrained(epoch_dir)
    tokenizer.save_pretrained(epoch_dir)
    logging.info(f"Model saved to {epoch_dir}")     
    
if __name__ == "__main__":
    main()

