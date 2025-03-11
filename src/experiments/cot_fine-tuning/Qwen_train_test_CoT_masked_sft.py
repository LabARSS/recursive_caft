import argparse
import logging
import os
import platform
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from ast import literal_eval

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

answer_marker = ("[[", "]]")


def cot_sys_prompt(subject: str | None = None):
    if subject is not None:
        sys_msg = f"The following are multiple choice questions about {subject}."
    else:
        sys_msg = "The following are multiple choice questions."

    sys_msg += f" Explain your thinking process step-by-step. At the end, write the correct answer letter by strictly following this format: {answer_marker[0]}correct_answer_letter{answer_marker[1]}."
    return sys_msg


def calculate_accuracy(logits, labels, tokenizer):
    predictions = torch.argmax(logits, dim=-1)
    shifted_labels = labels[..., 1:].contiguous()
    shifted_predictions = predictions[..., :-1].contiguous()
    mask = shifted_labels != -100
    mask = mask & (shifted_labels != tokenizer.pad_token_id)
    correct = (shifted_predictions == shifted_labels) & mask
    return correct.float().sum() / mask.float().sum()


def print_masking_example(dataset, tokenizer, desc="Пример маскирования"):
    print(f"\n=== {desc} ===")
    example = dataset[0]
    input_ids = example["input_ids"]
    labels = example["labels"]

    print("\nИсходный текст:")
    print(tokenizer.decode(input_ids, skip_special_tokens=False))

    print("\nТокены после маскирования:")
    masked = [
        f"{token}({tokenizer.decode(t)})" if l == -100 else f"[{token}({tokenizer.decode(t)})]"
        for token, (t, l) in enumerate(zip(input_ids, labels))
    ]
    print(" ".join(masked))


def evaluate(model, loader, scaler, tokenizer):
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    total_samples = 0

    pbar = tqdm(total=len(loader), desc="Validating", leave=False)

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            batch = {k: v.cuda() for k, v in batch.items()}
            with autocast(device_type="cuda:2", dtype=torch.bfloat16):
                outputs = model(**batch)
                loss = outputs.loss
                accuracy = calculate_accuracy(outputs.logits, batch["input_ids"], tokenizer)

            batch_size = batch["input_ids"].size(0)
            total_loss += loss.item() * batch_size
            total_accuracy += accuracy.item() * batch_size
            total_samples += batch_size
            pbar.update(1)

            pbar.set_postfix(
                {"loss": f"{total_loss / total_samples:.3f}", "acc": f"{(total_accuracy / total_samples) * 100:.1f}%"}
            )

    pbar.close()

    return {"loss": total_loss / total_samples, "accuracy": total_accuracy / total_samples}


def evaluate_qa(model, df, tokenizer, desc="Validating"):
    model.eval()
    total_correct = 0
    total_errors = 0
    processed_samples = 0
    debug_example_printed = False

    pbar = tqdm(total=len(df), desc=desc, leave=False)

    for idx in range(len(df)):
        row = df.iloc[idx]

        try:
            options = literal_eval(row["options"])
            num_choices = len(options)
            choice_letters = [chr(65 + i) for i in range(num_choices)]
            prompt = format_prompt_qa(row, include_answer=False)
            # Перемещаем блок отладки ВНУТРЬ try
            if idx == 0 and not debug_example_printed:
                print("\n=== Пример обработки вопроса ===")
                print("Исходный вопрос:")
                print(f"Вопрос: {row['question']}")
                print(f"Варианты: {options}")
                print(f"Правильный ответ: {row['answer']}")

            if not prompt:
                total_errors += 1
                continue

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768).to(model.device)

            # Генерируем полный ответ с объяснением
            output = model.generate(
                inputs.input_ids,
                max_new_tokens=1000,
                pad_token_id=tokenizer.eos_token_id,
                attention_mask=inputs.attention_mask,
                do_sample=False,
            )

            # Декодируем и извлекаем ответ
            full_response = tokenizer.decode(output[0], skip_special_tokens=False)

            # Вывод полного ответа для первого примера
            if idx == 0 and not debug_example_printed:
                print("\nПолный ответ модели:")
                print(full_response)
                debug_example_printed = True

            # Ищем паттерн [[ LETTER ]]
            start_idx = full_response.rfind(answer_marker[0])
            pred = None
            if start_idx != -1:
                end_idx = full_response.find(answer_marker[1], start_idx)
                if end_idx != -1:
                    answer = full_response[start_idx + len(answer_marker[0]) : end_idx].strip().upper()
                    pred = answer[0] if answer else None

            # Валидация предсказания
            if pred and pred in choice_letters:
                correct = pred == row["answer"].strip().upper()[0]
                total_correct += correct
                processed_samples += 1
            else:
                total_errors += 1

        except Exception:
            total_errors += 1

        pbar.update(1)
        if processed_samples > 0:
            pbar.set_postfix({"acc": f"{(total_correct / processed_samples) * 100:.1f}%", "errors": total_errors})

    pbar.close()
    return total_correct / processed_samples if processed_samples else 0.0


def format_prompt(row):
    error_count = [0]
    try:
        options = "\n".join([f"{chr(65 + i)}. {opt}" for i, opt in enumerate(literal_eval(row["options"]))])
        prompt = f"""<|im_start|>system Analyze the question and select the correct answer.
                Answer should be a single uppercase letter.<|im_end|>
                <|im_start|>user
                Question:{row["question"]}
                Options:{options}
                Answer:<|im_end|>
                <|im_start|>assistant
                {row["answer"]}<|im_end|>"""
    except (SyntaxError, ValueError, KeyError):
        error_count[0] += 1
        return None


def format_prompt_qa(row, include_answer=True):
    try:
        options = literal_eval(row["options"])
        num_choices = len(options)
        choice_letters = [chr(65 + i) for i in range(num_choices)]
        formatted_options = "\n".join([f"{letter}. {opt}" for letter, opt in zip(choice_letters, options)])

        sys_msg = cot_sys_prompt(subject=None)
        prompt = f"""<|im_start|>system {sys_msg}<|im_end|>
        <|im_start|>user
        Question: {row["question"]}
        Options: {formatted_options}
        Answer: <|im_end|>
        <|im_start|>assistant"""

        if include_answer:
            # Добавляем ответ в формате [[A]]
            answer_letter = row["answer"].strip().upper()[0]  # Берем первую букву
            prompt += f" {answer_marker[0]}{answer_letter}{answer_marker[1]}<|im_end|>"

        return prompt
    except Exception as e:
        print(f"Error formatting prompt: {str(e)}")
        return None


def find_subsequence(sequence, subsequence):
    """Ищет подпоследовательность в последовательности токенов"""
    n = len(sequence)
    m = len(subsequence)
    for i in range(n - m + 1):
        if sequence[i : i + m] == subsequence:
            return i
    return -1


def prepare_datasets(train_file, val_file, test_file, tokenizer):
    np.random.seed(42)
    torch.manual_seed(42)

    def load_and_filter_df(file_path):
        try:
            df = pd.read_csv(file_path, sep="\t")
            formatted = df.apply(format_prompt_qa, axis=1)
            valid_mask = formatted.notnull()
            print(f"Пропущено вопросов при подготовке ({os.path.basename(file_path)}): {len(df) - valid_mask.sum()}")
            return df[valid_mask], formatted[valid_mask]
        except Exception as e:
            print(f"Ошибка загрузки {file_path}: {str(e)}")
            return pd.DataFrame(), pd.Series()

    def create_dataset(formatted_series, original_df):
        try:
            if len(formatted_series) == 0:
                raise ValueError("Пустой набор данных")

            dataset = Dataset.from_pandas(pd.DataFrame({"text": formatted_series}), preserve_index=False)

            # Токенизация с паддингом
            dataset = dataset.map(
                lambda ex: tokenizer(
                    ex["text"],
                    truncation=True,
                    max_length=768,
                    padding="max_length",  # Добавлен паддинг
                    return_attention_mask=True,
                    add_special_tokens=False,
                ),
                remove_columns=["text"],
                num_proc=4,
            )

            # Обработка меток
            start_marker_ids = tokenizer.encode(answer_marker[0], add_special_tokens=False)
            end_marker_ids = tokenizer.encode(answer_marker[1], add_special_tokens=False)

            def process_labels(examples):
                new_labels = []
                for input_ids, attention_mask in zip(examples["input_ids"], examples["attention_mask"]):
                    # сначала возьмём все токены как цель
                    labels = input_ids.copy()

                    # затем замаскируем всё, что до ассистента (системные + вопрос + опции + Answer:)
                    # найдём начало <|im_start|>assistant
                    assistant_tok = tokenizer.encode("<|im_start|>assistant", add_special_tokens=False)
                    seq_len = sum(attention_mask)
                    st = find_subsequence(input_ids[:seq_len], assistant_tok)
                    if st != -1:
                        # маскируем всё до и включая сам маркер
                        for i in range(st + len(assistant_tok)):
                            labels[i] = -100

                    # замаскируем паддинги
                    for i, m in enumerate(attention_mask):
                        if m == 0:
                            labels[i] = -100

                    new_labels.append(labels)
                return {"labels": new_labels}

            dataset = dataset.map(process_labels, batched=True, batch_size=1000)
            return dataset, original_df

        except Exception as e:
            print(f"Ошибка создания датасета: {str(e)}")
            return None, None

    # Основная логика подготовки данных
    try:
        # Загрузка и фильтрация данных
        train_df, train_formatted = load_and_filter_df(train_file)
        val_df, val_formatted = load_and_filter_df(val_file)
        test_df, test_formatted = load_and_filter_df(test_file)

        # Создание датасетов
        train_dataset, train_df = create_dataset(train_formatted, train_df)
        val_dataset, val_df = create_dataset(val_formatted, val_df)
        test_dataset, test_df = create_dataset(test_formatted, test_df)

        # Проверка результатов
        if None in [train_dataset, val_dataset, test_dataset]:
            raise RuntimeError("Ошибка создания одного из датасетов")

            # После создания датасетов
        print_masking_example(train_dataset, tokenizer, "After process label")
        # print_masking_example(val_dataset, tokenizer, "Валидационный пример")
        # print_masking_example(test_dataset, tokenizer, "Тестовый пример")

        return (train_dataset, val_dataset, test_dataset, train_df, val_df, test_df)

    except Exception as e:
        print(f"Критическая ошибка подготовки данных: {str(e)}")
        raise


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
    parser.add_argument(
        "--save_dir", default="/mnt/data202/PERSONAL/VYAZHEV/PROJECT_1/full_reasoning_split/reasoning_easy_learning/"
    )

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    log_file = os.path.join(args.save_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file, mode="w"), logging.StreamHandler(sys.stdout)],
    )

    logging.info("=== Environment Info ===")
    logging.info(f"Python: {platform.python_version()}")
    logging.info(f"PyTorch: {torch.__version__}")
    logging.info("Starting training...")

    # Инициализация модели

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True, use_cache=False
    )

    model.gradient_checkpointing_enable()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    train_dataset, val_dataset, test_dataset, train_df, val_df, test_df = prepare_datasets(
        args.train_path, args.valid_path, args.test_path, tokenizer
    )

    test_balanced_df = pd.read_csv(args.test_balanced_path, sep="\t")

    #    data_collator = DataCollatorForLanguageModeling(
    #        tokenizer=tokenizer,
    #        mlm=False,
    #        pad_to_multiple_of=8
    #    )
    def causal_collate(batch):
        input_ids = torch.stack([torch.tensor(ex["input_ids"]) for ex in batch])
        attention_mask = torch.stack([torch.tensor(ex["attention_mask"]) for ex in batch])
        labels = torch.stack([torch.tensor(ex["labels"]) for ex in batch])
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=causal_collate, num_workers=4
    )

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=causal_collate, num_workers=4)

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=causal_collate, num_workers=4)

    optimizer = AdamW(model.parameters(), lr=args.lr, fused=True)

    scaler = GradScaler(enabled=False)

    # Обучение
    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0.0
        total_train_acc = 0.0
        total_samples = 0

        pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{args.epochs}")
        for batch_idx, batch in enumerate(train_loader):
            batch = {k: v.cuda() for k, v in batch.items()}

            with autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(**batch)
                loss = outputs.loss / args.gradient_accumulation
                accuracy = calculate_accuracy(outputs.logits, batch["input_ids"], tokenizer)

            scaler.scale(loss).backward()

            if (batch_idx + 1) % args.gradient_accumulation == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            batch_size = batch["input_ids"].size(0)
            total_train_loss += loss.item() * batch_size
            total_train_acc += accuracy.item() * batch_size
            total_samples += batch_size

            pbar.update(1)

            pbar.set_postfix(
                {
                    "loss": f"{(total_train_loss / total_samples) * args.gradient_accumulation:.3f}",
                    "acc": f"{(total_train_acc / total_samples) * 100:.2f}%",
                    "lr": f"{optimizer.param_groups[0]['lr']:.1e}",
                }
            )

            if (batch_idx + 1) % args.log_interval == 0:
                logging.info(
                    f"Epoch {epoch + 1}, Batch {batch_idx + 1}: Loss = {(total_train_loss / total_samples) * args.gradient_accumulation:.3f}, Acc = {(total_train_acc / total_samples) * 100:.2f}%"
                )
                total_train_loss = 0.0
                total_train_acc = 0.0
                total_samples = 0

        pbar.close()
        logging.info(f"Epoch {epoch + 1} completed.")

        # Валидация
        val_metrics = evaluate_qa(model, val_df, tokenizer, desc="Validation QA")
        logging.info(f"Validation QA Accuracy: {val_metrics * 100:.2f}%")

        test_metrics = evaluate_qa(model, test_df, tokenizer, desc="TEST QA")
        logging.info(f"TEST QA Accuracy: {test_metrics * 100:.2f}%")

        test_metrics = evaluate_qa(model, test_balanced_df, tokenizer, desc="TEST BALANCED QA")
        logging.info(f"TEST BALANCED QA Accuracy: {test_metrics * 100:.2f}%")


if __name__ == "__main__":
    main()
