from datasets import Dataset
import pandas as pd

from core.prompts.mmlu_option_ids import option_ids
from core.prompts import mmlu_cot_answer as cot_prompts
from core.prompts.thinking_markers import THINKING_START, THINKING_END


def prepare_dataset(
    tokenizer,
    get_sys_prompt,
    get_user_prompt,
    df,
    mask_input=False,
    use_thinking=False,
    use_cot=False,
):
    """
    Prepare a training/eval dataset.
    Answer wrapping modes:
      - default:        single answer letter token, e.g. "a"
      - use_cot=True:   answer wrapped in markers, e.g. "[[a]]"
      - use_thinking=True: <think>reasoning</think>a
    """
    assert not (use_thinking and use_cot), "use_thinking and use_cot are mutually exclusive"

    df = df.copy()
    df["sys_prompt"] = df.apply(get_sys_prompt, axis=1)
    df["user_prompt"] = df.apply(get_user_prompt, axis=1)

    answer_marker = cot_prompts.answer_marker

    def process_row(row):
        tokenized = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": row["sys_prompt"]},
                {"role": "user", "content": row["user_prompt"]},
            ],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        )

        answer = str(row["answer"]).strip().lower()

        if use_thinking:
            # <think>reasoning</think> + bare letter
            target_parts = []
            thinking_val = row.get("thinking", None)
            if pd.notna(thinking_val):
                thinking_str = str(thinking_val).strip()
                if thinking_str:
                    target_parts.append(f"{THINKING_START}{thinking_str}{THINKING_END}")
            target_parts.append(answer)
            target_text = " ".join(target_parts)
            target_tokens = tokenizer.encode(target_text, add_special_tokens=False)

        elif use_cot:
            # answer wrapped in [[...]] markers
            target_text = f"{answer_marker[0]}{answer}{answer_marker[1]}"
            target_tokens = tokenizer.encode(target_text, add_special_tokens=False)

        else:
            # single token - bare letter
            target_tokens = tokenizer.encode(answer, add_special_tokens=False)[:1]

        input_ids = tokenized["input_ids"] + target_tokens
        attention_mask = tokenized["attention_mask"] + [1] * len(target_tokens)

        labels = input_ids.copy()

        if mask_input:
            for i in range(len(tokenized["input_ids"])):
                labels[i] = -100

        question_id = row["question_id"]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "question_id": question_id,
        }

    dataset = Dataset.from_pandas(df)

    processed_ds = dataset.map(
        process_row,
        num_proc=4,
        remove_columns=dataset.column_names,
    )

    return processed_ds


def prepare_dataset_cot_eval(
    tokenizer,
    get_sys_prompt,
    get_user_prompt,
    df,
    use_thinking=False,
):
    """
    Prepare eval dataset for generative (CoT) evaluation.
    Only input is tokenized; label is a single token for answer matching.
    """
    df = df.copy()
    df["sys_prompt"] = df.apply(get_sys_prompt, axis=1)
    df["user_prompt"] = df.apply(get_user_prompt, axis=1)

    def process_row(row):
        tokenized = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": row["sys_prompt"]},
                {"role": "user", "content": row["user_prompt"]},
            ],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        )

        answer = str(row["answer"]).strip().lower()
        label_token_id = tokenizer.encode(answer, add_special_tokens=False)[0]

        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        labels = [label_token_id]

        question_id = row["question_id"]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "cot": True,
            "question_id": question_id,
        }

    dataset = Dataset.from_pandas(df)

    processed_ds = dataset.map(
        process_row,
        num_proc=4,
        remove_columns=dataset.column_names,
    )

    return processed_ds
