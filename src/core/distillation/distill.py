import os
from concurrent import futures

import pandas as pd
from tqdm import tqdm

from core.prompts.mmlu_single_token_answer import single_token_answer_prompt, single_token_sys_prompt
from core.utils.chunker import chunker
from core.utils.openrouter import openrouter

chunk_size = 30


def call_remote_llm(args):
    try:
        sys_prompt, user_prompt, index, model, max_tokens = args

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ]

        completion = openrouter.chat.completions.create(
            model="deepseek/deepseek-v4-flash",
            messages=messages,
            extra_body={
                "reasoning": {"enabled": True, "max_tokens": 8192},
                "provider": {"order": ["novita"], "allow_fallbacks": False},
            },
        )

        msg = completion.choices[0].message

        # `reasoning` and `reasoning_details` are NOT in the OpenAI pydantic schema.
        # The SDK keeps unknown fields accessible via attribute access OR model_extra.
        # getattr is the safe path:
        content = msg.content
        reasoning_details = getattr(msg, "reasoning_details", None)  # list[dict] or None

        # If your openai-python is old enough to strip extras, fall back:
        if reasoning_details is None:
            extra = getattr(msg, "model_extra", {}) or {}
            reasoning_details = extra.get("reasoning_details")

        return index, content, reasoning_details["text"]
    except:
        return None


def distill_on_dataset(
    in_filename,
    out_filename,
    get_subject_from_row,
    get_question_from_row,
    get_options_from_row,
    check_answer_correct,
    dump_every=100,
    max_tokens=8192,
    model="deepseek/deepseek-v4-flash",
    get_sys_prompt=single_token_sys_prompt,
    get_user_prompt=single_token_answer_prompt,
):
    invalid_answers = 0

    field_reasoning = "distill_reasoning"
    field_ans = "distill_answer"
    field_ans_correct = "distill_ans_correct"

    if os.path.exists(out_filename):
        df = pd.read_parquet(out_filename)
    else:
        df = pd.read_parquet(in_filename)

    # print(df.dtypes)

    if field_ans_correct not in df.columns:
        df[field_ans_correct] = False
    if field_reasoning not in df.columns:
        df[field_reasoning] = ""
    if field_ans not in df.columns:
        df[field_ans] = ""

    with futures.ThreadPoolExecutor(max_workers=chunk_size) as pool:
        for chunk_idx, chunk in tqdm(enumerate(chunker(df, chunk_size)), total=int(df.shape[0] / chunk_size)):
            args_list = []

            for index, row in chunk.iterrows():
                if df.at[index, field_reasoning] != "":
                    continue

                sys_prompt = get_sys_prompt(get_subject_from_row(row))
                user_prompt = get_user_prompt(get_question_from_row(row), get_options_from_row(row))
                args_list.append((sys_prompt, user_prompt, index, model, max_tokens))

            results = list(pool.map(call_remote_llm, args_list))

            for result in results:
                if result is None:
                    invalid_answers += 1
                    continue

                index, model_answer, model_reasoning = result

                df.at[index, field_ans] = model_answer
                df.at[index, field_reasoning] = model_reasoning
                df.at[index, field_ans_correct] = check_answer_correct(df.iloc[index], model_answer)

                print(
                    f"response: {model_reasoning}\nextracted_answer: {model_answer}\ncorrect:{df.at[index, field_ans_correct]}\n\n"
                )

            if chunk_idx % dump_every == 0:
                df.to_parquet(out_filename, index=False)

    df.to_parquet(out_filename, index=False)
    print(f"Processed dataset {out_filename}. Total entries: {df.shape[0]}. Invalid answers: {invalid_answers}")
    return df
