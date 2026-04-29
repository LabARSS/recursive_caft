import os
import time
from concurrent import futures

import pandas as pd
from tqdm import tqdm

from core.prompts.mmlu_single_token_answer import single_token_answer_prompt, single_token_sys_prompt
from core.utils.chunker import chunker
from core.utils.openrouter import openrouter

chunk_size = 16
REQUEST_BUDGET_S = 180.0


def call_remote_llm(args):
    try:
        sys_prompt, user_prompt, index, model, max_tokens = args

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ]

        t0 = time.monotonic()
        stream = openrouter.chat.completions.create(  # pyright: ignore[reportCallIssue]
            model="deepseek/deepseek-v4-flash",
            messages=messages,
            stream=True,
            extra_body={
                "reasoning": {"enabled": True, "max_tokens": 8192},
                "provider": {"order": ["deepseek"], "allow_fallbacks": False},
            },
        )

        content_parts = []
        reasoning_parts = []
        for chunk in stream:
            if time.monotonic() - t0 > REQUEST_BUDGET_S:
                stream.close()
                raise TimeoutError(f"Exceeded {REQUEST_BUDGET_S}s wall-clock budget for index {index}")
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                content_parts.append(delta.content)
            reasoning_piece = getattr(delta, "reasoning", None)
            if reasoning_piece:
                reasoning_parts.append(reasoning_piece)

        content = "".join(content_parts)
        reasoning = "".join(reasoning_parts) if reasoning_parts else None

        return index, content, reasoning
    except Exception as e:
        print(f"Error occurred: {e}")
        return None


def distill_on_dataset(
    in_filename,
    out_filename,
    get_subject_from_row,
    get_question_from_row,
    get_options_from_row,
    check_answer_correct,
    dump_every=50,
    max_tokens=8192,
    model="deepseek/deepseek-v4-flash",
    get_sys_prompt=single_token_sys_prompt,
    get_user_prompt=single_token_answer_prompt,
):
    invalid_answers = 0
    cnt = 0

    field_reasoning = "distill_reasoning"
    field_ans = "distill_answer"
    field_ans_correct = "distill_ans_correct"

    if os.path.exists(out_filename):
        df = pd.read_parquet(out_filename)
    else:
        df = pd.read_parquet(in_filename)

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

            if len(args_list) == 0:
                continue

            results = list(pool.map(call_remote_llm, args_list))

            for result in results:
                if result is None:
                    invalid_answers += 1
                    continue

                cnt += 1

                index, model_answer, model_reasoning = result

                df.at[index, field_ans] = model_answer
                df.at[index, field_reasoning] = model_reasoning
                df.at[index, field_ans_correct] = check_answer_correct(df.iloc[index], model_answer)

                if cnt < 5:
                    print(
                        f"response: {model_reasoning}\nextracted_answer: {model_answer}\ncorrect:{df.at[index, field_ans_correct]}\n\n"
                    )

            if chunk_idx % dump_every == 0:
                df.to_parquet(out_filename, compression=None, index=False)

    df.to_parquet(out_filename, index=False)
    print(f"Processed dataset {out_filename}. Total entries: {df.shape[0]}. Invalid answers: {invalid_answers}")
    return df
