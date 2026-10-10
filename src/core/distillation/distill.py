import os
import queue
import threading
import time
from pathlib import Path

import pandas as pd
from pydraconf import PydraConfig
from tqdm import tqdm

from core.datasets.qa_dataset import QADataset, QADatasetConfig
from core.utils.openrouter import openrouter

pool_size = 30
REQUEST_BUDGET_S = 180.0


def _worker(input_q: queue.Queue, output_q: queue.Queue):
    while True:
        item = input_q.get()
        if item is None:
            input_q.task_done()
            break
        try:
            result = call_remote_llm(item)
        except Exception as e:
            print(f"Worker error: {e}")
            result = None
        output_q.put(result)
        input_q.task_done()


def call_remote_llm(args):
    try:
        sys_prompt, user_prompt, index, model, max_tokens, timeout = args

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ]

        t0 = time.monotonic()
        stream = openrouter.chat.completions.create(  # pyright: ignore[reportCallIssue]
            model=model,
            messages=messages,
            stream=True,
            extra_body={
                "reasoning": {"enabled": True, "max_tokens": max_tokens},
            },
        )

        content_parts = []
        reasoning_parts = []
        for chunk in stream:
            if time.monotonic() - t0 > timeout:
                stream.close()
                raise TimeoutError(f"Exceeded {timeout}s wall-clock budget for index {index}")
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

        return DistillationResult(index=index, answer=content, reasoning=reasoning)
    except Exception as e:
        print(f"Error occurred: {e}")
        return None


class DistillationConfig(PydraConfig):
    out_filename: str
    model: str
    dataset: QADataset[QADatasetConfig]
    dump_every: int = 100
    max_tokens: int = 8192
    timeout: float = REQUEST_BUDGET_S
    regenerate_incorrect: bool = False
    field_reasoning: str = "distill_reasoning"
    field_ans: str = "distill_answer"
    field_ans_correct: str = "distill_ans_correct"


class DistillationResult(PydraConfig):
    index: int
    answer: str
    reasoning: str | None


class DistillationResultWriter:
    def write_to_df(self, df: pd.DataFrame, config: DistillationConfig, result: DistillationResult):
        df.at[result.index, config.field_ans] = result.answer
        df.at[result.index, config.field_reasoning] = result.reasoning
        df.at[result.index, config.field_ans_correct] = config.dataset.verify_assistant_response(
            df.iloc[result.index].to_dict(), result.answer
        )[1]


def distill_on_dataset(
    config: DistillationConfig,
    distillation_result_writer: DistillationResultWriter = DistillationResultWriter(),
):
    invalid_answers = 0
    cnt = 0

    tmp_path = Path(config.out_filename).with_suffix(".tmp.parquet")

    if os.path.exists(tmp_path):
        df = pd.read_parquet(tmp_path)
    elif os.path.exists(config.out_filename):
        df = pd.read_parquet(config.out_filename)
    else:
        df = pd.read_parquet(config.dataset.processed_path)

    if config.field_ans_correct not in df.columns:
        df[config.field_ans_correct] = False
    if config.field_reasoning not in df.columns:
        df[config.field_reasoning] = ""
    if config.field_ans not in df.columns:
        df[config.field_ans] = ""

    input_q: queue.Queue = queue.Queue()
    output_q: queue.Queue = queue.Queue()

    expected = 0
    for index, row in df.iterrows():
        row_dict = row.to_dict()

        if not config.regenerate_incorrect and row_dict[config.field_reasoning] != "":
            continue

        if config.regenerate_incorrect and row_dict[config.field_ans_correct]:
            continue

        sys_prompt = config.dataset.system_prompt(row_dict)
        user_prompt = config.dataset.user_prompt(row_dict)
        input_q.put((sys_prompt, user_prompt, index, config.model, config.max_tokens, config.timeout))
        expected += 1

    for _ in range(pool_size):
        input_q.put(None)

    threads = [threading.Thread(target=_worker, args=(input_q, output_q), daemon=True) for _ in range(pool_size)]
    for t in threads:
        t.start()

    with tqdm(total=expected) as pbar:
        for _ in range(expected):
            result = output_q.get()
            pbar.update(1)

            if result is None:
                invalid_answers += 1
                continue

            cnt += 1
            distillation_result_writer.write_to_df(df, config, result)

            if cnt < 2:
                print(
                    f"response: {df.at[result.index, config.field_reasoning]}\nextracted_answer: {df.at[result.index, config.field_ans]}\ncorrect:{df.at[result.index, config.field_ans_correct]}\n\n"
                )

            if cnt % config.dump_every == 0:
                df.to_parquet(tmp_path, compression=None, index=False)

    for t in threads:
        t.join()

    df.to_parquet(config.out_filename, index=False)
    if os.path.exists(tmp_path):
        os.unlink(tmp_path)

    print(f"Processed dataset {config.out_filename}. Total entries: {df.shape[0]}. Invalid answers: {invalid_answers}")
    return df
