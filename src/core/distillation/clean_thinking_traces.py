"""LLM cleaner for thinking traces on H100 80GB."""

import os
os.environ["VLLM_USE_V1"] = "0"

import pandas as pd
from pathlib import Path
from vllm import LLM, SamplingParams
from tqdm import tqdm
import torch
import gc
import time

from prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE


def format_prompt(reasoning_trace: str) -> list:
    """Format prompt for LLM cleaning."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(reasoning_trace=reasoning_trace)}
    ]


def truncate_thinking(thinking: str, max_chars: int = 50000) -> str:
    if len(thinking) <= max_chars:
        return thinking
    truncated = thinking[:max_chars]
    last_period = truncated.rfind('.')
    if last_period > max_chars - 200:
        return truncated[:last_period + 1]
    return truncated


def clean_thinking_traces(
    input_file: str | Path,
    output_file: str | Path,
    model_name: str = "Qwen/Qwen2.5-32B-Instruct",
    batch_size: int = 24,
    temperature: float = 0.0,
    gpu_memory_utilization: float = 0.95,
    max_model_len: int = 20480,
    max_thinking_chars: int = 35000,
    checkpoint_every_n_batches: int = 8
) -> Path:
    gc.collect()
    torch.cuda.empty_cache()

    input_path = Path(input_file).resolve()
    output_path = Path(output_file).resolve()
    
    checkpoint_name = f"checkpoint_{int(time.time())}_{os.getpid()}.parquet"
    checkpoint_path = output_path.parent / checkpoint_name
    
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    
    llm = LLM(
        model=model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=max_model_len,
        enforce_eager=False
    )
    
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=1.0,
        max_tokens=int(max_model_len * 0.9),
        repetition_penalty=1.05
    )
    
    df = pd.read_parquet(input_path)
    print(f"Records: {len(df)}")
    
    reasoning_traces = []
    indices_to_clean = []
    
    for idx, row in df.iterrows():
        out = row['output']
        if isinstance(out, dict) and out.get('error') is None and 'thinking' in out:
            thinking = out.get('thinking', '')
            if thinking:
                thinking = truncate_thinking(thinking, max_thinking_chars)
                reasoning_traces.append(thinking)
                indices_to_clean.append(idx)
    
    print(f"Traces to clean: {len(reasoning_traces)}")
    
    all_prompts = [format_prompt(trace) for trace in reasoning_traces]
    df_cleaned = df.copy()
    
    pbar = tqdm(total=len(all_prompts), desc="Cleaning")
    
    for i in range(0, len(all_prompts), batch_size):
        batch_prompts = all_prompts[i : i + batch_size]
        batch_indices = indices_to_clean[i : i + batch_size]
        
        outputs = llm.chat(messages=batch_prompts, sampling_params=sampling_params, use_tqdm=False)
        batch_texts = [output.outputs[0].text.strip() for output in outputs]
        
        for idx, cleaned_text in zip(batch_indices, batch_texts):
            output_dict = df_cleaned.at[idx, 'output'].copy()
            output_dict['thinking'] = cleaned_text
            df_cleaned.at[idx, 'output'] = output_dict
        
        pbar.update(len(batch_texts))
        
        if (i // batch_size + 1) % checkpoint_every_n_batches == 0:
            df_cleaned.to_parquet(checkpoint_path, index=False)
            
    pbar.close()
    df_cleaned.to_parquet(output_path, index=False)
    
    if checkpoint_path.exists():
        os.remove(checkpoint_path)
    
    print(f"Saved: {output_path}")
    return output_path
