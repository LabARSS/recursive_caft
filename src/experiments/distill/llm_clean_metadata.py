import logging
import pandas as pd
from pathlib import Path
from vllm import LLM, SamplingParams
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

SYSTEM_PROMPT = """You are a text cleaning assistant. Your task is to remove metadata and instructional phrases from reasoning traces while preserving the actual reasoning content.

Remove phrases like:
- "We need to explain why..."
- "Provide concise/specific/brief explanation"
- "Let's craft/produce/compute..."
- "Now produce final answer"
- "Explain why other options are wrong"
- Any other meta-instructions about what to explain or how to format

Keep:
- Actual reasoning steps
- Mathematical calculations
- Logical arguments
- Domain-specific explanations
- Conclusions and answers

Return ONLY the cleaned reasoning trace, nothing else."""

USER_PROMPT_TEMPLATE = """Clean this reasoning trace by removing all metadata and instructional phrases:

{reasoning_trace}

Cleaned reasoning:"""

def format_prompt(reasoning_trace: str) -> list:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(reasoning_trace=reasoning_trace)}
    ]

def clean_reasoning_batch(llm: LLM, reasoning_traces: list[str], sampling_params: SamplingParams, batch_size: int = 32):
    all_prompts = [format_prompt(trace) for trace in reasoning_traces]
    cleaned_results = []
    
    for i in tqdm(range(0, len(all_prompts), batch_size), desc="Cleaning batches"):
        outputs = llm.chat(messages=all_prompts[i:i + batch_size], sampling_params=sampling_params, use_tqdm=False)
        cleaned_results.extend([output.outputs[0].text.strip() for output in outputs])
    
    return cleaned_results

def clean_parquet_with_llm(
    input_file: str | Path,
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    batch_size: int = 32,
    temperature: float = 0.1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 4096
) -> Path:
    input_path = Path(input_file).resolve()
    output_path = input_path.parent / (input_path.stem + "_llm_cleaned" + input_path.suffix)
    
    logging.info(f"Loading model: {model_name}")
    llm = LLM(
        model=model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
        dtype="half",
        max_model_len=max_model_len,
    )
    sampling_params = SamplingParams(temperature=temperature, top_p=0.95, max_tokens=2048)
    
    logging.info(f"Loading data from {input_path}")
    df = pd.read_parquet(input_path)
    logging.info(f"Loaded {len(df)} records")
    
    reasoning_traces = []
    indices_to_clean = []
    for idx, row in df.iterrows():
        out = row['output']
        if isinstance(out, dict) and out.get('error') is None and 'thinking' in out:
            thinking = out.get('thinking', '')
            if thinking:
                reasoning_traces.append(thinking)
                indices_to_clean.append(idx)
    
    logging.info(f"Found {len(reasoning_traces)} reasoning traces to clean")
    cleaned_traces = clean_reasoning_batch(llm, reasoning_traces, sampling_params, batch_size)
    
    df_cleaned = df.copy()
    total_chars_removed = 0
    for idx, cleaned_text in zip(indices_to_clean, cleaned_traces):
        original_thinking = df.at[idx, 'output']['thinking']
        output_dict = df_cleaned.at[idx, 'output'].copy()
        output_dict['thinking'] = cleaned_text
        df_cleaned.at[idx, 'output'] = output_dict
        total_chars_removed += len(original_thinking) - len(cleaned_text)
    
    avg_removed = total_chars_removed / len(cleaned_traces) if cleaned_traces else 0
    logging.info(f"Total chars removed: {total_chars_removed} (avg: {avg_removed:.1f}/trace)")
    
    df_cleaned.to_parquet(output_path, index=False)
    logging.info(f"Saved to {output_path}")
    
    for i in range(min(3, len(reasoning_traces))):
        reduction = 100 * (len(reasoning_traces[i]) - len(cleaned_traces[i])) / len(reasoning_traces[i])
        logging.info(f"Example {i+1}: {len(reasoning_traces[i])} -> {len(cleaned_traces[i])} chars ({reduction:.1f}% reduction)")
    
    return output_path

if __name__ == "__main__":
    output_path = clean_parquet_with_llm(
        input_file="data/out/distillation/mmlu_synth_gptoss_b_t0_8.parquet",
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        batch_size=32,
        temperature=0.1,
        gpu_memory_utilization=0.9,
        max_model_len=4096
    )
    logging.info(f"Done! Output: {output_path}")
