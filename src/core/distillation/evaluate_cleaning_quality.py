import os
os.environ["VLLM_USE_V1"] = "0"

import json
import pandas as pd
from pathlib import Path
from vllm import LLM, SamplingParams
from tqdm import tqdm

from core.prompts.cleaning_judge import create_judge_prompt, FEWSHOT_EXAMPLE


def select_samples(branch_a_file: Path, branch_b_file: Path, n_fewshot: int = 3, n_test: int = 100):
    df_a = pd.read_parquet(branch_a_file)
    df_b = pd.read_parquet(branch_b_file)
    
    df_a['question_id'] = df_a['input'].apply(lambda x: str(x.get('question_id')) if isinstance(x, dict) and x.get('question_id') is not None else None)
    df_a['answer'] = df_a['output'].apply(lambda x: x.get('answer') if isinstance(x, dict) else None)
    df_a['error'] = df_a['output'].apply(lambda x: x.get('error') if isinstance(x, dict) else None)
    df_a['gold'] = df_a['input'].apply(lambda x: x.get('gold') if isinstance(x, dict) else None)
    df_a['thinking_a'] = df_a['output'].apply(lambda x: x.get('thinking', '') if isinstance(x, dict) else '')
    
    df_b['question_id'] = df_b['input'].apply(lambda x: str(x.get('question_id')) if isinstance(x, dict) and x.get('question_id') is not None else None)
    df_b['thinking_b'] = df_b['output'].apply(lambda x: x.get('thinking', '') if isinstance(x, dict) else '')
    
    correct_a = df_a[(df_a['error'].isna() | (df_a['error'] == False)) & (df_a['answer'] == df_a['gold'])].copy()
    
    merged = correct_a.merge(df_b[['question_id', 'thinking_b']], on='question_id', how='inner')
    
    matched = [
        {
            'question_id': row['question_id'],
            'original': row['thinking_a'],
            'cleaned': row['thinking_b']
        }
        for _, row in merged.iterrows()
    ]
    
    return matched[:n_fewshot], matched[n_fewshot:n_fewshot+n_test]


def evaluate_batch(samples: list, model_name: str, output_file: Path, batch_size: int = 10):
    llm = LLM(
        model=model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=32768,
        disable_custom_all_reduce=True,
        enforce_eager=True
    )
    
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=512,
        top_p=1.0
    )
    
    all_results = []
    
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i+batch_size]
        batch_results = []
        
        prompts = []
        for sample in batch:
            msgs = create_judge_prompt(sample['original'], sample['cleaned'], [FEWSHOT_EXAMPLE])
            prompt = llm.get_tokenizer().apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            prompts.append(prompt)
        
        outputs = llm.generate(prompts, sampling_params)
        
        for sample, output in zip(batch, outputs):
            text = output.outputs[0].text.strip()
            try:
                scores = json.loads(text)
            except:
                scores = {"error": "parse_failed", "raw": text}
            
            result = {
                'question_id': sample['question_id'],
                'scores': scores,
                'original_length': len(sample['original']),
                'cleaned_length': len(sample['cleaned'])
            }
            batch_results.append(result)
            all_results.append(result)
        
        if i == 0:
            pd.DataFrame(batch_results).to_parquet(output_file, index=False)
        else:
            existing = pd.read_parquet(output_file)
            combined = pd.concat([existing, pd.DataFrame(batch_results)], ignore_index=True)
            combined.to_parquet(output_file, index=False)
        
        print(f"Batch {i//batch_size + 1}/{(len(samples)-1)//batch_size + 1} completed, saved {len(batch_results)} results")
    
    print(f"All results saved to {output_file}")
    return all_results


def compute_stats(results_file: Path):
    df = pd.read_parquet(results_file)
    
    scores_list = []
    for scores in df['scores']:
        if isinstance(scores, dict) and 'logical_preservation' in scores:
            scores_list.append(scores)
    
    if not scores_list:
        print("No valid scores found")
        return
    
    criteria = ['logical_preservation', 'noise_removal']
    print("\nScores (mean ± std):")
    for c in criteria:
        vals = [s[c] for s in scores_list if c in s and s[c] is not None]
        if vals:
            print(f"{c}: {sum(vals)/len(vals):.2f} ± {(sum((x-sum(vals)/len(vals))**2 for x in vals)/len(vals))**0.5:.2f}")
    
    avg_original = df['original_length'].mean()
    avg_cleaned = df['cleaned_length'].mean()
    compression = (1 - avg_cleaned/avg_original) * 100
    print(f"\nCompression: {compression:.1f}% ({avg_original:.0f} → {avg_cleaned:.0f} chars)")
