import logging
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def format_branch_a(row):
    """Branch A: Q + options -> answer + CoT"""
    inp = row['input']
    out = row['output']
    
    if not isinstance(out, dict) or out.get('error') or not out.get('thinking'):
        return None
    
    # options is dict: {'A': 'text', 'B': 'text', ...}
    opts = inp['options']
    if isinstance(opts, dict):
        options_text = '\n'.join([f"{k}. {v}" for k, v in sorted(opts.items())])
    else:
        options_text = '\n'.join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(opts)])
    
    messages = [
        {"role": "system", "content": f"Answer the following multiple choice question about {inp.get('subject', 'the given topic')}."},
        {"role": "user", "content": f"Question: {inp['question']}\n\nOptions:\n{options_text}\n\nProvide reasoning and answer."},
        {"role": "assistant", "content": f"{out['thinking']}\n\nAnswer: {out['answer']}"}
    ]
    return messages

def format_branch_b(row):
    """Branch B: Q + options + gold -> explanation why correct"""
    inp = row['input']
    out = row['output']
    
    if not isinstance(out, dict) or out.get('error') or not out.get('thinking'):
        return None
    
    # options is dict: {'A': 'text', 'B': 'text', ...}
    opts = inp['options']
    if isinstance(opts, dict):
        options_text = '\n'.join([f"{k}. {v}" for k, v in sorted(opts.items())])
    else:
        options_text = '\n'.join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(opts)])
    
    gold_answer = inp['gold']
    
    messages = [
        {"role": "system", "content": f"Explain why the given answer is correct for this multiple choice question about {inp.get('subject', 'the given topic')}."},
        {"role": "user", "content": f"Question: {inp['question']}\n\nOptions:\n{options_text}\n\nCorrect answer: {gold_answer}\n\nExplain why this is correct."},
        {"role": "assistant", "content": out['thinking']}
    ]
    return messages

def format_branch_c(row):
    """Branch C: Q + options + previous wrong answer -> error analysis"""
    inp = row['input']
    out = row['output']
    
    if not isinstance(out, dict) or out.get('error') or not out.get('thinking'):
        return None
    
    # options is dict: {'A': 'text', 'B': 'text', ...}
    opts = inp['options']
    if isinstance(opts, dict):
        options_text = '\n'.join([f"{k}. {v}" for k, v in sorted(opts.items())])
    else:
        options_text = '\n'.join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(opts)])
    
    prev_answer = inp.get('model_answer_from_A', 'unknown')
    gold_answer = out.get('gold', inp.get('gold', 'unknown'))
    
    messages = [
        {"role": "system", "content": f"Analyze the error in the previous answer for this multiple choice question about {inp.get('subject', 'the given topic')}."},
        {"role": "user", "content": f"Question: {inp['question']}\n\nOptions:\n{options_text}\n\nPrevious answer: {prev_answer}\nCorrect answer: {gold_answer}\n\nAnalyze the error and explain the correct reasoning."},
        {"role": "assistant", "content": out['thinking']}
    ]
    return messages

def prepare_branch(branch_name, input_file, output_file, formatter):
    logging.info(f"Preparing {branch_name}...")
    df = pd.read_parquet(input_file)
    logging.info(f"Loaded {len(df)} records")
    
    messages_list = []
    for idx, row in df.iterrows():
        msgs = formatter(row)
        if msgs:
            messages_list.append({"messages": msgs})
    
    logging.info(f"Formatted {len(messages_list)} valid samples")
    
    output_df = pd.DataFrame(messages_list)
    output_df.to_parquet(output_file, index=False)
    logging.info(f"Saved to {output_file}")
    
    return output_df

if __name__ == "__main__":
    base_path = Path(__file__).parent / "../../../data/out/distillation"
    output_path = Path(__file__).parent / "../../../data/out/sft_data"
    output_path.mkdir(parents=True, exist_ok=True)
    
    prepare_branch(
        "Branch A",
        base_path / "mmlu_synth_gptoss_a_t0_8.parquet",
        output_path / "branch_a_sft.parquet",
        format_branch_a
    )
    
    prepare_branch(
        "Branch B",
        base_path / "mmlu_synth_gptoss_b_t0_8.parquet",
        output_path / "branch_b_sft.parquet",
        format_branch_b
    )
    
    prepare_branch(
        "Branch C",
        base_path / "mmlu_synth_gptoss_c_t0_8.parquet",
        output_path / "branch_c_sft.parquet",
        format_branch_c
    )
    
    logging.info("All branches prepared!")
