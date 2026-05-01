import logging
import pandas as pd
from pathlib import Path

from core.prompts.mmlu_branches_sft import (
    branch_a_sys_prompt, branch_a_user_prompt,
    branch_b_sys_prompt, branch_b_user_prompt,
    branch_c_sys_prompt, branch_c_user_prompt
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def extract_options_as_list(opts_dict):
    """Extract options from dict {'A': 'text', 'B': 'text', ...} to list"""
    if isinstance(opts_dict, dict):
        return [opts_dict[k] for k in sorted(opts_dict.keys())]
    return opts_dict

def prepare_branch_a_df(input_file):
    """Prepare Branch A: Q + options -> answer + CoT"""
    df = pd.read_parquet(input_file)
    
    valid_rows = []
    for _, row in df.iterrows():
        inp = row['input']
        out = row['output']
        
        if not isinstance(out, dict) or out.get('error') or not out.get('thinking'):
            continue
        
        options_list = extract_options_as_list(inp['options'])
        subject = inp.get('subject')
        
        valid_rows.append({
            'question': inp['question'],
            'options': str(options_list),
            'subject': subject or '',
            'thinking': out['thinking'],
            'answer_letter': out['answer']
        })
    
    return pd.DataFrame(valid_rows)

def prepare_branch_b_df(input_file):
    """Prepare Branch B: Q + options + gold -> explanation"""
    df = pd.read_parquet(input_file)
    
    valid_rows = []
    for _, row in df.iterrows():
        inp = row['input']
        out = row['output']
        
        if not isinstance(out, dict) or out.get('error') or not out.get('thinking'):
            continue
        
        options_list = extract_options_as_list(inp['options'])
        subject = inp.get('subject')
        
        valid_rows.append({
            'question': inp['question'],
            'options': str(options_list),
            'subject': subject or '',
            'gold_letter': inp['gold'],
            'thinking': out['thinking']
        })
    
    return pd.DataFrame(valid_rows)

def prepare_branch_c_df(input_file):
    """Prepare Branch C: Q + options + wrong answer -> error analysis"""
    df = pd.read_parquet(input_file)
    
    valid_rows = []
    for _, row in df.iterrows():
        inp = row['input']
        out = row['output']
        
        if not isinstance(out, dict) or out.get('error') or not out.get('thinking'):
            continue
        
        options_list = extract_options_as_list(inp['options'])
        subject = inp.get('subject')
        prev_answer = inp.get('model_answer_from_A', '')
        gold_answer = out.get('gold', inp.get('gold', ''))
        
        valid_rows.append({
            'question': inp['question'],
            'options': str(options_list),
            'subject': subject or '',
            'prev_answer': prev_answer,
            'gold_answer': gold_answer,
            'thinking': out['thinking']
        })
    
    return pd.DataFrame(valid_rows)

def save_branch_dataset(df, output_file, branch_name):
    """Save processed dataframe"""
    df.to_parquet(output_file, index=False)
    logging.info(f"{branch_name}: Saved {len(df)} samples to {output_file}")

if __name__ == "__main__":
    base_path = Path(__file__).parent / "../../../../data/out/distillation"
    output_path = Path(__file__).parent / "../../../../data/out/sft_data"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Branch A
    branch_a_file = base_path / "mmlu_synth_gptoss_a_t0_8.parquet"
    if branch_a_file.exists():
        logging.info("Preparing Branch A...")
        df_a = prepare_branch_a_df(branch_a_file)
        save_branch_dataset(df_a, output_path / "branch_a_train.parquet", "Branch A")
    else:
        logging.warning(f"Branch A data not found: {branch_a_file}")
    
    # Branch B
    branch_b_file = base_path / "mmlu_synth_gptoss_b_t0_8.parquet"
    if branch_b_file.exists():
        logging.info("Preparing Branch B...")
        df_b = prepare_branch_b_df(branch_b_file)
        save_branch_dataset(df_b, output_path / "branch_b_train.parquet", "Branch B")
    else:
        logging.warning(f"Branch B data not found: {branch_b_file}")
    
    # Branch C
    branch_c_file = base_path / "mmlu_synth_gptoss_c_t0_8.parquet"
    if branch_c_file.exists():
        logging.info("Preparing Branch C...")
        df_c = prepare_branch_c_df(branch_c_file)
        save_branch_dataset(df_c, output_path / "branch_c_train.parquet", "Branch C")
    else:
        logging.warning(f"Branch C data not found: {branch_c_file}")
    
    logging.info("Data preparation complete!")
    