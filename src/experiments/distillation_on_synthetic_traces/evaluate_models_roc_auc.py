import logging
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from vllm import LLM, SamplingParams
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Config
TEST_FILE = "data/source/mmlu_pro_stem.tsv"
MODELS = {
    "Branch A": "data/out/models/branch_a",
    "Branch B": "data/out/models/branch_b",
    "Branch C": "data/out/models/branch_c",
}
BATCH_SIZE = 32
MAX_TOKENS = 512

def load_test_data(test_file, limit=None):
    """Load test data (not used in training)"""
    df = pd.read_csv(test_file, sep="\t")
    if limit:
        df = df.head(limit)
    logging.info(f"Loaded {len(df)} test samples")
    return df

def format_prompt(question, options, subject=None):
    """Format question as chat prompt"""
    options_text = '\n'.join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
    sys_msg = f"Answer the following multiple choice question about {subject}." if subject else "Answer the following multiple choice question."
    
    messages = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": f"Question: {question}\n\nOptions:\n{options_text}\n\nProvide reasoning and answer."}
    ]
    return messages

def extract_answer_letter(text):
    """Extract answer letter from generated text"""
    import re
    patterns = [
        r'[Aa]nswer[:\s]+([A-J])',
        r'\b([A-J])\s*$',
        r'[Oo]ption\s+([A-J])',
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).upper()
    return None

def evaluate_model(model_path, test_df):
    """Evaluate single model with ROC-AUC"""
    logging.info(f"Loading model: {model_path}")
    
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        dtype="auto",
        gpu_memory_utilization=0.9,
        max_model_len=2048,
    )
    
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=MAX_TOKENS,
        logprobs=20,
    )
    
    # Prepare prompts
    prompts = []
    gold_answers = []
    
    for _, row in test_df.iterrows():
        import ast
        options = ast.literal_eval(row['options']) if isinstance(row['options'], str) else row['options']
        subject = row.get('base_cluster', None)
        
        messages = format_prompt(row['question'], options, subject)
        prompts.append(llm.llm_engine.tokenizer.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        
        gold_letter = row['answer']
        gold_idx = ord(gold_letter) - ord('A')
        gold_answers.append(gold_idx)
    
    logging.info(f"Generating responses for {len(prompts)} samples...")
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
    
    predictions = []
    all_logprobs = []
    
    for output in outputs:
        text = output.outputs[0].text
        answer_letter = extract_answer_letter(text)
        
        if answer_letter:
            pred_idx = ord(answer_letter) - ord('A')
        else:
            pred_idx = -1
        predictions.append(pred_idx)
        
        token_ids = llm.llm_engine.tokenizer.tokenizer.encode("ABCDEFGHIJ", add_special_tokens=False)
        
        if output.outputs[0].logprobs:
            last_logprobs = output.outputs[0].logprobs[-1]
            probs = []
            for tid in token_ids[:10]:
                if tid in last_logprobs:
                    probs.append(np.exp(last_logprobs[tid].logprob))
                else:
                    probs.append(0.0)
            total = sum(probs) + 1e-10
            probs = [p / total for p in probs]
            all_logprobs.append(probs)
        else:
            all_logprobs.append([0.1] * 10)
    
    predictions = np.array(predictions)
    gold_answers = np.array(gold_answers)
    all_logprobs = np.array(all_logprobs)
    
    valid_mask = predictions >= 0
    accuracy = (predictions[valid_mask] == gold_answers[valid_mask]).mean()
    
    try:
        n_classes = 10
        gold_one_hot = np.zeros((len(gold_answers), n_classes))
        for i, label in enumerate(gold_answers):
            if 0 <= label < n_classes:
                gold_one_hot[i, label] = 1
        
        roc_auc = roc_auc_score(gold_one_hot, all_logprobs, multi_class='ovr', average='macro')
    except Exception as e:
        logging.warning(f"ROC-AUC calculation failed: {e}")
        roc_auc = 0.0
    
    return {
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "total_samples": len(predictions),
        "valid_samples": valid_mask.sum()
    }

if __name__ == "__main__":
    test_df = load_test_data(TEST_FILE, limit=500)
    
    results = []
    for branch_name, model_path in MODELS.items():
        logging.info(f"\n{'='*80}\nEvaluating {branch_name}\n{'='*80}")
        
        try:
            metrics = evaluate_model(model_path, test_df)
            results.append({
                "Branch": branch_name,
                "ROC-AUC": f"{metrics['roc_auc']:.4f}",
                "Accuracy": f"{metrics['accuracy']:.4f}",
                "Valid/Total": f"{metrics['valid_samples']}/{metrics['total_samples']}"
            })
            logging.info(f"{branch_name}: ROC-AUC={metrics['roc_auc']:.4f}, Accuracy={metrics['accuracy']:.4f}")
        except Exception as e:
            logging.error(f"Failed to evaluate {branch_name}: {e}")
            results.append({
                "Branch": branch_name,
                "ROC-AUC": "ERROR",
                "Accuracy": "ERROR",
                "Valid/Total": "0/0"
            })
    
    logging.info(f"\n{'='*80}\nFINAL RESULTS\n{'='*80}")
    results_df = pd.DataFrame(results)
    logging.info(f"\n{results_df.to_string(index=False)}")
    
    output_path = Path("data/out/evaluation/branch_comparison.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    logging.info(f"\nResults saved to {output_path}")
