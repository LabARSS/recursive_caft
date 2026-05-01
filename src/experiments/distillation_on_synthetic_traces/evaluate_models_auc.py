import logging
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from core.complexity_estimation.entropy.estimate_single_token_entropy import estimate_dataset
from core.utils.correctness import check_answer_correct_mmlu

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

MODELS = {
    "Branch A": "data/out/models/branch_a",
    "Branch B": "data/out/models/branch_b",
    "Branch C": "data/out/models/branch_c",
}
TEST_FILE = "data/source/mmlu_pro_stem.tsv"
TEST_LIMIT = 500

def add_roc_cols(df):
    """Add columns for ROC-AUC evaluation"""
    if "model_response" not in df.columns:
        df["model_response"] = ""
    if "model_answer" not in df.columns:
        df["model_answer"] = ""
    if "is_correct" not in df.columns:
        df["is_correct"] = False
    if "answer_logits" not in df.columns:
        df["answer_logits"] = None

def process_single_token_response(df, row_idx, inputs, outputs, response_idx, response, model, tokenizer):
    """Extract logits for A-J tokens from first generated token"""
    final_token_logits = outputs.scores[-1][response_idx]
    
    # Get token IDs for A-J
    token_ids = [tokenizer.encode(chr(65+i), add_special_tokens=False)[0] for i in range(10)]
    
    # Extract logits for A-J
    logits_aj = [final_token_logits[tid].item() for tid in token_ids]
    
    df.at[row_idx, "answer_logits"] = logits_aj
    df.at[row_idx, "model_response"] = response
    df.at[row_idx, "model_answer"] = response.strip()
    
    return response

def evaluate_branch(branch_name, model_path):
    """Evaluate single branch model"""
    logging.info(f"Loading {branch_name}: {model_path}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    output_file = Path(f"data/out/evaluation/{branch_name.lower().replace(' ', '_')}_results.parquet")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    df = estimate_dataset(
        in_filename=TEST_FILE,
        out_filename=output_file,
        model=model,
        tokenizer=tokenizer,
        get_subject_from_row=lambda row: row.get("base_cluster"),
        get_question_from_row=lambda row: row["question"],
        get_options_from_row=lambda row: eval(row["options"]) if isinstance(row["options"], str) else row["options"],
        check_answer_correct=check_answer_correct_mmlu,
        max_new_tokens=1,
        batch_size=32,
        dump_every=100,
        add_columns=add_roc_cols,
        process_response=process_single_token_response,
    )
    
    # Calculate ROC-AUC
    valid_df = df[df["answer_logits"].notna()].head(TEST_LIMIT)
    
    logits_matrix = np.array([x for x in valid_df["answer_logits"].values])
    probs_matrix = torch.softmax(torch.tensor(logits_matrix), dim=1).numpy()
    
    # Gold answers as one-hot
    gold_letters = valid_df["answer"].values
    gold_indices = [ord(letter) - ord('A') for letter in gold_letters]
    gold_one_hot = np.zeros((len(gold_indices), 10))
    for i, idx in enumerate(gold_indices):
        if 0 <= idx < 10:
            gold_one_hot[i, idx] = 1
    
    # Predictions
    pred_indices = np.argmax(probs_matrix, axis=1)
    accuracy = (pred_indices == gold_indices).mean()
    
    try:
        roc_auc = roc_auc_score(gold_one_hot, probs_matrix, multi_class='ovr', average='macro')
    except Exception as e:
        logging.warning(f"ROC-AUC failed: {e}")
        roc_auc = 0.0
    
    del model
    torch.cuda.empty_cache()
    
    return {
        "Branch": branch_name,
        "ROC-AUC": f"{roc_auc:.4f}",
        "Accuracy": f"{accuracy:.4f}",
        "Total": len(valid_df)
    }

if __name__ == "__main__":
    results = []
    
    for branch_name, model_path in MODELS.items():
        try:
            metrics = evaluate_branch(branch_name, model_path)
            results.append(metrics)
            logging.info(f"{branch_name}: ROC-AUC={metrics['ROC-AUC']}, Accuracy={metrics['Accuracy']}")
        except Exception as e:
            logging.error(f"Failed {branch_name}: {e}")
            results.append({
                "Branch": branch_name,
                "ROC-AUC": "ERROR",
                "Accuracy": "ERROR",
                "Total": 0
            })
    
    results_df = pd.DataFrame(results)
    logging.info(f"\n{'='*80}\nFINAL RESULTS\n{'='*80}\n{results_df.to_string(index=False)}")
    
    output_path = Path("data/out/evaluation/branch_comparison.csv")
    results_df.to_csv(output_path, index=False)
    logging.info(f"\nSaved to {output_path}")
