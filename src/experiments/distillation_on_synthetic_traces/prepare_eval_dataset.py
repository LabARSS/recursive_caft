"""
Prepare unified eval dataset for all branches from MMLU PRO STEM.
This ensures all branches are evaluated on the same questions.
"""
import logging
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def prepare_unified_eval_dataset(
    source_file: str | Path = "data/source/mmlu_pro_stem.tsv",
    output_file: str | Path = "data/out/sft_data/branches_eval.tsv",
    eval_size: int = 500,
    seed: int = 42,
):
    """
    Create a unified eval dataset for all branches.
    
    Args:
        source_file: Path to MMLU PRO STEM source file
        output_file: Path to output eval dataset
        eval_size: Number of samples to use for evaluation
        seed: Random seed for sampling
    """
    source_file = Path(source_file).resolve()
    output_file = Path(output_file).resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Loading source data from {source_file}")
    df = pd.read_csv(source_file, sep="\t", dtype=str, keep_default_na=False)
    logging.info(f"Loaded {len(df)} samples")
    
    # Sample eval_size questions
    df_eval = df.sample(n=min(eval_size, len(df)), random_state=seed)
    logging.info(f"Sampled {len(df_eval)} eval samples")
    
    # Save eval dataset
    df_eval.to_csv(output_file, sep="\t", index=False)
    logging.info(f"Saved eval dataset to {output_file}")
    
    # Print sample statistics
    logging.info(f"Question IDs range: {df_eval['question_id'].min()} - {df_eval['question_id'].max()}")
    logging.info(f"Categories: {df_eval['category'].value_counts().to_dict()}")
    
    return output_file


if __name__ == "__main__":
    prepare_unified_eval_dataset()
