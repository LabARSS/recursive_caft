from pathlib import Path
import pandas as pd
from core.distillation.evaluate_cleaning_quality import select_samples, evaluate_batch, compute_stats


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent.parent
    data_dir = project_root / "data" / "out" / "distillation"
    eval_dir = project_root / "data" / "out" / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    branch_a = data_dir / "mmlu_synth_gptoss_a_t0_8.parquet"
    branch_b = data_dir / "mmlu_synth_gptoss_b_t0_8_cleaned_32b.parquet"
    
    print("Selecting samples...")
    _, test = select_samples(branch_a, branch_b, n_test=100)
    print(f"Test samples: {len(test)}")
    
    output = eval_dir / "cleaning_quality_results.parquet"
    
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"
    
    if output.exists():
        print(f"Found existing results with {len(pd.read_parquet(output))} samples")
        print("Computing statistics on existing results...")
        compute_stats(output)
        print("To resume evaluation, remove the existing file or modify the script")
    else:
        print("\nRunning evaluation with kosbu/Llama-3.3-70B-Instruct-AWQ on H100...")
        evaluate_batch(
            samples=test,
            model_name="kosbu/Llama-3.3-70B-Instruct-AWQ",
            output_file=output,
            batch_size=10
        )
        
        print("\nComputing statistics...")
        compute_stats(output)
