"""Clean gptoss Branch C thinking traces."""

from pathlib import Path
from clean_thinking_traces import clean_thinking_traces

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent.parent
    data_dir = project_root / "data" / "out" / "distillation"
    
    clean_thinking_traces(
        input_file=data_dir / "mmlu_synth_gptoss_c_t0_8.parquet",
        output_file=data_dir / "mmlu_synth_gptoss_c_t0_8_cleaned_32b.parquet",
        model_name="Qwen/Qwen2.5-32B-Instruct",
        batch_size=12,
        temperature=0.1,
        max_model_len=20480,
        max_thinking_chars=35000,
        checkpoint_every_n_batches=4
    )
