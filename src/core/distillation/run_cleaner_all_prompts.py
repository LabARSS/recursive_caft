"""Clean gptoss_b and qwen_b thinking traces with 3 prompt variants.

Loads model once, runs all 6 combinations sequentially.
Output files:
  mmlu_synth_gptoss_b_t0_8_cleaned_32b_prompt1.parquet
  mmlu_synth_gptoss_b_t0_8_cleaned_32b_prompt2.parquet
  mmlu_synth_gptoss_b_t0_8_cleaned_32b_prompt3.parquet
  mmlu_synth_qwen3_b_t0_8_cleaned_32b_prompt1.parquet
  mmlu_synth_qwen3_b_t0_8_cleaned_32b_prompt2.parquet
  mmlu_synth_qwen3_b_t0_8_cleaned_32b_prompt3.parquet
"""

import os
os.environ["VLLM_USE_V1"] = "0"

from pathlib import Path
from vllm import LLM
from clean_thinking_traces import clean_thinking_traces
from prompts_for_test import PROMPT_1, PROMPT_2, PROMPT_3, USER_PROMPT_TEMPLATE

PROMPTS = {
    "prompt1": PROMPT_1,
    "prompt2": PROMPT_2,
    "prompt3": PROMPT_3,
}

DATASETS = {
    "gptoss_b": "mmlu_synth_gptoss_b_t0_8.parquet",
    "qwen3_b": "mmlu_synth_qwen3_b_t0_8.parquet",
}

MODEL_NAME = "Qwen/Qwen2.5-32B-Instruct"
MAX_MODEL_LEN = 20480


def main():
    project_root = Path(__file__).parent.parent.parent.parent
    data_dir = project_root / "data" / "out" / "distillation"

    print(f"Loading model {MODEL_NAME}...")
    llm = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.95,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=MAX_MODEL_LEN,
        enforce_eager=False,
    )

    for ds_name, ds_file in DATASETS.items():
        input_file = data_dir / ds_file
        if not input_file.exists():
            print(f"SKIP {ds_name}: {input_file} not found")
            continue

        for prompt_name, prompt_text in PROMPTS.items():
            stem = ds_file.replace(".parquet", "")
            output_file = data_dir / f"{stem}_cleaned_32b_{prompt_name}.parquet"

            if output_file.exists():
                print(f"\nSKIP {ds_name} x {prompt_name}: {output_file.name} already exists")
                continue

            print(f"\n{'='*60}")
            print(f"Dataset: {ds_name}  |  Prompt: {prompt_name}")
            print(f"{'='*60}")

            clean_thinking_traces(
                input_file=input_file,
                output_file=output_file,
                batch_size=12,
                temperature=0.1,
                max_model_len=MAX_MODEL_LEN,
                max_thinking_chars=35000,
                checkpoint_every_n_batches=4,
                system_prompt=prompt_text,
                user_prompt_template=USER_PROMPT_TEMPLATE,
                llm=llm,
            )

    print("\nAll done!")


if __name__ == "__main__":
    main()
