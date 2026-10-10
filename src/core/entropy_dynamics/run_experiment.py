"""
Three usage modes:

  # 1. Student entropy (original)
  python -m core.entropy_dynamics.run_experiment \
      --teacher_reasoning_path data/out/distillation/mmlu_synth_gptoss_b_t0_8.parquet \
      --out_dir artifacts/entropy_dynamics/mmlu_forced \
      --role student \
      --mode forced \
      --students qwen_3b phi4_mini

  # 2. Proxy/teacher entropy (NEW — same chunks, different model)
  python -m core.entropy_dynamics.run_experiment \
      --teacher_reasoning_path data/out/distillation/mmlu_synth_gptoss_b_t0_8.parquet \
      --out_dir artifacts/entropy_dynamics/mmlu_forced \
      --role proxy \
      --mode forced \
      --students qwen_72b

  # 3. Analysis only (combine student + proxy results)
  python -m core.entropy_dynamics.run_experiment \
      --teacher_reasoning_path data/out/distillation/mmlu_synth_gptoss_b_t0_8.parquet \
      --out_dir artifacts/entropy_dynamics/mmlu_forced \
      --analysis_only

Output files are named by role + teacher source, e.g.:
  entropy_dynamics_student_mmlu_synth_gptoss_b_t0_8.parquet
  entropy_dynamics_proxy_mmlu_synth_gptoss_b_t0_8.parquet
So different runs never overwrite each other.
"""

import argparse
import sys

#from core.entropy_dynamics.analyzer import run_full_analysis
from core.entropy_dynamics.config import (
    EntropyDynamicsConfig, ExperimentRole, InferenceMode, StudentModelConfig,
)
from core.entropy_dynamics.runner import EntropyDynamicsRunner

# ── Model presets by role ──
STUDENT_MODELS = [
    StudentModelConfig(model_id="Qwen/Qwen2.5-3B", label="qwen_3b"),
    StudentModelConfig(model_id="microsoft/Phi-4-mini-instruct", label="phi4_mini"),
    StudentModelConfig(model_id="meta-llama/Llama-3.2-3B-Instruct", label="llama_3b"),
]

PROXY_MODELS = [
    StudentModelConfig(model_id="/mnt/data198/LLM/models/Qwen2.5-32B-Instruct", label="qwen_32b"),
    StudentModelConfig(model_id="/mnt/data198/LLM/models/Qwen2.5-14B-Instruct", label="qwen_14b"),
    StudentModelConfig(model_id="/mnt/data198/LLM/models/Mistral-Small-3.2-24B-Instruct-2506", label="mistral_24b"),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Entropy Dynamics Experiment")
    parser.add_argument("--teacher_reasoning_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="artifacts/entropy_dynamics")
    parser.add_argument("--role", type=str, default="student",
                        choices=["student", "proxy"],
                        help="Which model role to run: student or proxy/teacher")
    parser.add_argument("--mode", type=str, default="forced",
                        choices=["forced", "continuation"])
    parser.add_argument("--dataset_type", type=str, default="mmlu",
                        choices=["mmlu", "gpqa", "gsm8k"])
    parser.add_argument("--window_size", type=int, default=32)
    parser.add_argument("--students", type=str, nargs="*", default=None,
                        help="Filter models by label, e.g. --students qwen_3b qwen_72b")
    parser.add_argument("--analysis_only", action="store_true",
                        help="Skip inference, just run analysis on existing results.")
    parser.add_argument("--use_vllm", action="store_true",
                        help="Use vLLM for fast batched inference (10-50x speedup)")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85,
                        help="vLLM GPU memory fraction (default 0.85)")
    parser.add_argument("--max_model_len", type=int, default=8192,
                        help="vLLM max sequence length")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="vLLM tensor parallelism (use >1 for multi-GPU)")
    return parser.parse_args()


def main():
    args = parse_args()

    role = ExperimentRole(args.role)

    # Pick model list based on role
    if role == ExperimentRole.STUDENT:
        all_models = STUDENT_MODELS
    else:
        all_models = PROXY_MODELS

    # Filter by --students if provided
    models = all_models
    if args.students:
        models = [m for m in all_models if m.label in args.students]
        if not models:
            all_labels = [m.label for m in all_models]
            print(f"No matching models for {args.students}. Available for role={role.value}: {all_labels}")
            sys.exit(1)

    config = EntropyDynamicsConfig(
        teacher_reasoning_path=args.teacher_reasoning_path,
        out_dir=args.out_dir,
        students=models,
        mode=InferenceMode(args.mode),
        role=role,
        dataset_type=args.dataset_type,
        window_size=args.window_size,
    )

    if not args.analysis_only:
        print(f"Running entropy dynamics experiment:")
        print(f"  Role: {config.role.value}")
        print(f"  Mode: {config.mode.value}")
        print(f"  Dataset: {config.dataset_type}")
        print(f"  Window: {config.window_size} tokens")
        print(f"  Models: {[m.label for m in config.students]}")
        print(f"  Teacher reasoning: {config.teacher_reasoning_path}")
        print(f"  Output file: {config.results_filename}")
        print(f"  Backend: {'vLLM' if args.use_vllm else 'HuggingFace'}")
        print()

        if args.use_vllm:
            from core.entropy_dynamics.runner_vllm import VLLMEntropyRunner
            runner = VLLMEntropyRunner(
                config,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_model_len=args.max_model_len,
                tensor_parallel_size=args.tensor_parallel_size,
            )
        else:
            runner = EntropyDynamicsRunner(config)

        df = runner.run()
        print(f"\nInference complete. {len(df)} measurements collected.")

    # Run basic analysis on student results
    results_path = config.out_path / config.results_filename
    if results_path.exists():
        print(f"\nRunning analysis on {results_path}...")
    #    run_full_analysis(results_path, config.out_path / "analysis")
    else:
        print(f"Results file not found at {results_path}. Run inference first.")


if __name__ == "__main__":
    main()
