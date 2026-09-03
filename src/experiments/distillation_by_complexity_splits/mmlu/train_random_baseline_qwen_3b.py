from experiments.distillation_by_complexity_splits.mmlu.random_baseline_shared import run

# seeds=[42, 43, 44] for a spread over seeds instead of a single run
run(model_name="Qwen2.5-3B-Instruct", seeds=[42])
