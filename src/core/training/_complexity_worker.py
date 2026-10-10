"""Standalone worker: run one ComplexityEstimationRunner.estimate() in a fresh process.

Spawned by EstimateComplexityCallback (resampling_trainer.py) — one process per epoch's
estimation — so a flaky-GPU crash (SIGSEGV/139) only kills the estimation, which the
parent training process restarts, instead of taking the whole run down. Launched by file
path:

    python _complexity_worker.py <spec_path>

where <spec_path> is a pickle of
(model_dir, runner_config, estimator, dataset_adapter, attn_implementation). The live
training weights were saved to model_dir by the parent (adapter + thinking_token_rows.pt);
the worker reloads them via the evaluator's hardened LoRA load path. The runner writes the
per-epoch parquet to runner_config.out_path and is self-resuming via its sibling .tmp, so
a restart picks up where the crash left off. The parent reconciles the parquet afterwards.
"""

import os
import pickle
import sys

# Make `core.*` importable when launched by file path: insert the repo `src/`
# (this file is src/core/training/_complexity_worker.py -> three dirs up is src/).
_SRC = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# Importing the evaluator also installs core.utils.runtime_trace (faulthandler,
# per-run log files, signal/excepthook handlers) for this worker process, and gives
# us its base+LoRA+thinking-rows load path for free.
from core.complexity_estimation.complexity_estimation_runner import ComplexityEstimationRunner  # noqa: E402
from core.evaluation.evaluator import Evaluator, EvaluatorConfig, GenerationConfig  # noqa: E402
from core.utils.logger import logger  # noqa: E402


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} <spec_path>")
    spec_path = sys.argv[1]

    with open(spec_path, "rb") as f:
        model_dir, runner_config, estimator, dataset_adapter, attn_implementation = pickle.load(f)

    # Reuse the evaluator's LoRA load path: it reads adapter_config.json's
    # base_model_name_or_path, reapplies thinking_token_rows.pt, and attaches the adapter.
    # Only generation.attn_implementation is consulted by _load_model; the rest is unused.
    evaluator = Evaluator(
        EvaluatorConfig(
            model_path=model_dir,
            eval_dataset=dataset_adapter,
            generation=GenerationConfig(
                max_new_tokens=1,
                max_batch_size=1,
                attn_implementation=attn_implementation,
            ),
        ),
        tokenizer=dataset_adapter.dataset.tokenizer,
    )
    model, _ = evaluator._load_model()
    model.eval()
    logger.info(
        f"[worker] complexity model={model_dir} "
        f"attn={getattr(model.config, '_attn_implementation', '?')} out={runner_config.out_path}"
    )

    # Writes the per-epoch parquet (resumable via its .tmp); the parent reads it back.
    ComplexityEstimationRunner(config=runner_config, complexity_estimator=estimator).estimate(
        dataset_adapter=dataset_adapter, model=model
    )


if __name__ == "__main__":
    main()
