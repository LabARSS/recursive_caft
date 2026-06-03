"""Standalone worker: run one Evaluator._evaluate_single in a fresh process.

Spawned by Evaluator.evaluate() — one process per (checkpoint, dataset) — so
GPU/CPU resources fully reset between units and a flaky-GPU crash (SIGSEGV/139)
only kills the unit, which the parent restarts. Launched by file path:

    python _eval_worker.py <spec_path> <ds_idx>

where <spec_path> is a pickle of (EvaluatorConfig, tokenizer). Results are
written to disk by _evaluate_single (results.json / responses.parquet); the
parent recovers them via Evaluator._load_cached_result.
"""

import os
import pickle
import sys

# Make `core.*` importable when launched by file path: insert the repo `src/`
# (this file is src/core/evaluation/_eval_worker.py -> three dirs up is src/).
_SRC = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# Importing the evaluator also installs core.utils.runtime_trace (faulthandler,
# per-run log files, signal/excepthook handlers) for this worker process.
from core.evaluation.evaluator import Evaluator  # noqa: E402
from core.utils.logger import logger  # noqa: E402


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: {sys.argv[0]} <spec_path> <ds_idx>")
    spec_path, ds_idx = sys.argv[1], int(sys.argv[2])

    with open(spec_path, "rb") as f:
        config, tokenizer = pickle.load(f)

    evaluator = Evaluator(config, tokenizer)
    model, tok = evaluator._load_model()
    model.eval()
    logger.info(
        f"[worker] ds_idx={ds_idx} model={config.model_path} "
        f"attn={getattr(model.config, '_attn_implementation', '?')}"
    )
    # Writes results.json / responses.parquet; the parent reads them back.
    evaluator._evaluate_single(evaluator._datasets[ds_idx], model, tok)


if __name__ == "__main__":
    main()
