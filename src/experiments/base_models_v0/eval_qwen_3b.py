from pathlib import Path

from core.evaluation.v0_embedding_eval import V0EmbeddingEvalConfig, run_v0_embedding_eval

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
MODEL_NICK = "qwen_3b"

REPO_ROOT = Path(__file__).resolve().parents[3]
V0_DIR = REPO_ROOT / "artifacts" / "base_models_v0" / MODEL_NICK
MMLU_PARQUET = REPO_ROOT / "data" / "source" / "mmlu_pro_stem.parquet"
EVAL_OUT = V0_DIR / "eval_runs"


def main():
    run_v0_embedding_eval(
        V0EmbeddingEvalConfig(
            v0_dir=V0_DIR.as_posix(),
            base_model_id=MODEL_NAME,
            mmlu_test_parquet=MMLU_PARQUET.as_posix(),
            eval_out_path=EVAL_OUT.as_posix(),
        )
    )


if __name__ == "__main__":
    main()
