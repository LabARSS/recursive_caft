from pathlib import Path


def get_last_checkpoint_dir(path):
    """
    Return the numerically-last `checkpoint-<step>` child directory of *path*,
    or None if there is none.

    Only checkpoint dirs count: other children (e.g. ResamplingTrainer's
    `resampling_trainer_data`) used to win the old alphabetical sort, which made
    BaseTrainer._run_training's trainer_state.json probe miss and silently
    disabled resume for resampling runs. Numeric ordering also fixes
    checkpoint-99 sorting after checkpoint-100.
    """
    p = Path(path)

    if not p.is_dir():
        raise NotADirectoryError(f"{p} is not a directory")

    checkpoint_dirs = [d for d in p.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")]
    checkpoint_dirs.sort(key=lambda d: int(d.name.split("-")[1]))

    return checkpoint_dirs[-1] if checkpoint_dirs else None
