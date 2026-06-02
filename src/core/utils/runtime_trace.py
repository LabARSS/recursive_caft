"""Runtime tracing hooks for diagnosing silent process deaths.

Importing this module installs:
- Line-buffered stdout/stderr so tqdm/print output is not lost on abrupt exit
- faulthandler with all-threads stack dumps (heartbeat every 5 min, SIGUSR1 on demand,
  and a one-shot dump on fatal signals like SIGSEGV/SIGABRT)
- sys.excepthook + threading.excepthook that route uncaught exceptions through loguru
  so the traceback lands in the per-run log file before the process disappears
- SIGTERM / SIGABRT signal handlers that log before exiting

SIGKILL cannot be caught — the absence of the atexit "=== process exit ===" marker
emitted by logger.py is itself the signal that the OS OOM-killer struck.
"""

from __future__ import annotations

import faulthandler
import signal
import subprocess
import sys
import threading
from pathlib import Path

from core.utils.logger import LOG_DIR, RUN_BASENAME, logger


def _git_sha() -> str:
    """Best-effort short git SHA of the running checkout, '?' if unavailable."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).strip()
    except Exception:
        return "?"

try:
    sys.stdout.reconfigure(line_buffering=True)
except (AttributeError, OSError):
    pass
try:
    sys.stderr.reconfigure(line_buffering=True)
except (AttributeError, OSError):
    pass

faulthandler.enable(all_threads=True)

_FAULTS_FILE = LOG_DIR / f"{RUN_BASENAME}.faults.log"
# Plain Python file object with line buffering. Kept alive at module scope so
# faulthandler can write into it for the full lifetime of the process — even
# if Python's allocator is starved.
_faults_fp = open(_FAULTS_FILE, "a", buffering=1)

if hasattr(signal, "SIGUSR1"):
    faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False, file=_faults_fp)

faulthandler.dump_traceback_later(300, repeat=True, file=_faults_fp)


def _excepthook(exc_type, exc_value, exc_tb) -> None:
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_tb)
        return
    logger_obj = logger
    logger_obj.error(f"[trace] uncaught exception: {exc_type.__name__}: {exc_value}")
    from loguru import logger as _raw_logger

    _raw_logger.opt(exception=(exc_type, exc_value, exc_tb)).error("[trace] traceback follows")
    _raw_logger.complete()
    sys.__excepthook__(exc_type, exc_value, exc_tb)


sys.excepthook = _excepthook


def _thread_excepthook(args) -> None:
    from loguru import logger as _raw_logger

    _raw_logger.opt(exception=(args.exc_type, args.exc_value, args.exc_traceback)).error(
        f"[trace] uncaught thread exception in {args.thread.name}"
    )
    _raw_logger.complete()


threading.excepthook = _thread_excepthook


def _signal_handler(signum, frame) -> None:
    name = signal.Signals(signum).name
    logger.error(f"[trace] received signal {name} ({signum}) — exiting")
    from loguru import logger as _raw_logger

    _raw_logger.complete()
    sys.exit(128 + signum)


for _sig_name in ("SIGTERM", "SIGABRT", "SIGHUP"):
    _sig = getattr(signal, _sig_name, None)
    if _sig is not None:
        try:
            signal.signal(_sig, _signal_handler)
        except (OSError, ValueError):
            pass

logger.info(
    f"[trace] runtime_trace installed (faulthandler, excepthook, signal handlers); "
    f"git_sha={_git_sha()} faults_file={_FAULTS_FILE}"
)
