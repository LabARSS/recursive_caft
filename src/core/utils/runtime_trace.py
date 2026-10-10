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
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil

from core.utils.logger import LOG_DIR, RUN_BASENAME, logger
from core.utils.memory_limit import MEM_WATCHDOG_EXIT_CODE, read_cgroup_mem_limit


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

_FAULTS_FILE = LOG_DIR / f"{RUN_BASENAME}.faults.log"
# Plain Python file object with line buffering. Kept alive at module scope so
# faulthandler can write into it for the full lifetime of the process — even
# if Python's allocator is starved.
_faults_fp = open(_FAULTS_FILE, "a", buffering=1)

# Route fatal-signal tracebacks (SIGSEGV/SIGBUS/SIGFPE/SIGILL — the flaky eval GPU
# dies with a native SIGSEGV) to the faults file, not the default stderr. The parent
# supervisor inherits the child's stderr without capturing it, so a stderr-only dump
# lands in no file; this makes the native trace durable in <run>.faults.log.
faulthandler.enable(all_threads=True, file=_faults_fp)

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


def _start_mem_watchdog() -> bool:
    """In-process backstop: self-terminate before the container RAM limit wedges us.

    A daemon thread samples this process's RSS against the detected cgroup limit
    and, on breach, dumps all-thread stacks (so we know *where*) then os._exit()s
    with MEM_WATCHDOG_EXIT_CODE — a clean death instead of the uninterruptible
    kernel-reclaim hang that ignores Ctrl-C and SIGKILL.

    Defence-in-depth: the parent supervisor's watchdog (subprocess_supervision) is
    the hard guarantee and fires first (lower fraction); this covers the
    in-process / unsupervised path and adds an internal stack dump. Caveat: a
    Python thread cannot preempt a GIL-holding C malloc stall, so it relies on the
    approach to the limit being gradual (it is — minutes in practice) while the
    process is still responsive. No-op when disabled or no container limit exists.
    """
    if os.environ.get("EVAL_MEM_WATCHDOG", "1") == "0":
        return False
    frac = float(os.environ.get("EVAL_MEM_WATCHDOG_SELF_FRAC", "0.95"))
    interval = float(os.environ.get("EVAL_MEM_WATCHDOG_INTERVAL_S", "2"))
    if frac <= 0:
        return False
    limit = read_cgroup_mem_limit()
    if limit is None:
        return False  # bare metal / dev host — nothing to guard against
    ceiling = limit * frac
    proc = psutil.Process()

    def _loop() -> None:
        while True:
            time.sleep(interval)
            try:
                rss = proc.memory_info().rss
            except Exception:
                continue
            if rss < ceiling:
                continue
            msg = (
                f"[trace] MEM WATCHDOG: RSS {rss / 1e9:.1f}GB >= {frac:.2f}×"
                f"{limit / 1e9:.0f}GB container limit — self-terminating before the "
                f"cgroup wedges this process (would ignore Ctrl-C and SIGKILL)."
            )
            try:
                logger.error(msg)
            except Exception:
                pass
            try:
                faulthandler.dump_traceback(all_threads=True, file=_faults_fp)
                _faults_fp.flush()
            except Exception:
                pass
            try:
                sys.stderr.write(msg + "\n")
                sys.stderr.flush()
            except Exception:
                pass
            os._exit(MEM_WATCHDOG_EXIT_CODE)

    threading.Thread(target=_loop, name="mem-watchdog", daemon=True).start()
    return True


_mem_watchdog_on = _start_mem_watchdog()

logger.info(
    f"[trace] runtime_trace installed (faulthandler, excepthook, signal handlers, "
    f"mem_watchdog={'on' if _mem_watchdog_on else 'off'}); "
    f"git_sha={_git_sha()} faults_file={_FAULTS_FILE}"
)
