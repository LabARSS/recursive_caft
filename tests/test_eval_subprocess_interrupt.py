"""Worker-teardown contract for the subprocess supervisor.

Ctrl+C / scheduler SIGTERM must tear down the worker *and its whole process
group* (DataLoader / vLLM grandchildren) — not leave orphans, and not be mistaken
for a transient crash and retried. These exercise the terminate_process_group
helper directly; CPU-only, no model — just short-lived python subprocesses.
"""

import os
import signal
import subprocess
import sys
import time

import psutil

from core.utils.subprocess_supervision import terminate_process_group

# Worker that spawns a long-sleeping grandchild (inheriting the worker's process
# group), records both pids, then sleeps. Used to prove killpg reaches the whole tree.
_SPAWN_GRANDCHILD = (
    "import os, sys, subprocess, time\n"
    "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'])\n"
    "with open(sys.argv[1], 'w') as f:\n"
    "    f.write(f'{os.getpid()}\\n{child.pid}\\n'); f.flush(); os.fsync(f.fileno())\n"
    "time.sleep(300)\n"
)

# Worker that ignores SIGTERM and sleeps. Used to prove the SIGKILL escalation fires.
_IGNORE_SIGTERM = (
    "import os, sys, signal, time\n"
    "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
    "with open(sys.argv[1], 'w') as f:\n"
    "    f.write(str(os.getpid())); f.flush(); os.fsync(f.fileno())\n"
    "time.sleep(300)\n"
)


def _poll(predicate, timeout=10.0, interval=0.02):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def _dead(pid):
    """True once `pid` is gone or a not-yet-reaped zombie."""
    if not psutil.pid_exists(pid):
        return True
    try:
        return psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return True


def _force_kill(pid):
    try:
        os.kill(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def test_terminate_kills_whole_group_including_grandchild(tmp_path):
    pidfile = tmp_path / "pids.txt"
    proc = subprocess.Popen(
        [sys.executable, "-c", _SPAWN_GRANDCHILD, str(pidfile)],
        start_new_session=True,
    )
    grandchild_pid = None
    try:
        assert _poll(lambda: pidfile.exists() and len(pidfile.read_text().split()) == 2), (
            "worker never reported its pids"
        )
        worker_pid, grandchild_pid = (int(x) for x in pidfile.read_text().split())
        assert worker_pid == proc.pid

        terminate_process_group(proc, grace_s=5.0)

        assert proc.poll() is not None
        assert _dead(worker_pid), "worker survived teardown"
        assert _poll(lambda: _dead(grandchild_pid)), "grandchild survived teardown"
    finally:
        _force_kill(proc.pid)
        if grandchild_pid is not None:
            _force_kill(grandchild_pid)


def test_terminate_escalates_to_sigkill_when_sigterm_ignored(tmp_path):
    readyfile = tmp_path / "ready.txt"
    proc = subprocess.Popen(
        [sys.executable, "-c", _IGNORE_SIGTERM, str(readyfile)],
        start_new_session=True,
    )
    try:
        assert _poll(lambda: readyfile.exists() and readyfile.read_text().strip().isdigit()), (
            "worker never became ready"
        )
        worker_pid = int(readyfile.read_text())
        assert worker_pid == proc.pid

        start = time.monotonic()
        terminate_process_group(proc, grace_s=1.0)
        elapsed = time.monotonic() - start

        assert proc.poll() is not None
        assert _dead(worker_pid), "SIGTERM-ignoring worker survived teardown"
        assert elapsed >= 1.0, "should have waited the grace period before escalating to SIGKILL"
    finally:
        _force_kill(proc.pid)
