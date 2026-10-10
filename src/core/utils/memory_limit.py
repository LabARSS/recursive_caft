"""Container memory-limit + process-tree RSS helpers for the memory watchdogs.

The eval/training workers can drive RSS up to the container's Docker/cgroup RAM
limit; once the memory cgroup is at memory.max the kernel reclaims in-thread and
the process wedges (high CPU, unresponsive to Ctrl-C *and* SIGKILL). The
watchdogs in `subprocess_supervision` (parent-side) and `runtime_trace`
(in-process) use these helpers to detect the approach early — while the worker is
still responsive — and kill it cleanly instead of letting it hang.
"""

from __future__ import annotations

import psutil

# Distinct nonzero exit code the in-process watchdog uses (runtime_trace) so the
# parent supervisor can recognise a self-triggered memory exit and count it
# against the memory-kill budget rather than the generic crash breaker.
MEM_WATCHDOG_EXIT_CODE = 137

# cgroup pseudo-files; module-level so tests can repoint them at fixtures.
_CGROUP_V2_MAX = "/sys/fs/cgroup/memory.max"
_CGROUP_V1_LIMIT = "/sys/fs/cgroup/memory/memory.limit_in_bytes"


def read_cgroup_mem_limit() -> int | None:
    """Container memory limit in bytes, or None if unlimited/undetectable.

    Reads cgroup v2 (memory.max) then v1 (memory.limit_in_bytes). A literal
    "max" or any value >= physical RAM means "no real container limit" → None,
    so callers fall back to physical RAM or simply disable the watchdog. Reading
    the exact byte value makes the watchdog robust to GiB-vs-GB ambiguity in how
    the limit was specified.
    """
    total = psutil.virtual_memory().total
    for path in (_CGROUP_V2_MAX, _CGROUP_V1_LIMIT):
        try:
            with open(path) as f:
                raw = f.read().strip()
        except OSError:
            continue
        if raw == "max":
            continue
        try:
            val = int(raw)
        except ValueError:
            continue
        # cgroup v1 "unlimited" is a near-INT64 sentinel; treat anything at or
        # above physical RAM as no binding limit.
        if val <= 0 or val >= total:
            continue
        return val
    return None


def process_tree_rss(pid: int) -> int:
    """Summed RSS (bytes) of `pid` and all its descendants; 0 if it's gone.

    Anonymous resident memory is what actually pushes the container toward its
    cgroup limit (page cache is reclaimable), so this — not cgroup
    memory.current — is the watchdog's trigger signal. It is also the quantity
    the production hang was observed on (RSS hit ~253GB before wedging).
    """
    try:
        proc = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return 0
    procs = [proc]
    try:
        procs.extend(proc.children(recursive=True))
    except psutil.NoSuchProcess:
        pass
    total = 0
    for p in procs:
        try:
            total += p.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total
