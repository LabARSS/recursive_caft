"""Supervise a GPU work unit in a fresh subprocess, restarting it on crash.

The eval/estimation GPU is flaky and dies with a native SIGSEGV (EXIT=139) mid-run.
Running each unit in its own `sys.executable` subprocess (not a fork) gives a full
resource reset (CUDA context, caching allocator, host RAM) even on clean exit and
isolates crashes — the parent does no GPU work, so it survives. `supervise_unit`
restarts a crashed worker with exponential backoff, bounded by a circuit breaker that
detects deterministic failures (repeated fast crashes) so we never spin forever.

Resume rides whatever per-unit checkpoints the worker writes; the supervisor just
re-runs the same command.
"""

import os
import signal
import subprocess
import time

from core.utils.logger import logger
from core.utils.memory_limit import MEM_WATCHDOG_EXIT_CODE, process_tree_rss, read_cgroup_mem_limit


def terminate_process_group(proc: subprocess.Popen, grace_s: float = 10.0) -> None:
    """SIGTERM the worker's whole process group, then SIGKILL stragglers.

    The worker is spawned with start_new_session=True, so it leads its own process
    group; signalling that group reaches the worker *and* any descendants it spawned
    (DataLoader / vLLM workers) — which proc.send_signal() to the direct child would
    miss. SIGTERM first lets runtime_trace's handler shut down cleanly and flush
    logs; SIGKILL after grace_s guarantees teardown if it's wedged in a CUDA C call.
    Catching KeyboardInterrupt means an impatient second Ctrl+C escalates immediately.
    """
    if proc.poll() is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
        try:
            proc.wait(timeout=grace_s)
            return
        except subprocess.TimeoutExpired:
            logger.warning(f"[unit] worker pid={proc.pid} ignored SIGTERM after {grace_s:.0f}s — sending SIGKILL")
    except KeyboardInterrupt:
        logger.warning("[unit] second interrupt — sending SIGKILL now")
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        return
    proc.wait()


def _request_child_stackdump(proc: subprocess.Popen, settle_s: float = 0.5) -> None:
    """Ask the worker to dump all-thread stacks before we kill it.

    runtime_trace registers SIGUSR1 → faulthandler all-threads dump into the
    worker's .faults.log, so this records *where* it was when it breached the
    memory ceiling. Sent to the worker pid only (not the group) since the handler
    lives in the main worker; best-effort, and only useful because the watchdog
    fires early while the worker still services signals.
    """
    sigusr1 = getattr(signal, "SIGUSR1", None)
    if sigusr1 is None:
        return
    try:
        os.kill(proc.pid, sigusr1)
        time.sleep(settle_s)  # give faulthandler a moment to write before SIGKILL
    except (ProcessLookupError, OSError):
        pass


def _wait_with_mem_watchdog(
    proc: subprocess.Popen,
    *,
    label: str,
    limit_bytes: int,
    frac: float,
    poll_interval_s: float,
) -> tuple[int, bool]:
    """Wait for `proc`, killing it early if the container nears its RAM limit.

    Polls the worker process tree's RSS every `poll_interval_s`; once it reaches
    `frac * limit_bytes` the worker is dumped + terminated *while still
    responsive*, before the cgroup pushes it into the uninterruptible reclaim
    hang. Returns (exit_code, mem_killed). KeyboardInterrupt/SystemExit propagate
    to the caller's teardown path unchanged.
    """
    ceiling = limit_bytes * frac
    while True:
        try:
            return proc.wait(timeout=poll_interval_s), False
        except subprocess.TimeoutExpired:
            rss = process_tree_rss(proc.pid)
            if rss >= ceiling:
                logger.error(
                    f"[unit] {label} worker pid={proc.pid} RSS {rss / 1e9:.1f}GB "
                    f">= {frac:.2f}×{limit_bytes / 1e9:.0f}GB container limit — killing "
                    f"early so it exits cleanly instead of wedging past the cgroup limit."
                )
                _request_child_stackdump(proc)
                # Short grace: the worker's checkpoint is already on disk, so we
                # don't need a clean shutdown — just get it dead while killable.
                terminate_process_group(proc, grace_s=5.0)
                code = proc.poll()
                return (code if code is not None else MEM_WATCHDOG_EXIT_CODE), True


def supervise_unit(
    cmd: list[str],
    env: dict[str, str],
    *,
    label: str,
    min_healthy_s: float,
    max_fast: int,
    max_attempts: int,
    mem_watchdog_frac: float = 0.0,
    mem_poll_interval_s: float = 2.0,
    max_mem_kills: int = 3,
) -> None:
    """Run `cmd` in a fresh subprocess, restarting on any nonzero exit until it exits 0.

    The flaky GPU dies with SIGSEGV/139; resume rides the worker's own checkpoints. A
    circuit breaker hard-stops on repeated fast failures (a deterministic bug, not
    transient infra) so we never spin forever.

    Interrupts (Ctrl+C / scheduler SIGTERM) must stop the unit, not be mistaken for a
    transient crash and retried. The worker is spawned in its own session
    (start_new_session=True) so terminal Ctrl+C reaches only this supervisor; we then
    tear down the worker's whole process group and re-raise, breaking the retry loop
    and propagating out so the program exits.

    Memory watchdog (mem_watchdog_frac > 0): the parent does no GPU work, so it stays
    responsive even when the worker thrashes against the container RAM limit. It polls
    the worker tree's RSS and kills it early — while still killable — once RSS reaches
    `mem_watchdog_frac` of the detected cgroup limit, instead of letting it wedge in
    uninterruptible kernel reclaim (the hang that ignores Ctrl-C and SIGKILL). A
    memory kill restarts the worker (a fresh process resets the pinned pool /
    fragmentation, so it may now fit); `max_mem_kills` consecutive memory kills means a
    genuinely over-budget unit, so we stop with a clear error rather than loop. The
    watchdog is disabled (plain blocking wait) when mem_watchdog_frac <= 0 or no
    container limit is detectable (bare metal / dev hosts).

    `label` identifies the unit in logs (e.g. "dataset=3" or "epoch=1").
    """
    mem_limit = read_cgroup_mem_limit() if mem_watchdog_frac > 0 else None
    if mem_watchdog_frac > 0 and mem_limit is None:
        logger.info(f"[unit] {label} memory watchdog off (no container RAM limit detected)")
    elif mem_limit is not None:
        logger.info(
            f"[unit] {label} memory watchdog armed at {mem_watchdog_frac:.2f}×"
            f"{mem_limit / 1e9:.0f}GB (poll {mem_poll_interval_s:.0f}s, max_mem_kills={max_mem_kills})"
        )

    consecutive_fast = 0
    consecutive_mem_kills = 0
    for attempt in range(1, max_attempts + 1):
        start = time.monotonic()
        logger.info(f"[unit] {label} starting worker attempt={attempt}")
        proc = subprocess.Popen(cmd, env=env, start_new_session=True)
        try:
            if mem_limit is None:
                code, mem_killed = proc.wait(), False
            else:
                code, mem_killed = _wait_with_mem_watchdog(
                    proc,
                    label=label,
                    limit_bytes=mem_limit,
                    frac=mem_watchdog_frac,
                    poll_interval_s=mem_poll_interval_s,
                )
        except BaseException as exc:  # KeyboardInterrupt (Ctrl+C) or SystemExit (parent SIGTERM)
            logger.warning(
                f"[unit] {label} supervisor interrupted by {type(exc).__name__} — "
                f"terminating worker group pid={proc.pid}"
            )
            terminate_process_group(proc)
            raise
        dur = time.monotonic() - start
        if code == 0:
            logger.info(f"[unit] {label} worker attempt={attempt} finished in {dur:.0f}s.")
            return

        # A memory kill — by us, or the worker's own in-process watchdog (exit
        # MEM_WATCHDOG_EXIT_CODE) — is its own failure class: it runs healthy for a
        # while (so it never trips the fast-crash breaker), but if it recurs it's an
        # over-budget unit, not transient infra. Bound it separately.
        if mem_killed or code == MEM_WATCHDOG_EXIT_CODE:
            consecutive_mem_kills += 1
            consecutive_fast = 0
            logger.warning(
                f"[unit] {label} worker attempt={attempt} pid={proc.pid} killed near the "
                f"RAM limit after {dur:.0f}s (consecutive_mem_kills={consecutive_mem_kills})"
            )
            if consecutive_mem_kills >= max_mem_kills:
                raise RuntimeError(
                    f"[unit] {label}: {consecutive_mem_kills} consecutive memory-limit kills — "
                    f"the unit is over budget for this container, not transient. Aborting "
                    f"(reduce batch/chunk size or raise the container RAM limit)."
                )
            backoff = min(30.0, 2.0**consecutive_mem_kills)
            logger.info(f"[unit] {label} restarting in {backoff:.0f}s …")
            time.sleep(backoff)
            continue

        consecutive_mem_kills = 0
        consecutive_fast = consecutive_fast + 1 if dur < min_healthy_s else 0
        logger.warning(
            f"[unit] {label} worker attempt={attempt} pid={proc.pid} crashed: "
            f"exit={code} after {dur:.0f}s (consecutive_fast={consecutive_fast})"
        )
        if consecutive_fast >= max_fast:
            raise RuntimeError(
                f"[unit] {label}: {consecutive_fast} consecutive crashes in "
                f"<{min_healthy_s:.0f}s — looks deterministic, not transient infra. "
                f"Aborting (last exit={code})."
            )
        backoff = min(30.0, 2.0**consecutive_fast)
        logger.info(f"[unit] {label} restarting in {backoff:.0f}s …")
        time.sleep(backoff)
    raise RuntimeError(f"[unit] {label}: exceeded max_attempts={max_attempts}.")
