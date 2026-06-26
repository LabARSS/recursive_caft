"""read_cgroup_mem_limit / process_tree_rss: container RAM-limit detection.

CPU-only. Points the cgroup path constants at temp fixtures and fakes
virtual_memory().total so the v1/v2 parsing and "unlimited" rules are exercised
without depending on the host's actual cgroup layout.
"""

import os
import types

import core.utils.memory_limit as mem

_TOTAL = 500_000_000_000  # fake physical RAM


def _files(monkeypatch, tmp_path, *, v2=None, v1=None):
    """Repoint the cgroup constants at fixtures (missing file if value is None)."""
    for attr, name, val in (("_CGROUP_V2_MAX", "v2", v2), ("_CGROUP_V1_LIMIT", "v1", v1)):
        p = tmp_path / name
        if val is not None:
            p.write_text(val)
        monkeypatch.setattr(mem, attr, str(p))
    monkeypatch.setattr(mem.psutil, "virtual_memory", lambda: types.SimpleNamespace(total=_TOTAL))


def test_v2_real_limit_wins(monkeypatch, tmp_path):
    _files(monkeypatch, tmp_path, v2="250000000000\n", v1="123\n")
    assert mem.read_cgroup_mem_limit() == 250_000_000_000


def test_v2_max_falls_through_to_v1(monkeypatch, tmp_path):
    _files(monkeypatch, tmp_path, v2="max\n", v1="200000000000\n")
    assert mem.read_cgroup_mem_limit() == 200_000_000_000


def test_value_at_or_above_total_is_unlimited(monkeypatch, tmp_path):
    # cgroup v1 "unlimited" sentinel is a near-INT64 value >= physical RAM → None.
    _files(monkeypatch, tmp_path, v2="max", v1="9223372036854771712")
    assert mem.read_cgroup_mem_limit() is None


def test_none_when_no_cgroup_files(monkeypatch, tmp_path):
    _files(monkeypatch, tmp_path, v2=None, v1=None)
    assert mem.read_cgroup_mem_limit() is None


def test_garbage_value_ignored(monkeypatch, tmp_path):
    _files(monkeypatch, tmp_path, v2="not-a-number", v1=None)
    assert mem.read_cgroup_mem_limit() is None


def test_process_tree_rss_self_is_positive():
    assert mem.process_tree_rss(os.getpid()) > 0


def test_process_tree_rss_dead_pid_is_zero():
    # PID 0 is never a real user process for psutil.Process → treated as gone.
    assert mem.process_tree_rss(2**31 - 1) == 0
