"""Tests for the self-contained resource monitor.

These are pure unit tests: cgroup/proc reads are pointed at temp fixtures so the
suite is deterministic across platforms and never touches the real ``/proc`` or
``/sys/fs/cgroup``.
"""

from __future__ import annotations

import time

from klit_flow import monitor
from klit_flow.monitor import (
    ResourceMonitor,
    ResourceSample,
    _human,
    format_sample,
    read_cgroup_memory,
    read_rss_bytes,
)


def test_human_readable_sizes() -> None:
    assert _human(None) == "-"
    assert _human(512) == "512B"
    assert _human(1024) == "1.0KiB"
    assert _human(5 * 1024 * 1024) == "5.0MiB"
    assert _human(3 * 1024**3) == "3.0GiB"


def test_cgroup_fraction() -> None:
    s = ResourceSample(cgroup_current_bytes=512, cgroup_limit_bytes=1024)
    assert s.cgroup_fraction == 0.5
    # Unknown / unlimited limit => no fraction
    assert ResourceSample(cgroup_current_bytes=512).cgroup_fraction is None
    assert ResourceSample(cgroup_current_bytes=512, cgroup_limit_bytes=0).cgroup_fraction is None


def test_read_cgroup_v2(tmp_path) -> None:
    (tmp_path / "memory.current").write_text("2097152\n")
    (tmp_path / "memory.max").write_text("8388608\n")
    current, limit = read_cgroup_memory(tmp_path)
    assert current == 2097152
    assert limit == 8388608


def test_read_cgroup_v2_unlimited(tmp_path) -> None:
    (tmp_path / "memory.current").write_text("2097152\n")
    (tmp_path / "memory.max").write_text("max\n")
    current, limit = read_cgroup_memory(tmp_path)
    assert current == 2097152
    assert limit is None


def test_read_cgroup_v1(tmp_path) -> None:
    mem = tmp_path / "memory"
    mem.mkdir()
    (mem / "memory.usage_in_bytes").write_text("1048576\n")
    (mem / "memory.limit_in_bytes").write_text("4194304\n")
    current, limit = read_cgroup_memory(tmp_path)
    assert current == 1048576
    assert limit == 4194304


def test_read_cgroup_v1_unlimited_sentinel(tmp_path) -> None:
    mem = tmp_path / "memory"
    mem.mkdir()
    (mem / "memory.usage_in_bytes").write_text("1048576\n")
    # v1 reports a huge sentinel instead of "max" when no limit is set.
    (mem / "memory.limit_in_bytes").write_text(str(1 << 63) + "\n")
    current, limit = read_cgroup_memory(tmp_path)
    assert current == 1048576
    assert limit is None


def test_read_cgroup_absent(tmp_path) -> None:
    # No cgroup files at all -> graceful (None, None), never raises.
    assert read_cgroup_memory(tmp_path) == (None, None)


def test_read_rss_from_statm(tmp_path) -> None:
    statm = tmp_path / "statm"
    # fields: size resident shared text lib data dt ; resident is field index 1
    statm.write_text("1000 256 64 1 0 700 0\n")
    rss = read_rss_bytes(statm)
    assert rss is not None
    assert rss > 0  # 256 pages * page size


def test_read_rss_missing_statm_degrades(tmp_path) -> None:
    # Missing /proc and (likely) no psutil -> None, never raises. If psutil is
    # installed it may return a real number; either way it must not error.
    result = read_rss_bytes(tmp_path / "nope")
    assert result is None or result > 0


def test_format_sample_with_and_without_label() -> None:
    s = ResourceSample(rss_bytes=1024, cgroup_current_bytes=512, cgroup_limit_bytes=1024)
    line = format_sample(s, label="persist")
    assert "[persist]" in line
    assert "rss=1.0KiB" in line
    assert "cgroup=512B/1.0KiB" in line
    assert "(50%)" in line
    # Without cgroup data, only rss is shown.
    assert "cgroup" not in format_sample(ResourceSample(rss_bytes=1024))


def test_monitor_peak_summary_is_deterministic() -> None:
    """Peak tracking keeps the max rss/cgroup across samples (no timing involved)."""
    mon = ResourceMonitor(label="t")
    mon._record_peak(
        ResourceSample(rss_bytes=1000, cgroup_current_bytes=1000, cgroup_limit_bytes=4000)
    )
    mon._record_peak(
        ResourceSample(rss_bytes=3000, cgroup_current_bytes=3000, cgroup_limit_bytes=4000)
    )
    mon._record_peak(
        ResourceSample(rss_bytes=2000, cgroup_current_bytes=2000, cgroup_limit_bytes=4000)
    )
    summary = mon._summary_line()
    assert "peak rss=2.9KiB" in summary  # 3000B
    assert "peak cgroup=2.9KiB" in summary
    assert "75% of limit" in summary  # 3000/4000


def test_monitor_thread_emits_lines_and_summary(monkeypatch) -> None:
    """The background thread samples, emits at least one line, then a summary."""
    monkeypatch.setattr(
        monitor,
        "sample",
        lambda: ResourceSample(rss_bytes=2000, cgroup_current_bytes=2000, cgroup_limit_bytes=4000),
    )
    lines: list[str] = []
    with ResourceMonitor(label="t", interval=0.05, sink=lines.append):
        time.sleep(0.12)  # allow a couple of samples
    assert any("[t]" in line for line in lines)
    assert "peak" in lines[-1]


def test_monitor_stop_without_start_is_safe() -> None:
    ResourceMonitor(label="t").stop()  # must not raise
