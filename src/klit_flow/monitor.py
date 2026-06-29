"""Self-contained resource monitoring (memory / cgroup headroom).

This module lets klit-flow log its own memory usage during long-running steps —
chiefly the OOM-prone "Persisting graph …" step that can be SIGKILL'd (exit
137) on large repos. It samples two things:

* **Process RSS** — resident memory of the current process.
* **cgroup memory** — usage and limit of the control group the process runs in.
  This is the number that actually matters under Docker/Kubernetes: the kernel
  OOM-killer fires (→ exit 137) when *cgroup* usage hits the *cgroup* limit, not
  when the host runs out of RAM. cgroup accounting also includes page cache from
  the growing DB file, so it explains "the DB got large and it died" cases that
  process RSS alone would miss.

Everything is best-effort and stdlib-only: every read is guarded so monitoring
can never crash or slow down the actual run, and it degrades gracefully on
platforms (e.g. Windows dev machines) without ``/proc`` or cgroups. No network,
no telemetry — consistent with the project's offline guarantee.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Where the sampler writes each line. Defaults to the module logger; the CLI
# passes a sink that echoes to stderr (and flushes) so lines survive a kill.
Sink = Callable[[str], None]

_PROC_STATM = Path("/proc/self/statm")
_CGROUP_ROOT = Path("/sys/fs/cgroup")


def _page_size() -> int:
    """Resolve the OS page size, falling back to 4 KiB where unavailable.

    ``os.sysconf`` exists on POSIX only; on Windows (dev machines) it is absent,
    so we default to 4096 rather than raise.
    """
    try:
        return os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, ValueError, OSError):
        return 4096


_PAGE_SIZE = _page_size()

# A cgroup limit reported as one of these (or larger) means "unlimited".
_CGROUP_UNLIMITED = "max"
_CGROUP_V1_UNLIMITED_MIN = 1 << 62  # v1 reports a huge sentinel instead of "max"


class ResourceSample(BaseModel):
    """One point-in-time snapshot of memory usage. All sizes in bytes."""

    rss_bytes: int | None = None
    cgroup_current_bytes: int | None = None
    cgroup_limit_bytes: int | None = None  # None => unlimited / unknown

    @property
    def cgroup_fraction(self) -> float | None:
        """Fraction of the cgroup memory limit in use (0.0–1.0), if known."""
        if self.cgroup_current_bytes is None or not self.cgroup_limit_bytes:
            return None
        return self.cgroup_current_bytes / self.cgroup_limit_bytes


def _human(n: int | None) -> str:
    """Render a byte count as a compact human string (``-`` for unknown)."""
    if n is None:
        return "-"
    size = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.0f}{unit}" if unit == "B" else f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TiB"  # unreachable, keeps type-checkers happy


def read_rss_bytes(statm: Path = _PROC_STATM) -> int | None:
    """Return current process resident memory in bytes, or ``None`` if unknown.

    Reads ``/proc/self/statm`` on Linux (the Docker case). Falls back to
    ``psutil`` if it happens to be installed, else returns ``None`` so callers
    on platforms without ``/proc`` degrade gracefully rather than error.
    """
    try:
        if statm.exists():
            resident_pages = int(statm.read_text().split()[1])
            return resident_pages * _PAGE_SIZE
    except (OSError, ValueError, IndexError):
        pass
    try:  # optional, never a hard dependency
        import psutil

        return int(psutil.Process().memory_info().rss)
    except Exception:
        return None


def _read_int(path: Path) -> int | None:
    """Read a single integer from a sysfs/cgroup file, ``None`` on any failure."""
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def read_cgroup_memory(root: Path = _CGROUP_ROOT) -> tuple[int | None, int | None]:
    """Return ``(current_bytes, limit_bytes)`` for the process's memory cgroup.

    Tries cgroup v2 first (unified hierarchy at ``<root>/memory.{current,max}``)
    then falls back to v1 (``<root>/memory/memory.{usage,limit}_in_bytes``). A
    limit of ``None`` means unlimited or unknown. All failures degrade to
    ``(None, None)`` — monitoring must never break the run.
    """
    # cgroup v2 (unified)
    v2_current = root / "memory.current"
    v2_max = root / "memory.max"
    if v2_current.exists():
        current = _read_int(v2_current)
        raw_max = ""
        try:
            raw_max = v2_max.read_text().strip()
        except OSError:
            raw_max = ""
        limit = None if raw_max in ("", _CGROUP_UNLIMITED) else _read_int(v2_max)
        return current, limit

    # cgroup v1
    v1_current = root / "memory" / "memory.usage_in_bytes"
    v1_limit = root / "memory" / "memory.limit_in_bytes"
    if v1_current.exists():
        current = _read_int(v1_current)
        limit = _read_int(v1_limit)
        if limit is not None and limit >= _CGROUP_V1_UNLIMITED_MIN:
            limit = None  # sentinel meaning "no limit set"
        return current, limit

    return None, None


def sample() -> ResourceSample:
    """Take a single best-effort snapshot of process + cgroup memory."""
    current, limit = read_cgroup_memory()
    return ResourceSample(
        rss_bytes=read_rss_bytes(),
        cgroup_current_bytes=current,
        cgroup_limit_bytes=limit,
    )


def format_sample(s: ResourceSample, *, label: str | None = None) -> str:
    """Format a sample as a single compact log line."""
    parts = []
    if label:
        parts.append(f"[{label}]")
    parts.append(f"rss={_human(s.rss_bytes)}")
    frac = s.cgroup_fraction
    if s.cgroup_current_bytes is not None:
        cg = f"cgroup={_human(s.cgroup_current_bytes)}/{_human(s.cgroup_limit_bytes)}"
        if frac is not None:
            cg += f" ({frac * 100:.0f}%)"
        parts.append(cg)
    return " ".join(parts)


class ResourceMonitor:
    """Background sampler that periodically logs memory and tracks the peak.

    Use as a context manager around a long step::

        with ResourceMonitor(label="persist", sink=echo):
            store.add_edges(edges)

    A daemon thread samples every ``interval`` seconds, emits a line via *sink*,
    and records the peak RSS / peak cgroup usage. On exit it emits a one-line
    peak summary — including how close usage got to the cgroup limit, which is
    the key signal for diagnosing exit-137 OOM kills.
    """

    def __init__(
        self,
        *,
        label: str = "run",
        interval: float = 5.0,
        sink: Sink | None = None,
    ) -> None:
        self._label = label
        self._interval = max(0.05, interval)
        self._sink: Sink = sink or logger.info
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._peak_rss = 0
        self._peak_cgroup = 0
        self._peak_fraction: float | None = None

    def _record_peak(self, s: ResourceSample) -> None:
        if s.rss_bytes is not None:
            self._peak_rss = max(self._peak_rss, s.rss_bytes)
        if s.cgroup_current_bytes is not None:
            self._peak_cgroup = max(self._peak_cgroup, s.cgroup_current_bytes)
        frac = s.cgroup_fraction
        if frac is not None and (self._peak_fraction is None or frac > self._peak_fraction):
            self._peak_fraction = frac

    def _run(self) -> None:
        # Emit an immediate first sample, then every interval until stopped.
        while True:
            try:
                s = sample()
                self._record_peak(s)
                self._sink(f"  {format_sample(s, label=self._label)}")
            except Exception:  # monitoring must never crash the run
                logger.debug("resource sample failed", exc_info=True)
            if self._stop.wait(self._interval):
                return

    def start(self) -> ResourceMonitor:
        """Start the background sampler (idempotent)."""
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._run, name="klit-resource-monitor", daemon=True
            )
            self._thread.start()
        return self

    def _summary_line(self) -> str:
        """Build the one-line peak summary emitted on stop()."""
        summary = f"  [{self._label}] peak rss={_human(self._peak_rss or None)}"
        if self._peak_cgroup:
            summary += f", peak cgroup={_human(self._peak_cgroup)}"
            if self._peak_fraction is not None:
                summary += f" ({self._peak_fraction * 100:.0f}% of limit)"
        return summary

    def stop(self) -> None:
        """Stop sampling and emit a peak summary."""
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=self._interval + 1.0)
        self._thread = None
        self._sink(self._summary_line())

    def __enter__(self) -> ResourceMonitor:
        return self.start()

    def __exit__(self, *_: object) -> None:
        self.stop()
