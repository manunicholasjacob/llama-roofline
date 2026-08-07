"""Portable memory-bandwidth microbenchmark -- the roofline's ceiling.

We need an *achievable* streaming-read rate, the same quantity a STREAM-style benchmark
reports, without shipping a compiler. numpy's ufuncs run in C with the GIL released, so
a thread pool over disjoint slices parallelises across cores.

Three read-only kernels are run because no single one saturates every machine:

  ``sum``  one stream, pairwise reduction            (bytes = n)
  ``max``  one stream, no accumulation dependency    (bytes = n)
  ``dot``  two streams through BLAS                  (bytes = 2n)

plus ``copy`` (read+write, reported for context only). The ceiling is the best
*read-only* result; copy is excluded because a write-allocating store costs extra
traffic and would inflate the number relative to what decode actually does.

This is a measured lower bound on the true ceiling, never an upper bound. Decode can
legitimately exceed it -- if that happens the report says so instead of printing a
nonsensical ">100% of peak".
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

# Big enough to blow past any last-level cache, small enough to be polite on a 2 GB Pi.
DEFAULT_WORKING_SET_MB = 512
MIN_WORKING_SET_MB = 64


class NumpyMissing(RuntimeError):
    """numpy is required to measure the ceiling; use --peak-bw to supply it instead."""


def _require_numpy():
    try:
        import numpy  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise NumpyMissing(
            "numpy is required to measure the memory-bandwidth ceiling.\n"
            "Install it (pip install numpy) or pass --peak-bw <GB/s> to skip this step."
        ) from exc
    return numpy


def choose_working_set_mb(requested: Optional[int] = None,
                          ram_bytes: Optional[int] = None) -> int:
    """Pick a working set that exceeds cache but will not push the machine into swap."""
    if requested:
        return max(MIN_WORKING_SET_MB, int(requested))
    mb = DEFAULT_WORKING_SET_MB
    if ram_bytes:
        # The copy kernel needs two buffers, so cap total at ~1/8 of RAM.
        mb = min(mb, max(MIN_WORKING_SET_MB, int(ram_bytes / 1e6 / 8)))
    return int(mb)


def _samples(fn, slices, nthreads: int, bytes_moved: int, reps: int) -> List[float]:
    """Every repetition's throughput, so the caller can judge stability, not just the best."""
    out: List[float] = []
    with ThreadPoolExecutor(max_workers=nthreads) as pool:
        for _ in range(reps):
            t0 = time.perf_counter()
            list(pool.map(fn, slices))
            dt = time.perf_counter() - t0
            if dt > 0:
                out.append(bytes_moved / dt / 1e9)
    return out


def _best_of(fn, slices, nthreads: int, bytes_moved: int, reps: int) -> float:
    return max(_samples(fn, slices, nthreads, bytes_moved, reps), default=0.0)


def _median(values: List[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else 0.5 * (s[mid - 1] + s[mid])


# Below this median-to-best ratio, the repetitions disagree enough that something else was
# using the machine. Best-of-N hides that: if the contention is constant, every repetition
# is depressed and the "best" is simply wrong.
STABILITY_THRESHOLD = 0.75


def measure(working_set_mb: Optional[int] = None,
            thread_counts: Optional[List[int]] = None,
            reps: int = 5,
            ram_bytes: Optional[int] = None,
            progress=None) -> Dict[str, Any]:
    """Measure achievable memory bandwidth. Returns a JSON-safe dict.

    ``peak_read_GBs`` is the headline: the best read-only kernel over all thread counts.
    """
    np = _require_numpy()

    mb = choose_working_set_mb(working_set_mb, ram_bytes)
    nbytes = mb * 1024 * 1024
    logical = os.cpu_count() or 4
    if not thread_counts:
        thread_counts = sorted({1, max(1, logical // 4), max(1, logical // 2), logical})
    thread_counts = [t for t in thread_counts if t >= 1]

    n64 = nbytes // 8
    a64 = np.ones(n64, dtype=np.float64)
    b64 = np.empty_like(a64)
    n32 = nbytes // 4
    a32 = np.ones(n32, dtype=np.float32)
    b32 = np.ones(n32, dtype=np.float32)

    results: Dict[str, Dict[str, float]] = {}
    raw: Dict[str, Dict[str, List[float]]] = {}
    try:
        for nt in thread_counts:
            sl64 = [(i * n64 // nt, (i + 1) * n64 // nt) for i in range(nt)]
            sl32 = [(i * n32 // nt, (i + 1) * n32 // nt) for i in range(nt)]
            cell = {
                "sum": _samples(lambda s: float(a64[s[0]:s[1]].sum()),
                                sl64, nt, a64.nbytes, reps),
                "max": _samples(lambda s: float(a64[s[0]:s[1]].max()),
                                sl64, nt, a64.nbytes, reps),
                "dot": _samples(lambda s: float(np.dot(a32[s[0]:s[1]], b32[s[0]:s[1]])),
                                sl32, nt, 2 * a32.nbytes, reps),
                "copy": _samples(lambda s: b64.__setitem__(slice(s[0], s[1]), a64[s[0]:s[1]]),
                                 sl64, nt, 2 * a64.nbytes, reps),
            }
            raw[str(nt)] = cell
            row = {k: max(v, default=0.0) for k, v in cell.items()}
            results[str(nt)] = row
            if progress:
                progress(nt, row)
    finally:
        del a64, b64, a32, b32

    read_kernels = ("sum", "max", "dot")
    peak = 0.0
    best_kernel = None
    best_threads = None
    for nt, row in results.items():
        for k in read_kernels:
            if row[k] > peak:
                peak, best_kernel, best_threads = row[k], k, int(nt)
    copy_peak = max((row["copy"] for row in results.values()), default=0.0)

    winning = raw.get(str(best_threads), {}).get(best_kernel, []) if best_kernel else []
    stability = (_median(winning) / peak) if peak and winning else None
    out: Dict[str, Any] = {
        "peak_read_GBs": peak,
        "best_kernel": best_kernel,
        "best_threads": best_threads,
        "copy_GBs": copy_peak,
        "working_set_mb": mb,
        "reps": reps,
        "stability": stability,
        "by_threads": results,
        "method": ("numpy threaded read kernels (sum/max/dot); ceiling is the best "
                   "read-only kernel, a measured lower bound on true peak"),
        "source": "measured",
    }
    if stability is not None and stability < STABILITY_THRESHOLD:
        out["unstable"] = True
        out["warning"] = (
            f"the bandwidth measurement was unstable (repetitions varied by "
            f"{100 * (1 - stability):.0f}% around the best result), which usually means "
            f"something else was using the machine. The ceiling is probably too low, and "
            f"every utilisation percentage derived from it is therefore too high. Close "
            f"other applications and re-run, or pass --peak-bw with a known-good number."
        )
    return out
