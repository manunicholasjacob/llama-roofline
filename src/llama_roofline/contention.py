"""Was anything else using the machine while we measured?

``membw.STABILITY_THRESHOLD`` catches contention that comes and goes, because that makes
the repetitions disagree. The comment beside it already names the case it cannot catch:

    if the contention is constant, every repetition is depressed and the "best" is simply
    wrong

That case is not hypothetical. On a Galaxy M21 (Exynos 9611) running Android 12, the same
read kernels with the same arguments measured 6.35 GB/s while Android was running its
background application optimization job and 9.32 GB/s once it had finished: the contended
reading is 32% low. Stability was 0.85 in the contended run and 0.87 in the quiet one, both
comfortably above the 0.75 flag, because the job ran for the whole measurement and depressed
every repetition equally. Nothing in the output marked the low number as wrong.

Worse, the job was running at nice 10. Being nice yields the CPU, so such a job looks
harmless to a scheduler-shaped intuition, but it does not yield memory bandwidth or cache,
which is all a bandwidth measurement cares about. Android schedules that job when the device
is idle and charging, which is exactly the state someone creates in order to benchmark.

So this module measures the thing directly rather than inferring it: how much CPU time went
to processes that are not us, over the window in which we measured. That is readable from
``/proc`` on Linux and Android. Elsewhere it returns ``None`` and callers carry on, because a
check that cannot run must not look like a check that passed.

What this does and does not tell you. It reports that the machine was busy, not that the
ceiling is wrong, and those are not the same thing. Two CPU-only spinners on an otherwise
idle Raspberry Pi 5 raise the foreign fraction to 0.50 and leave the measured ceiling
unchanged at 13.6 GB/s, because the Pi reaches its ceiling with one or two threads and the
spinners never touch memory. The Android job cost 32% because it was compiling, which means
streaming. A warning here is a reason to re-measure and compare, not a correction factor.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple

# Fraction of total CPU capacity consumed by other processes during the measurement, above
# which the ceiling should not be trusted. An otherwise idle desktop sits in the low
# percents. The Android case that prompted this module sat near 0.6.
CONTENTION_THRESHOLD = 0.10


def _clock_ticks() -> Optional[int]:
    try:
        hz = os.sysconf("SC_CLK_TCK")
    except (ValueError, OSError, AttributeError):
        return None
    return int(hz) if hz and hz > 0 else None


def _total_busy_ticks() -> Optional[int]:
    """Busy CPU ticks across all cores, from the aggregate line of /proc/stat.

    Busy excludes idle and iowait. It deliberately includes ``nice``: a nice'd process
    competes for bandwidth exactly as hard as any other, which is the whole point here.
    """
    try:
        with open("/proc/stat", "r", errors="replace") as f:
            first = f.readline()
    except OSError:
        return None
    parts = first.split()
    if not parts or parts[0] != "cpu" or len(parts) < 5:
        return None
    try:
        fields = [int(x) for x in parts[1:]]
    except ValueError:
        return None
    # user nice system idle iowait irq softirq steal guest guest_nice
    idle = fields[3] + (fields[4] if len(fields) > 4 else 0)
    return sum(fields) - idle


def _own_ticks() -> Optional[int]:
    """CPU ticks used by this process, all threads included.

    utime and stime are fields 14 and 15 of /proc/self/stat, and they aggregate the whole
    thread group, so the measurement's worker threads are counted here and not mistaken for
    someone else's work. The executable name can contain spaces and brackets, so the fields
    are located relative to the final ')' rather than by splitting the whole line.
    """
    try:
        with open("/proc/self/stat", "r", errors="replace") as f:
            line = f.read()
    except OSError:
        return None
    close = line.rfind(")")
    if close < 0:
        return None
    rest = line[close + 2:].split()
    # rest[0] is field 3 (state), so field 14 is rest[11] and field 15 is rest[12]
    if len(rest) < 13:
        return None
    try:
        return int(rest[11]) + int(rest[12])
    except ValueError:
        return None


def snapshot() -> Optional[Dict[str, Any]]:
    """Capture what is needed to work out foreign CPU use later. None when unsupported."""
    hz = _clock_ticks()
    total = _total_busy_ticks()
    own = _own_ticks()
    if hz is None or total is None or own is None:
        return None
    return {
        "hz": hz,
        "total_busy_ticks": total,
        "own_ticks": own,
        "monotonic": time.monotonic(),
        "ncpu": os.cpu_count() or 1,
    }


def foreign_fraction(before: Optional[Dict[str, Any]],
                     after: Optional[Dict[str, Any]]) -> Optional[float]:
    """Fraction of total CPU capacity used by other processes between two snapshots.

    1.0 would mean every core was fully busy with somebody else's work for the whole window.
    Returns None when either snapshot is missing or the window is too short to divide by.
    """
    # `is None` rather than falsiness: an empty mapping is a table we read successfully,
    # and None is the only value that means "could not check".
    if before is None or after is None:
        return None
    elapsed = after["monotonic"] - before["monotonic"]
    if elapsed <= 0:
        return None
    capacity = elapsed * before["hz"] * max(1, before["ncpu"])
    if capacity <= 0:
        return None
    total_delta = after["total_busy_ticks"] - before["total_busy_ticks"]
    own_delta = after["own_ticks"] - before["own_ticks"]
    foreign = total_delta - own_delta
    # Counters are sampled a moment apart from different files, so small negatives are
    # normal rounding rather than evidence of anything.
    if foreign < 0:
        foreign = 0
    return min(1.0, foreign / capacity)


def process_table() -> Optional[Dict[int, Tuple[str, int]]]:
    """pid -> (name, busy ticks), for naming whatever is competing. None when unsupported."""
    try:
        pids = [int(e) for e in os.listdir("/proc") if e.isdigit()]
    except (OSError, ValueError):
        return None
    out: Dict[int, Tuple[str, int]] = {}
    own = os.getpid()
    for pid in pids:
        if pid == own:
            continue
        try:
            with open(f"/proc/{pid}/stat", "r", errors="replace") as f:
                line = f.read()
        except OSError:
            continue  # the process exited between listdir and open, which is routine
        open_paren = line.find("(")
        close = line.rfind(")")
        if open_paren < 0 or close < open_paren:
            continue
        name = line[open_paren + 1:close]
        rest = line[close + 2:].split()
        if len(rest) < 13:
            continue
        try:
            out[pid] = (name, int(rest[11]) + int(rest[12]))
        except ValueError:
            continue
    return out


def top_foreign(before: Optional[Dict[int, Tuple[str, int]]],
                after: Optional[Dict[int, Tuple[str, int]]],
                limit: int = 3) -> List[Tuple[str, float]]:
    """The processes that used the most CPU between two process tables, as (name, ticks).

    Processes that started during the window count their whole usage, which is the honest
    reading: they were not there before and they were busy after.
    """
    if before is None or after is None:
        return []
    deltas: List[Tuple[str, float]] = []
    for pid, (name, ticks) in after.items():
        prior = before.get(pid)
        delta = ticks - prior[1] if prior and prior[0] == name else ticks
        if delta > 0:
            deltas.append((name, float(delta)))
    deltas.sort(key=lambda kv: kv[1], reverse=True)
    return deltas[:limit]


def describe(fraction: Optional[float],
             offenders: Optional[List[Tuple[str, float]]] = None) -> Optional[str]:
    """A warning for a measurement taken on a busy machine, or None when it looks clean."""
    if fraction is None or fraction < CONTENTION_THRESHOLD:
        return None
    names = ", ".join(name for name, _ in (offenders or [])[:3])
    who = f" The busiest were: {names}." if names else ""
    return (
        f"other processes used {100 * fraction:.0f}% of this machine's CPU capacity while the "
        f"bandwidth ceiling was being measured, so this ceiling, and every utilisation "
        f"percentage derived from it, should be treated as unreliable.{who} How much it costs "
        f"depends on what the other work was doing: competition for memory bandwidth measured "
        f"32% low in one case, while processes that only burn CPU on otherwise spare cores can "
        f"cost nothing at all. Constant competition does not show up as instability, because "
        f"it depresses every repetition equally, so a healthy stability figure does not rule "
        f"this out. Re-run on an idle machine and compare, or pass --peak-bw with a "
        f"known-good number."
    )
