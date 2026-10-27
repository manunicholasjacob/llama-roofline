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

So this module measures the thing directly rather than inferring it from scatter: of all the
CPU time the machine spent during our measurement window, how much went to somebody else.

Method, and why it needs no notion of clock rate or core count. The aggregate CPU counter is
summed over every logical processor, so over a window it advanced by exactly the machine's
whole CPU capacity, in whatever units the platform counts in: jiffies from ``/proc/stat`` on
Linux and Android, 100 ns ticks from ``GetSystemTimes`` on Windows. Dividing by it cancels
the units out, which is why this works on all three without a per-platform constant and
without needing ``SC_CLK_TCK``, ``os.cpu_count()`` or the elapsed wall time. Where no counter
is readable it returns ``None`` and callers carry on, because a check that cannot run must
not look like a check that passed.

What a flag does and does not mean. It says the machine was busy, not that the ceiling is
wrong, and those are different claims. Two CPU-only spinners on an otherwise idle Raspberry
Pi 5 take the foreign fraction to 0.50 and leave the measured ceiling unchanged at
13.6 GB/s, because that board reaches its ceiling with one or two threads and the spinners
never touch memory. The Android compiler cost 32% because compiling means streaming. Treat a
flag as a reason to re-measure and compare, never as a correction factor.

Observed values, so the 0.10 threshold can be calibrated against something:

* idle Raspberry Pi 5, nothing else running                       0.017
* Raspberry Pi 5 with two CPU-only spinners                       0.497
* Windows laptop during ordinary desktop use                      0.205 to 0.228
* Galaxy M21 running Android's background dexopt job              around 0.6
* a shared GitHub Actions runner                                  flagged, Runner.Worker

The Windows figure is the common case rather than the exotic one. On that machine two
back-to-back ceiling measurements came out at 52.8 and 57.8 GB/s, 9% apart, while
``stability`` reported 0.976 and 0.977 for them. Neither run looked unstable, both were
wrong by some unknown amount, and the foreign fraction is the only output that said so.
"""

from __future__ import annotations

import os
import platform
from typing import Any, Dict, List, Optional, Tuple

# Fraction of the machine's CPU capacity consumed by other processes during the measurement,
# above which the ceiling should not be trusted. An otherwise idle board sits in the low
# percents. The Android case sat near 0.6.
CONTENTION_THRESHOLD = 0.10


# --------------------------------------------------------------------- Linux and Android


def _linux_machine_times() -> Optional[Tuple[int, int]]:
    """(idle, total) CPU ticks across all cores, from the aggregate line of /proc/stat.

    ``total`` includes idle, and deliberately includes ``nice``: a nice'd process competes
    for bandwidth exactly as hard as any other, which is the case this module exists for.
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
        values = [int(x) for x in parts[1:]]
    except ValueError:
        return None
    # user nice system idle iowait irq softirq steal guest guest_nice
    idle = values[3] + (values[4] if len(values) > 4 else 0)
    return idle, sum(values)


def _linux_own_time() -> Optional[int]:
    """This process's CPU time in the same jiffies /proc/stat counts in.

    utime and stime are fields 14 and 15 of /proc/self/stat and they aggregate the whole
    thread group, so the measurement's own worker threads land here rather than being
    mistaken for somebody else's work. The fields are located relative to the final ')'
    because an executable name can contain spaces and brackets, which defeats splitting the
    whole line.
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


# ------------------------------------------------------------------------------- Windows


def _windows_machine_times() -> Optional[Tuple[int, int]]:
    """(idle, total) in 100 ns ticks, summed over all processors, from GetSystemTimes."""
    try:
        import ctypes
        from ctypes import wintypes

        class FILETIME(ctypes.Structure):
            _fields_ = [("dwLowDateTime", wintypes.DWORD),
                        ("dwHighDateTime", wintypes.DWORD)]

        def as_int(ft: "FILETIME") -> int:
            return (ft.dwHighDateTime << 32) | ft.dwLowDateTime

        idle_ft, kernel_ft, user_ft = FILETIME(), FILETIME(), FILETIME()
        ok = ctypes.windll.kernel32.GetSystemTimes(
            ctypes.byref(idle_ft), ctypes.byref(kernel_ft), ctypes.byref(user_ft))
        if not ok:
            return None
        # Kernel time already includes idle time on Windows, so total is kernel + user.
        return as_int(idle_ft), as_int(kernel_ft) + as_int(user_ft)
    except Exception:
        return None


def _windows_own_time() -> Optional[int]:
    """This process's kernel + user time in the same 100 ns ticks GetSystemTimes uses.

    GetProcessTimes covers every thread of the process, including ones that have since
    exited, which is what is wanted: the thread pool's time must not count as foreign.
    """
    try:
        import ctypes
        from ctypes import wintypes

        class FILETIME(ctypes.Structure):
            _fields_ = [("dwLowDateTime", wintypes.DWORD),
                        ("dwHighDateTime", wintypes.DWORD)]

        # The signatures have to be declared. GetCurrentProcess returns the pseudo-handle
        # (HANDLE)-1, and without an explicit restype ctypes hands it back as a 32-bit
        # c_int, which does not widen to a 64-bit HANDLE on x64. GetProcessTimes then fails
        # and sets no last-error, so it looks like an unsupported platform rather than a
        # calling-convention mistake.
        kernel32 = ctypes.WinDLL("kernel32")
        LPFILETIME = ctypes.POINTER(FILETIME)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.GetCurrentProcess.argtypes = []
        kernel32.GetProcessTimes.restype = wintypes.BOOL
        kernel32.GetProcessTimes.argtypes = [wintypes.HANDLE, LPFILETIME, LPFILETIME,
                                             LPFILETIME, LPFILETIME]

        creation, exit_, kernel, user = FILETIME(), FILETIME(), FILETIME(), FILETIME()
        ok = kernel32.GetProcessTimes(
            kernel32.GetCurrentProcess(), ctypes.byref(creation), ctypes.byref(exit_),
            ctypes.byref(kernel), ctypes.byref(user))
        if not ok:
            return None

        def as_int(ft: "FILETIME") -> int:
            return (ft.dwHighDateTime << 32) | ft.dwLowDateTime

        return as_int(kernel) + as_int(user)
    except Exception:
        return None


# ----------------------------------------------------------------------------- portable


def machine_times() -> Optional[Tuple[int, int]]:
    """(idle, total) CPU time for the whole machine, in the platform's counter units."""
    if platform.system() == "Windows":
        return _windows_machine_times()
    return _linux_machine_times()


def own_time() -> Optional[int]:
    """CPU time used by this process, in the platform's counter units."""
    if platform.system() == "Windows":
        return _windows_own_time()
    return _linux_own_time()


def snapshot() -> Optional[Dict[str, Any]]:
    """Capture what is needed to work out foreign CPU use later. None when unsupported."""
    machine = machine_times()
    own = own_time()
    if machine is None or own is None:
        return None
    return {"idle": machine[0], "total": machine[1], "own": own}


def foreign_fraction(before: Optional[Dict[str, Any]],
                     after: Optional[Dict[str, Any]]) -> Optional[float]:
    """Share of the machine's CPU capacity used by other processes between two snapshots.

    1.0 would mean every core was fully busy with somebody else's work for the whole
    window. Returns None when a snapshot is missing or a counter moved backwards.
    """
    # `is None` rather than falsiness: None is the only value that means "could not check",
    # and a check that cannot run must not look like a check that passed.
    if before is None or after is None:
        return None
    capacity = after["total"] - before["total"]
    idle_delta = after["idle"] - before["idle"]
    own_delta = after["own"] - before["own"]
    # These counters are not strictly monotonic on Windows. GetSystemTimes sums
    # per-processor times, and cores being parked and unparked makes the sum go backwards
    # now and then: on one laptop `total` fell by 0.58 s worth of ticks between two reads a
    # fraction of a second apart. A regression inside a single interface means the window
    # cannot be measured, so the honest answer is None. Returning 0.0 would be far worse,
    # because 0.0 reads as "checked, and the machine was idle".
    if capacity <= 0 or idle_delta < 0 or own_delta < 0:
        return None
    busy = capacity - idle_delta
    if busy < 0:
        return None
    foreign = busy - own_delta
    # Unlike the above, this subtraction crosses two interfaces read a moment apart, so a
    # small negative is ordinary rounding rather than evidence of anything.
    if foreign < 0:
        foreign = 0
    return min(1.0, foreign / capacity)


def process_table() -> Optional[Dict[int, Tuple[str, int]]]:
    """pid -> (name, cpu time), for naming whatever is competing.

    Linux and Android only. Windows returns None: the fraction above is still reported
    there, just without names, which is the useful half.
    """
    if platform.system() == "Windows":
        return None
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
        rest = line[close + 2:].split()
        if len(rest) < 13:
            continue
        try:
            out[pid] = (line[open_paren + 1:close], int(rest[11]) + int(rest[12]))
        except ValueError:
            continue
    return out


def top_foreign(before: Optional[Dict[int, Tuple[str, int]]],
                after: Optional[Dict[int, Tuple[str, int]]],
                limit: int = 3) -> List[Tuple[str, float]]:
    """The processes that used the most CPU between two process tables, as (name, time).

    A process that started during the window counts all of its usage, which is the honest
    reading: it was not there before and it was busy after. A reused pid is treated as new
    rather than credited with the previous owner's time.
    """
    if before is None or after is None:
        return []
    deltas: List[Tuple[str, float]] = []
    for pid, (name, used) in after.items():
        prior = before.get(pid)
        delta = used - prior[1] if prior and prior[0] == name else used
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
