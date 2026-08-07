"""Portable-ish machine description. Everything here degrades to None rather than failing."""

from __future__ import annotations

import os
import platform
import re
import subprocess
from typing import Any, Dict, Optional


def _run(cmd, timeout: float = 5.0) -> Optional[str]:
    try:
        out = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL, timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.decode("utf-8", errors="replace") if out.returncode == 0 else None


def cpu_model() -> Optional[str]:
    sysname = platform.system()
    if sysname == "Linux":
        try:
            with open("/proc/cpuinfo", "r", errors="replace") as f:
                text = f.read()
        except OSError:
            text = ""
        for key in ("model name", "Model name", "Hardware", "Model"):
            m = re.search(rf"^{key}\s*:\s*(.+)$", text, re.MULTILINE)
            if m:
                return m.group(1).strip()
        # Raspberry Pi and friends put the board name here.
        try:
            with open("/proc/device-tree/model", "rb") as f:
                return f.read().decode("utf-8", errors="replace").strip("\x00 \n")
        except OSError:
            pass
    elif sysname == "Darwin":
        out = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        if out:
            return out.strip()
    elif sysname == "Windows":
        out = os.environ.get("PROCESSOR_IDENTIFIER")
        wmic = _run(["wmic", "cpu", "get", "name", "/value"])
        if wmic:
            m = re.search(r"Name=(.+)", wmic)
            if m:
                return m.group(1).strip()
        if out:
            return out.strip()
    return platform.processor() or None


def physical_cores() -> Optional[int]:
    sysname = platform.system()
    if sysname == "Darwin":
        out = _run(["sysctl", "-n", "hw.physicalcpu"])
        if out and out.strip().isdigit():
            return int(out.strip())
    elif sysname == "Linux":
        try:
            with open("/proc/cpuinfo", "r", errors="replace") as f:
                text = f.read()
            pairs = set()
            phys = core = None
            for line in text.splitlines():
                if line.startswith("physical id"):
                    phys = line.split(":")[1].strip()
                elif line.startswith("core id"):
                    core = line.split(":")[1].strip()
                    pairs.add((phys, core))
            if pairs:
                return len(pairs)
        except OSError:
            pass
    elif sysname == "Windows":
        out = _run(["wmic", "cpu", "get", "NumberOfCores", "/value"])
        if out:
            nums = [int(x) for x in re.findall(r"NumberOfCores=(\d+)", out)]
            if nums:
                return sum(nums)
    return None


def total_ram_bytes() -> Optional[int]:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")  # Linux/macOS
    except (ValueError, OSError, AttributeError):
        pass
    if platform.system() == "Windows":
        try:
            import ctypes

            class _MEMSTAT(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = _MEMSTAT()
            stat.dwLength = ctypes.sizeof(_MEMSTAT)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return int(stat.ullTotalPhys)
        except Exception:
            return None
    return None


def collect() -> Dict[str, Any]:
    """One dict describing the machine, safe to embed in results JSON."""
    logical = os.cpu_count()
    return {
        "os": platform.system(),
        "os_release": platform.release(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu": cpu_model(),
        "logical_cores": logical,
        "physical_cores": physical_cores(),
        "ram_bytes": total_ram_bytes(),
    }


def default_thread_sweep(max_threads: Optional[int] = None) -> list:
    """A small, well-spaced thread sweep: 1, half, all physical, all logical.

    Decode saturates at a low thread count, so a dense sweep wastes the user's time;
    what matters is bracketing the knee and showing prefill still scaling past it.
    """
    logical = max_threads or os.cpu_count() or 4
    phys = physical_cores() or logical
    phys = min(phys, logical)
    candidates = {1, max(1, phys // 2), phys, logical}
    return sorted(t for t in candidates if 1 <= t <= logical)
