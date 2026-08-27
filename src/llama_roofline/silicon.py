"""Work out which CPU core the machine is actually going to decode on.

The quantization advisor is only useful if it knows what it is advising. Rankings
measured on one microarchitecture do not carry to another, so the first job is to
identify the core, and the second is to be honest about how confident that is.

Four levels of confidence come out of here, and they are the difference between advice
and guesswork:

    exact         the machine in the measurement matrix, or the same board
    same-uarch    a different chip with the same core design
    related       a descendant or close relative, so extrapolation, clearly labelled
    unknown       not measured and not close to anything measured

Everything degrades to unknown rather than guessing. Pure standard library.
"""

from __future__ import annotations

import os
import platform
import re
from typing import Any, Dict, List, Optional

from . import sysinfo

# Arm implementer 0x41 part numbers. Only cores worth naming are listed; anything else
# comes back as its raw part id, which is still more useful than "aarch64".
ARM_PARTS = {
    0xd03: "Cortex-A53", 0xd04: "Cortex-A35", 0xd05: "Cortex-A55",
    0xd07: "Cortex-A57", 0xd08: "Cortex-A72", 0xd09: "Cortex-A73",
    0xd0a: "Cortex-A75", 0xd0b: "Cortex-A76", 0xd0c: "Neoverse-N1",
    0xd0d: "Cortex-A77", 0xd40: "Neoverse-V1", 0xd41: "Cortex-A78",
    0xd44: "Cortex-X1", 0xd46: "Cortex-A510", 0xd47: "Cortex-A710",
    0xd48: "Cortex-X2", 0xd49: "Neoverse-N2", 0xd4d: "Cortex-A715",
    0xd4e: "Cortex-X3", 0xd80: "Cortex-A520", 0xd81: "Cortex-A720",
    0xd82: "Cortex-X4",
}

# Cores measured in the shipped matrix.
MEASURED = ("cortex-a76", "golden-cove", "gracemont")

# Arm cores that descend from the A76 line closely enough that its result is worth
# quoting as an extrapolation. A55/A510/A520 are a different, in-order design and are
# deliberately absent: the format tax binds harder there, not the same way.
A76_RELATIVES = {
    "Cortex-A77": "direct successor to the A76",
    "Cortex-A78": "A76 line, two generations on",
    "Neoverse-N1": "server sibling of the A76",
    "Cortex-A710": "A76 line, Armv9",
    "Cortex-A715": "A76 line, Armv9",
    "Cortex-A720": "A76 line, Armv9",
}

_INTEL_GEN_RE = re.compile(r"(\d{2})th Gen Intel", re.IGNORECASE)
_INTEL_SKU_RE = re.compile(r"i[3579]-(\d{4,5})([A-Z]*)", re.IGNORECASE)
_CORE_ULTRA_RE = re.compile(r"Core\(?TM\)?\s*Ultra", re.IGNORECASE)

# Vendor words that settle which instruction set a CPU marketing string is describing.
# Only consulted when the caller overrides the CPU, because at that point the ISA of
# the machine actually running this code says nothing about the machine being asked about.
_ARM_HINTS = ("cortex-", "neoverse", "snapdragon", "apple m", "ampere")
_X86_HINTS = ("intel", "amd", "ryzen", "epyc", "athlon", "xeon", "pentium", "celeron")


def _isa_from_cpu_name(cpu: str) -> Optional[str]:
    """Which ISA a CPU name implies, or None if the name does not say."""
    low = cpu.lower()
    if any(h in low for h in _ARM_HINTS):
        return "aarch64"
    if any(h in low for h in _X86_HINTS):
        return "x86_64"
    return None


# --------------------------------------------------------------------------- arm

def _arm_parts() -> List[int]:
    """Distinct CPU part ids in /proc/cpuinfo, in the order the kernel lists them."""
    try:
        with open("/proc/cpuinfo", "r", errors="replace") as f:
            text = f.read()
    except OSError:
        return []
    out: List[int] = []
    for m in re.finditer(r"^CPU part\s*:\s*(0x[0-9a-fA-F]+)", text, re.MULTILINE):
        val = int(m.group(1), 16)
        if val not in out:
            out.append(val)
    return out


def _board_model() -> Optional[str]:
    for path in ("/proc/device-tree/model", "/sys/firmware/devicetree/base/model"):
        try:
            with open(path, "rb") as f:
                return f.read().decode("utf-8", errors="replace").strip("\x00 \n")
        except OSError:
            continue
    return None


# --------------------------------------------------------------------------- x86

def _intel_hybrid_from_sysfs() -> bool:
    """Linux exposes a hybrid part as two PMUs. Cheapest reliable check there is."""
    return os.path.isdir("/sys/devices/cpu_core") and os.path.isdir("/sys/devices/cpu_atom")


def _intel_generation(cpu: str) -> Optional[int]:
    m = _INTEL_GEN_RE.search(cpu)
    if m:
        return int(m.group(1))
    m = _INTEL_SKU_RE.search(cpu)
    if m:
        digits = m.group(1)
        # 12700H -> 12, 9750H -> 9. Five digits mean a two-digit generation.
        return int(digits[:2]) if len(digits) == 5 else int(digits[0])
    return None


# --------------------------------------------------------------------------- detect

def detect(cpu_override: Optional[str] = None) -> Dict[str, Any]:
    """Describe the decoding silicon.

    Returns a dict with ``cores``: a list of {core, label, confidence, why}, most
    likely first. An empty list means nothing in the matrix applies.
    """
    system = sysinfo.collect()
    cpu = cpu_override or system.get("cpu") or ""
    if cpu_override:
        # An override names a CPU that is deliberately not this one, so every probe of
        # the running system below would be answering about the wrong machine. Take the
        # ISA from the name and read nothing off the host.
        machine = _isa_from_cpu_name(cpu_override) or ""
        board = None
    else:
        machine = (system.get("machine") or platform.machine() or "").lower()
        board = _board_model() if platform.system() == "Linux" else None

    out: Dict[str, Any] = {
        "cpu": cpu,
        "board": board,
        "isa": machine,
        "logical_cores": system.get("logical_cores"),
        "physical_cores": system.get("physical_cores"),
        "ram_bytes": system.get("ram_bytes"),
        "os": system.get("os"),
        "cores": [],
        "hybrid": False,
        "detail": None,
    }

    if machine in ("aarch64", "arm64"):
        if cpu_override:
            # No /proc/cpuinfo to consult for a machine we are only being told about, so
            # the core names have to come out of the name itself. Bounded so that asking
            # about a Cortex-A720 does not also match the Cortex-A72.
            names = [n for n in dict.fromkeys(ARM_PARTS.values())
                     if re.search(rf"\b{re.escape(n)}\b", cpu, re.IGNORECASE)]
        else:
            names = [ARM_PARTS.get(p, f"Arm part {p:#x}") for p in _arm_parts()]
        out["detail"] = ", ".join(names) if names else None
        out["hybrid"] = len(names) > 1
        if board and "raspberry pi 5" in board.lower():
            out["cores"] = [{
                "core": "cortex-a76", "label": "Arm Cortex-A76 (Raspberry Pi 5)",
                "confidence": "exact",
                "why": "this is the board the matrix was measured on",
            }]
            return out
        if "Cortex-A76" in names:
            out["cores"] = [{
                "core": "cortex-a76", "label": "Arm Cortex-A76",
                "confidence": "same-uarch",
                "why": "same core design as the measured Pi 5, but a different SoC, so "
                       "the memory system underneath it is not the one measured",
            }]
            return out
        for name in names:
            if name in A76_RELATIVES:
                out["cores"] = [{
                    "core": "cortex-a76", "label": name,
                    "confidence": "related",
                    "why": f"{A76_RELATIVES[name]}; nothing on this core was measured, so "
                           f"the A76 numbers are an extrapolation",
                }]
                return out
        return out

    if machine in ("x86_64", "amd64"):
        gen = _intel_generation(cpu)
        is_intel = "intel" in cpu.lower()
        hybrid = (not cpu_override and _intel_hybrid_from_sysfs()) or bool(
            is_intel and ((gen or 0) >= 12 or _CORE_ULTRA_RE.search(cpu)))
        out["hybrid"] = hybrid
        if not is_intel:
            return out
        if _CORE_ULTRA_RE.search(cpu):
            # Meteor Lake and later: Redwood Cove or Lion Cove performance cores with
            # Crestmont or Skymont efficiency cores. Hybrid, but not these hybrids.
            out["detail"] = ("Core Ultra: Redwood Cove or Lion Cove performance cores "
                             "with Crestmont or Skymont efficiency cores, none of them "
                             "measured here")
            return out
        if gen == 12 or (hybrid and gen is None):
            exact = "12700h" in cpu.lower().replace("-", "").replace(" ", "")
            conf = "exact" if exact else "same-uarch"
            why = ("this is the CPU the x86 half of the matrix was measured on"
                   if exact else
                   "same Golden Cove and Gracemont cores as the measured i7-12700H, "
                   "different SKU")
            out["cores"] = [
                {"core": "golden-cove", "label": "Intel Golden Cove (P-core)",
                 "confidence": conf, "why": why},
                {"core": "gracemont", "label": "Intel Gracemont (E-core)",
                 "confidence": conf, "why": why},
            ]
            out["detail"] = ("hybrid: performance and efficiency cores share one memory "
                             "system. Some SKUs in this generation ship without "
                             "efficiency cores, in which case ignore that half.")
            return out
        if gen in (13, 14):
            out["cores"] = [
                {"core": "golden-cove", "label": "Intel Raptor Cove (P-core)",
                 "confidence": "related",
                 "why": "Raptor Cove is a Golden Cove derivative with a larger cache; "
                        "close, but not the core that was measured"},
                {"core": "gracemont", "label": "Intel Gracemont (E-core)",
                 "confidence": "same-uarch",
                 "why": "Raptor Lake ships the same Gracemont E-core that was measured"},
            ]
            out["detail"] = ("hybrid: performance and efficiency cores share one memory "
                             "system. Some SKUs in this generation ship without "
                             "efficiency cores, in which case ignore that half.")
            return out
        return out

    return out


def summary_line(det: Dict[str, Any]) -> str:
    """One line naming the machine, for the top of a report."""
    bits = [det.get("cpu") or "unknown CPU"]
    if det.get("board"):
        bits.append(det["board"])
    cores = det.get("logical_cores")
    if cores:
        phys = det.get("physical_cores")
        bits.append(f"{cores} logical" + (f" / {phys} physical" if phys else "") + " cores")
    ram = det.get("ram_bytes")
    if ram:
        bits.append(f"{ram / 1e9:.1f} GB RAM")
    return ", ".join(bits)
