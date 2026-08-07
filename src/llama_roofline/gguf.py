"""Minimal read-only GGUF metadata parser.

We only need the header key/value block -- enough to know the architecture, the
quantization type, the parameter count and (critically) whether the model is a
mixture-of-experts, which changes the bytes-per-token assumption the roofline rests on.
Deliberately dependency-free and defensive: a parse failure is never fatal, the caller
falls back to the file size and to ``llama-bench``'s own metadata.

Format reference: ggml/docs/gguf.md in ggerganov/llama.cpp (GGUF v2/v3).
"""

from __future__ import annotations

import os
import re
import struct
from typing import Any, Dict, Optional

MAGIC = b"GGUF"

# GGUF metadata value type enum.
_UINT8, _INT8, _UINT16, _INT16, _UINT32, _INT32 = 0, 1, 2, 3, 4, 5
_FLOAT32, _BOOL, _STRING, _ARRAY, _UINT64, _INT64, _FLOAT64 = 6, 7, 8, 9, 10, 11, 12

_SCALAR_FMT = {
    _UINT8: "<B", _INT8: "<b", _UINT16: "<H", _INT16: "<h",
    _UINT32: "<I", _INT32: "<i", _FLOAT32: "<f", _BOOL: "<?",
    _UINT64: "<Q", _INT64: "<q", _FLOAT64: "<d",
}

# llama.cpp's LLAMA_FTYPE_* enum -> human-readable quant label. Values not listed here
# are reported as ``ftype<N>``; llama-bench's model_type string is the richer source.
FTYPE_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1",
    10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M",
    16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K", 19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S",
    22: "IQ3_XS", 23: "IQ3_XXS", 24: "IQ1_S", 25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M",
    28: "IQ2_S", 29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16",
    36: "TQ1_0", 37: "TQ2_0", 38: "MXFP4_MOE",
}

# Fallback: pull a quant label out of the filename when metadata is unavailable.
_QUANT_RE = re.compile(
    r"(?<![A-Za-z0-9])((?:I?Q\d(?:_[A-Z0-9]+)*)|F16|BF16|F32|MXFP4(?:_MOE)?|TQ\d_\d)(?![A-Za-z0-9])",
    re.IGNORECASE,
)


class GGUFError(Exception):
    """Raised when a file is not parseable as GGUF."""


def _read(f, n: int) -> bytes:
    b = f.read(n)
    if len(b) != n:
        raise GGUFError("unexpected end of file")
    return b


def _read_scalar(f, vtype: int) -> Any:
    fmt = _SCALAR_FMT.get(vtype)
    if fmt is None:
        raise GGUFError(f"unknown value type {vtype}")
    return struct.unpack(fmt, _read(f, struct.calcsize(fmt)))[0]


def _read_string(f) -> str:
    (n,) = struct.unpack("<Q", _read(f, 8))
    if n > 64 * 1024 * 1024:  # a sane bound; guards against a mis-seek
        raise GGUFError("implausible string length")
    return _read(f, n).decode("utf-8", errors="replace")


def _read_value(f, vtype: int, _depth: int = 0) -> Any:
    if vtype == _STRING:
        return _read_string(f)
    if vtype == _ARRAY:
        if _depth > 1:
            raise GGUFError("nested arrays too deep")
        (etype,) = struct.unpack("<I", _read(f, 4))
        (count,) = struct.unpack("<Q", _read(f, 8))
        # Token vocabularies are huge and useless to us: skip past them cheaply.
        if etype in _SCALAR_FMT:
            f.seek(struct.calcsize(_SCALAR_FMT[etype]) * count, os.SEEK_CUR)
            return {"_array": True, "type": etype, "len": count, "skipped": True}
        for _ in range(count):
            _read_value(f, etype, _depth + 1)
        return {"_array": True, "type": etype, "len": count, "skipped": True}
    return _read_scalar(f, vtype)


def read_metadata(path: str, max_kv: int = 4096) -> Dict[str, Any]:
    """Return the raw GGUF key/value metadata block. Raises GGUFError on bad input."""
    with open(path, "rb") as f:
        if _read(f, 4) != MAGIC:
            raise GGUFError("not a GGUF file (bad magic)")
        (version,) = struct.unpack("<I", _read(f, 4))
        if version not in (2, 3):
            raise GGUFError(f"unsupported GGUF version {version}")
        (n_tensors,) = struct.unpack("<Q", _read(f, 8))
        (n_kv,) = struct.unpack("<Q", _read(f, 8))
        if n_kv > max_kv:
            raise GGUFError(f"implausible metadata count {n_kv}")
        kv: Dict[str, Any] = {"_gguf_version": version, "_n_tensors": n_tensors}
        for _ in range(n_kv):
            key = _read_string(f)
            (vtype,) = struct.unpack("<I", _read(f, 4))
            kv[key] = _read_value(f, vtype)
        return kv


def _quant_from_name(path: str) -> Optional[str]:
    m = _QUANT_RE.search(os.path.basename(path))
    return m.group(1).upper() if m else None


def describe(path: str) -> Dict[str, Any]:
    """Best-effort description of a GGUF file. Never raises.

    Keys: path, name, file_bytes, arch, quant, n_params, n_expert, n_expert_used,
    context_length, is_moe, metadata_ok, metadata_error.
    """
    info: Dict[str, Any] = {
        "path": os.path.abspath(path),
        "name": os.path.splitext(os.path.basename(path))[0],
        "file_bytes": None,
        "arch": None,
        "quant": _quant_from_name(path),
        "n_params": None,
        "n_expert": None,
        "n_expert_used": None,
        "context_length": None,
        "is_moe": False,
        "metadata_ok": False,
        "metadata_error": None,
    }
    try:
        info["file_bytes"] = os.path.getsize(path)
    except OSError as exc:
        info["metadata_error"] = str(exc)
        return info

    try:
        kv = read_metadata(path)
    except (GGUFError, OSError, struct.error) as exc:
        info["metadata_error"] = str(exc)
        return info

    info["metadata_ok"] = True
    arch = kv.get("general.architecture")
    if isinstance(arch, str):
        info["arch"] = arch
    if isinstance(kv.get("general.name"), str):
        info["name"] = kv["general.name"]

    ftype = kv.get("general.file_type")
    if isinstance(ftype, int):
        info["quant"] = FTYPE_NAMES.get(ftype, f"ftype{ftype}")
    if isinstance(kv.get("general.size_label"), str):
        info["size_label"] = kv["general.size_label"]

    if arch:
        for key, dest in (
            (f"{arch}.expert_count", "n_expert"),
            (f"{arch}.expert_used_count", "n_expert_used"),
            (f"{arch}.context_length", "context_length"),
            (f"{arch}.block_count", "n_layer"),
            (f"{arch}.embedding_length", "n_embd"),
        ):
            val = kv.get(key)
            if isinstance(val, int):
                info[dest] = val

    info["is_moe"] = bool(info.get("n_expert"))
    return info
