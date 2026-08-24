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
from typing import Any, Dict, List, Optional

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
        kv, _ = _read_header(f, max_kv)
        return kv


def _read_header(f, max_kv: int = 4096):
    """Read magic, counts and the key/value block. Leaves f at the tensor table."""
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
    return kv, n_tensors


# --------------------------------------------------------------------------- tensors

# ggml type id -> (name, elements per block, bytes per block). A quantized tensor stores
# whole blocks, so its size is elements/block_size * type_size and not a bit count.
# Values follow the block structs in ggml-common.h; a type absent here makes the tensor
# unsized rather than mis-sized.
GGML_TYPES = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110), 12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74), 18: ("IQ3_XXS", 256, 98),
    19: ("IQ1_S", 256, 50), 20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82), 23: ("IQ4_XS", 256, 136),
    24: ("I8", 1, 1), 25: ("I16", 1, 2), 26: ("I32", 1, 4), 27: ("I64", 1, 8),
    28: ("F64", 1, 8), 29: ("IQ1_M", 256, 56), 30: ("BF16", 1, 2),
    34: ("TQ1_0", 256, 54), 35: ("TQ2_0", 256, 66), 39: ("MXFP4", 32, 17),
}


def read_tensors(path: str, max_tensors: int = 100000) -> List[Dict[str, Any]]:
    """The per-tensor type map: name, ggml type, shape and size in bytes.

    This is the part of a GGUF file that actually determines what it costs to run, and
    the part its filename does not describe. Two files carrying the same format label
    can hold different types per tensor depending on what they were quantized from.
    """
    with open(path, "rb") as f:
        _, n_tensors = _read_header(f)
        if n_tensors > max_tensors:
            raise GGUFError(f"implausible tensor count {n_tensors}")
        out: List[Dict[str, Any]] = []
        for _ in range(n_tensors):
            name = _read_string(f)
            (n_dims,) = struct.unpack("<I", _read(f, 4))
            if n_dims > 8:
                raise GGUFError(f"implausible tensor rank {n_dims}")
            dims = list(struct.unpack(f"<{n_dims}Q", _read(f, 8 * n_dims)))
            (tid,) = struct.unpack("<I", _read(f, 4))
            (offset,) = struct.unpack("<Q", _read(f, 8))
            elements = 1
            for d in dims:
                elements *= d
            info = GGML_TYPES.get(tid)
            nbytes = None
            if info and elements % info[1] == 0:
                nbytes = elements // info[1] * info[2]
            out.append({
                "name": name, "type_id": tid,
                "type": info[0] if info else f"type{tid}",
                "dims": dims, "elements": elements,
                "bytes": nbytes, "offset": offset,
            })
        return out


# Tensors read once per token during generation: every repeating block, the final norm,
# and the output head. The token embedding is a row lookup, so only one row of it is
# touched per token, which is why file size overstates decode traffic.
_EMBED_NAMES = ("token_embd.weight",)


def streamed_bytes(tensors: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Bytes read per generated token, and how that splits.

    Returns None if any tensor could not be sized, because a partial total would be
    worse than no total.
    """
    if not tensors or any(t["bytes"] is None for t in tensors):
        return None
    total = sum(t["bytes"] for t in tensors)
    embed = sum(t["bytes"] for t in tensors if t["name"] in _EMBED_NAMES)
    head = sum(t["bytes"] for t in tensors if t["name"].startswith("output.weight"))
    tied = head == 0 and embed > 0
    # A tied model has no separate head: the embedding matrix is the output projection,
    # so it is streamed in full every token and must stay in the total.
    streamed = total if tied else total - embed
    return {
        "total_bytes": total,
        "streamed_bytes": streamed,
        "embedding_bytes": embed,
        "head_bytes": embed if tied else head,
        "tied_embedding": tied,
        "head_share_pct": (100.0 * (embed if tied else head) / streamed) if streamed else None,
        "n_tensors": len(tensors),
    }


def type_map(tensors: List[Dict[str, Any]]) -> Dict[str, int]:
    """How many tensors of each ggml type, biggest contributor first."""
    counts: Dict[str, int] = {}
    for t in tensors:
        counts[t["type"]] = counts.get(t["type"], 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


# Norms and biases are stored in full precision in every recipe. They are a rounding error
# by bytes and including them would drag the effective bit rate towards 32 for reasons that
# have nothing to do with the quantization.
_FULL_PRECISION = ("F32", "F16", "BF16", "F64")


def composition(tensors: List[Dict[str, Any]],
                label: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """How much of a file is actually the type its name claims.

    A GGUF format label names a recipe. The recipe substitutes a different type per tensor
    when a shape does not divide by the block size, and llama-quantize does that silently.
    At small model sizes the substitution can take over the file: a Qwen2.5-0.5B built as
    Q3_K_M contains no Q3_K tensors at all.

    Measured over the repeating layers, because that is where the substitution happens and
    where decode spends its bytes. Returns None if any tensor could not be sized.
    """
    repeating = [t for t in tensors
                 if t["name"].startswith("blk.") and t["type"] not in _FULL_PRECISION]
    if not repeating or any(t["bytes"] is None for t in repeating):
        return None

    total = sum(t["bytes"] for t in repeating)
    elements = sum(t["elements"] for t in repeating)
    by_type: Dict[str, int] = {}
    for t in repeating:
        by_type[t["type"]] = by_type.get(t["type"], 0) + t["bytes"]

    expected = None
    if label:
        label = label.upper()
        known = {name for name, _, _ in GGML_TYPES.values()}
        # Q4_K_M means Q4_K tensors, Q4_0 means Q4_0, IQ4_XS means IQ4_XS. Try the label
        # itself, then the label with its recipe suffix removed. A file can legitimately
        # contain none of the expected type, so resolve against the ggml type names rather
        # than against what happens to be in this file: otherwise Q2_K with no Q2_K
        # tensors reports its expected type as "Q2", which is not a type.
        for candidate in (label, label.rsplit("_", 1)[0]):
            if candidate in known:
                expected = candidate
                break
        if expected is None:
            expected = label

    on_label = by_type.get(expected, 0) if expected else None
    return {
        "expected_type": expected,
        "on_label_bytes": on_label,
        "on_label_pct": (100.0 * on_label / total) if on_label is not None and total else None,
        "repeating_bytes": total,
        "repeating_tensors": len(repeating),
        # What the file actually costs per weight, against what its name implies.
        "bits_per_weight": (8.0 * total / elements) if elements else None,
        "by_type_pct": {k: round(100.0 * v / total, 1)
                        for k, v in sorted(by_type.items(), key=lambda kv: -kv[1])},
        "by_type_bytes": dict(sorted(by_type.items(), key=lambda kv: -kv[1])),
    }


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
