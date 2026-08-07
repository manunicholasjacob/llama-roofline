"""GGUF parsing tests. We synthesise minimal GGUF headers rather than shipping a model."""

import struct

import pytest

from llama_roofline import gguf

U32, U64, STR, ARRAY = 4, 10, 8, 9


def _str(s: bytes) -> bytes:
    return struct.pack("<Q", len(s)) + s


def build_gguf(kvs, version=3, n_tensors=1) -> bytes:
    """kvs: list of (key, type, packed_value_bytes)."""
    out = b"GGUF" + struct.pack("<I", version)
    out += struct.pack("<Q", n_tensors) + struct.pack("<Q", len(kvs))
    for key, vtype, payload in kvs:
        out += _str(key.encode()) + struct.pack("<I", vtype) + payload
    return out


def write(tmp_path, name, data):
    p = tmp_path / name
    p.write_bytes(data)
    return str(p)


def test_describe_dense_model(tmp_path):
    data = build_gguf([
        ("general.architecture", STR, _str(b"qwen2")),
        ("general.name", STR, _str(b"Qwen2.5 0.5B Instruct")),
        ("general.file_type", U32, struct.pack("<I", 15)),   # Q4_K_M
        ("qwen2.block_count", U32, struct.pack("<I", 24)),
        ("qwen2.context_length", U32, struct.pack("<I", 32768)),
    ])
    info = gguf.describe(write(tmp_path, "qwen.gguf", data))
    assert info["metadata_ok"] is True
    assert info["arch"] == "qwen2"
    assert info["name"] == "Qwen2.5 0.5B Instruct"
    assert info["quant"] == "Q4_K_M"
    assert info["context_length"] == 32768
    assert info["n_layer"] == 24
    assert info["is_moe"] is False
    assert info["file_bytes"] == len(data)


def test_describe_detects_moe(tmp_path):
    data = build_gguf([
        ("general.architecture", STR, _str(b"qwen3moe")),
        ("general.file_type", U32, struct.pack("<I", 15)),
        ("qwen3moe.expert_count", U32, struct.pack("<I", 128)),
        ("qwen3moe.expert_used_count", U32, struct.pack("<I", 8)),
    ])
    info = gguf.describe(write(tmp_path, "moe.gguf", data))
    assert info["is_moe"] is True
    assert info["n_expert"] == 128
    assert info["n_expert_used"] == 8


def test_array_values_are_skipped_not_materialised(tmp_path):
    """A 150k-token vocab must not be read into memory just to learn the architecture."""
    vocab = struct.pack("<I", U32) + struct.pack("<Q", 150000) + b"\x00" * (4 * 150000)
    data = build_gguf([
        ("tokenizer.ggml.token_type", ARRAY, vocab),
        ("general.architecture", STR, _str(b"llama")),
        ("general.file_type", U32, struct.pack("<I", 7)),   # Q8_0
    ])
    info = gguf.describe(write(tmp_path, "vocab.gguf", data))
    assert info["arch"] == "llama"
    assert info["quant"] == "Q8_0"


def test_string_array_is_walked(tmp_path):
    arr = struct.pack("<I", STR) + struct.pack("<Q", 3) + _str(b"a") + _str(b"bb") + _str(b"ccc")
    data = build_gguf([
        ("tokenizer.ggml.tokens", ARRAY, arr),
        ("general.architecture", STR, _str(b"gemma")),
    ])
    info = gguf.describe(write(tmp_path, "toks.gguf", data))
    assert info["arch"] == "gemma"


def test_unknown_ftype_is_labelled_not_dropped(tmp_path):
    data = build_gguf([("general.file_type", U32, struct.pack("<I", 199))])
    info = gguf.describe(write(tmp_path, "weird.gguf", data))
    assert info["quant"] == "ftype199"


def test_bad_magic_falls_back_to_filename(tmp_path):
    p = write(tmp_path, "mymodel-Q5_K_M.gguf", b"NOTGGUF" + b"\x00" * 64)
    info = gguf.describe(p)
    assert info["metadata_ok"] is False
    assert info["metadata_error"]
    assert info["quant"] == "Q5_K_M"          # recovered from the filename
    assert info["file_bytes"] == 71


def test_unsupported_version_is_not_fatal(tmp_path):
    data = b"GGUF" + struct.pack("<I", 1) + struct.pack("<Q", 0) + struct.pack("<Q", 0)
    info = gguf.describe(write(tmp_path, "old-IQ4_XS.gguf", data))
    assert info["metadata_ok"] is False
    assert info["quant"] == "IQ4_XS"


def test_missing_file_is_not_fatal(tmp_path):
    info = gguf.describe(str(tmp_path / "absent.gguf"))
    assert info["metadata_ok"] is False
    assert info["file_bytes"] is None


def test_implausible_kv_count_rejected(tmp_path):
    data = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 1) + struct.pack("<Q", 10 ** 9)
    with pytest.raises(gguf.GGUFError):
        gguf.read_metadata(write(tmp_path, "huge.gguf", data))


@pytest.mark.parametrize("fname,expected", [
    ("Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf", "Q4_K_M"),
    ("model.f16.gguf", "F16"),
    ("tiny-IQ2_XXS.gguf", "IQ2_XXS"),
    ("no-quant-here.gguf", None),
])
def test_filename_quant_fallback(fname, expected):
    assert gguf._quant_from_name(fname) == expected
