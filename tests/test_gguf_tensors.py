"""Reading the tensor table, which is where a GGUF file's real cost lives.

These build small synthetic GGUF files rather than shipping a model, so the suite still
runs on a machine with no models and no llama.cpp.
"""

import struct

import pytest

from llama_roofline import gguf


def _string(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _kv_string(key: str, value: str) -> bytes:
    return _string(key) + struct.pack("<I", 8) + _string(value)


def _kv_u32(key: str, value: int) -> bytes:
    return _string(key) + struct.pack("<I", 4) + struct.pack("<I", value)


def write_gguf(path, tensors, kv=(("general.architecture", "llama"),), version=3):
    """tensors: list of (name, dims, ggml type id)."""
    body = b"GGUF" + struct.pack("<I", version)
    body += struct.pack("<Q", len(tensors)) + struct.pack("<Q", len(kv))
    for k, v in kv:
        body += _kv_string(k, v) if isinstance(v, str) else _kv_u32(k, v)
    offset = 0
    for name, dims, tid in tensors:
        body += _string(name) + struct.pack("<I", len(dims))
        body += struct.pack(f"<{len(dims)}Q", *dims)
        body += struct.pack("<I", tid) + struct.pack("<Q", offset)
        offset += 1024
    path.write_bytes(body)
    return str(path)


Q4_0, Q6_K, Q8_0, F32 = 2, 14, 8, 0


def test_tensor_sizes_follow_the_block_layout(tmp_path):
    # 256 elements of Q4_0 is 8 blocks of 32, 18 bytes each.
    p = write_gguf(tmp_path / "m.gguf", [("blk.0.attn_q.weight", [256], Q4_0)])
    tensors = gguf.read_tensors(p)
    assert tensors[0]["bytes"] == 8 * 18
    assert tensors[0]["type"] == "Q4_0"


def test_a_tensor_that_does_not_divide_into_blocks_is_left_unsized(tmp_path):
    p = write_gguf(tmp_path / "m.gguf", [("blk.0.w", [30], Q4_0)])
    assert gguf.read_tensors(p)[0]["bytes"] is None


def test_an_unknown_ggml_type_is_named_not_guessed(tmp_path):
    p = write_gguf(tmp_path / "m.gguf", [("blk.0.w", [32], 250)])
    t = gguf.read_tensors(p)[0]
    assert t["type"] == "type250"
    assert t["bytes"] is None


def test_streamed_bytes_excludes_the_embedding_when_there_is_a_head(tmp_path):
    p = write_gguf(tmp_path / "m.gguf", [
        ("token_embd.weight", [32000, 32], Q4_0),
        ("blk.0.attn_q.weight", [32, 32], Q4_0),
        ("output.weight", [32000, 32], Q8_0),
    ])
    s = gguf.streamed_bytes(gguf.read_tensors(p))
    embed = 32000 * 32 // 32 * 18
    head = 32000 * 32 // 32 * 34
    block = 32 * 32 // 32 * 18
    assert s["total_bytes"] == embed + head + block
    assert s["streamed_bytes"] == head + block
    assert s["tied_embedding"] is False


def test_a_tied_model_streams_its_embedding_because_it_is_the_head(tmp_path):
    p = write_gguf(tmp_path / "m.gguf", [
        ("token_embd.weight", [32000, 32], Q8_0),
        ("blk.0.attn_q.weight", [32, 32], Q4_0),
    ])
    s = gguf.streamed_bytes(gguf.read_tensors(p))
    assert s["tied_embedding"] is True
    assert s["streamed_bytes"] == s["total_bytes"]


def test_one_unsized_tensor_makes_the_whole_total_unavailable(tmp_path):
    # A partial byte count would be quietly wrong in the direction that flatters the
    # machine, so there is no total at all rather than a low one.
    p = write_gguf(tmp_path / "m.gguf", [
        ("blk.0.w", [256], Q4_0),
        ("blk.1.w", [30], Q4_0),
    ])
    assert gguf.streamed_bytes(gguf.read_tensors(p)) is None


def test_type_map_counts_by_type(tmp_path):
    p = write_gguf(tmp_path / "m.gguf", [
        ("blk.0.w", [256], Q4_0), ("blk.1.w", [256], Q4_0),
        ("blk.0.norm", [32], F32), ("output.weight", [256], Q6_K),
    ])
    assert gguf.type_map(gguf.read_tensors(p)) == {"Q4_0": 2, "F32": 1, "Q6_K": 1}


def test_metadata_still_reads_after_the_header_refactor(tmp_path):
    p = write_gguf(tmp_path / "m.gguf", [("blk.0.w", [256], Q4_0)],
                   kv=(("general.architecture", "qwen2"),))
    kv = gguf.read_metadata(p)
    assert kv["general.architecture"] == "qwen2"
    assert kv["_n_tensors"] == 1


def test_a_non_gguf_file_raises(tmp_path):
    p = tmp_path / "not.gguf"
    p.write_bytes(b"nope, this is not a model")
    with pytest.raises(gguf.GGUFError):
        gguf.read_tensors(str(p))


def test_an_implausible_tensor_count_is_refused(tmp_path):
    body = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 10 ** 9)
    body += struct.pack("<Q", 0)
    p = tmp_path / "huge.gguf"
    p.write_bytes(body)
    with pytest.raises(gguf.GGUFError):
        gguf.read_tensors(str(p))


def test_every_shipped_ggml_type_has_a_sane_block():
    for tid, (name, block, size) in gguf.GGML_TYPES.items():
        assert block >= 1 and size >= 1, name
        # No quantized format stores more than 8 bits per weight, and none stores less
        # than half a bit; a typo in the table would land outside that.
        bits = 8.0 * size / block
        assert 0.5 <= bits <= 64.0, (name, bits)
