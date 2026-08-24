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


# ------------------------------------------------- what the label does and does not say

Q4_K, Q5_0, Q6_K, IQ4_NL = 12, 6, 14, 20


def test_composition_measures_the_repeating_layers_only(tmp_path):
    # The output head is chosen by the recipe and is routinely a different type on
    # purpose, so counting it would report every honest file as mislabelled.
    p = write_gguf(tmp_path / "m.gguf", [
        ("blk.0.attn_q.weight", [256], Q4_0),
        ("blk.1.attn_q.weight", [256], Q4_0),
        ("output.weight", [2560], Q8_0),
        ("blk.0.attn_norm.weight", [256], F32),
    ])
    comp = gguf.composition(gguf.read_tensors(p), "Q4_0")
    assert comp["on_label_pct"] == 100.0
    assert comp["repeating_tensors"] == 2


def test_a_recipe_that_substituted_everything_reports_zero(tmp_path):
    # A Qwen2.5-0.5B built as Q3_K_M really does contain no Q3_K tensors. Reporting that
    # as anything other than zero would hide the finding the tool exists to surface.
    p = write_gguf(tmp_path / "m.gguf", [
        ("blk.0.w", [256], Q4_0),
        ("blk.1.w", [256], Q5_0),
    ])
    comp = gguf.composition(gguf.read_tensors(p), "Q3_K_M")
    assert comp["expected_type"] == "Q3_K"
    assert comp["on_label_pct"] == 0.0


def test_the_multi_letter_suffix_resolves_to_the_type_it_means(tmp_path):
    p = write_gguf(tmp_path / "m.gguf", [("blk.0.w", [256], Q4_K)])
    assert gguf.composition(gguf.read_tensors(p), "Q4_K_M")["expected_type"] == "Q4_K"
    p2 = write_gguf(tmp_path / "n.gguf", [("blk.0.w", [256], IQ4_NL)])
    assert gguf.composition(gguf.read_tensors(p2), "IQ4_NL")["expected_type"] == "IQ4_NL"


def test_bits_per_weight_is_what_the_file_stores_not_what_it_claims(tmp_path):
    # 256 Q4_0 elements is 8 blocks of 18 bytes, which is 4.5 bits per weight. Swap half
    # of them for Q8_0 and the file stores far more than its name implies.
    honest = write_gguf(tmp_path / "a.gguf", [("blk.0.w", [256], Q4_0)])
    assert abs(gguf.composition(gguf.read_tensors(honest), "Q4_0")["bits_per_weight"]
               - 4.5) < 0.01

    inflated = write_gguf(tmp_path / "b.gguf", [
        ("blk.0.w", [256], Q4_0), ("blk.1.w", [256], Q8_0)])
    comp = gguf.composition(gguf.read_tensors(inflated), "Q4_0")
    assert comp["bits_per_weight"] > 6.0
    assert comp["on_label_pct"] < 40


def test_full_precision_tensors_do_not_drag_the_bit_rate(tmp_path):
    # Norms are F32 in every recipe and are a rounding error by bytes. Including them
    # would push every file towards 32 bits per weight for no useful reason.
    p = write_gguf(tmp_path / "m.gguf", [
        ("blk.0.w", [256], Q4_0),
        ("blk.0.attn_norm.weight", [8], F32),
    ])
    comp = gguf.composition(gguf.read_tensors(p), "Q4_0")
    assert abs(comp["bits_per_weight"] - 4.5) < 0.01


def test_a_file_with_no_repeating_layers_has_no_composition(tmp_path):
    p = write_gguf(tmp_path / "m.gguf", [("token_embd.weight", [256], Q4_0)])
    assert gguf.composition(gguf.read_tensors(p), "Q4_0") is None


def test_an_unsized_tensor_makes_the_composition_unavailable(tmp_path):
    p = write_gguf(tmp_path / "m.gguf", [
        ("blk.0.w", [256], Q4_0),
        ("blk.1.w", [30], Q4_0),
    ])
    assert gguf.composition(gguf.read_tensors(p), "Q4_0") is None


def test_a_format_the_file_contains_none_of_still_names_a_real_type(tmp_path):
    # Q2_K with no Q2_K tensors must report Q2_K as the expected type, not "Q2", which
    # is not a ggml type and would read as a parser failure rather than as the finding.
    p = write_gguf(tmp_path / "m.gguf", [
        ("blk.0.w", [256], Q4_0), ("blk.1.w", [256], Q5_0)])
    comp = gguf.composition(gguf.read_tensors(p), "Q2_K")
    assert comp["expected_type"] == "Q2_K"
    assert comp["on_label_pct"] == 0.0
