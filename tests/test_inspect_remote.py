"""inspect beyond one local file: remote headers, split models, JSON, newer ggml types.

Nothing here touches the network. Remote reads go through a fake opener that serves a
synthetic GGUF from memory and honours Range headers the way the Hub's file store does.
"""

import email.message
import io
import json
import os
import struct
import urllib.error

import pytest

from llama_roofline import cli, gguf, remote

Q4_0, Q8_0, F32 = 2, 8, 0
NVFP4, Q1_0, Q2_0, Q4_0_4_4 = 40, 41, 42, 31


def _string(s):
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def build_gguf(tensors, kv=(("general.architecture", "llama"),), sizes=None,
               align=32, version=3, extra_kv=()):
    """A complete GGUF with real tensor data, so offsets and file size are honest.

    tensors: (name, dims, type id); sizes: optional per-tensor byte counts for types
    the table under test does not know.
    """
    items = list(kv) + list(extra_kv)
    head = b"GGUF" + struct.pack("<I", version)
    head += struct.pack("<Q", len(tensors)) + struct.pack("<Q", len(items))
    for k, v in items:
        if isinstance(v, str):
            head += _string(k) + struct.pack("<I", 8) + _string(v)
        else:
            head += _string(k) + struct.pack("<I", 4) + struct.pack("<I", v)
    offset = 0
    blobs = []
    for i, (name, dims, tid) in enumerate(tensors):
        n = 1
        for d in dims:
            n *= d
        if sizes and sizes[i] is not None:
            nbytes = sizes[i]
        else:
            _, block, tsize = gguf.GGML_TYPES[tid]
            nbytes = n // block * tsize
        head += _string(name) + struct.pack("<I", len(dims))
        head += struct.pack(f"<{len(dims)}Q", *dims)
        head += struct.pack("<I", tid) + struct.pack("<Q", offset)
        pad = (-nbytes) % align
        blobs.append(b"\x00" * nbytes + b"\x00" * pad)
        offset += nbytes + pad
    head += b"\x00" * ((-len(head)) % align)
    return head + b"".join(blobs)


MODEL = [
    ("token_embd.weight", [64, 128], Q8_0),
    ("blk.0.attn_q.weight", [256, 64], Q4_0),
    ("blk.0.attn_norm.weight", [64], F32),
    ("output.weight", [64, 128], Q8_0),
]
KV = (("general.architecture", "llama"), ("general.file_type", 2))


# ------------------------------------------------------------------ newer ggml types

@pytest.mark.parametrize("tid,name,block,size", [
    (NVFP4, "NVFP4", 64, 36), (Q1_0, "Q1_0", 128, 18), (Q2_0, "Q2_0", 64, 18),
    (Q4_0_4_4, "Q4_0_4_4", 32, 18), (32, "Q4_0_4_8", 32, 18), (33, "Q4_0_8_8", 32, 18),
])
def test_types_added_in_this_release_are_sized(tid, name, block, size):
    # Layouts from ggml-common.h; each was also checked against the tensor offsets of a
    # real file on the Hub before it went in the table.
    assert gguf.GGML_TYPES[tid] == (name, block, size)


def test_new_file_type_labels():
    assert gguf.FTYPE_NAMES[39] == "NVFP4"
    assert gguf.FTYPE_NAMES[40] == "Q1_0"
    assert gguf.FTYPE_NAMES[41] == "Q2_0"
    assert gguf.FTYPE_NAMES[33] == "Q4_0_4_4"


def test_a_known_but_unsized_type_is_named(tmp_path):
    p = tmp_path / "m.gguf"
    p.write_bytes(build_gguf([("blk.0.w", [32], 36)], sizes=[18]))
    t = gguf.read_tensors(str(p))[0]
    assert t["type"] == "IQ4_NL_4_4" and t["bytes"] is None
    assert gguf.unsized_types([t]) == ["IQ4_NL_4_4"]


# ------------------------------------------------------------- what a bad file is

@pytest.mark.parametrize("content,needle", [
    (struct.pack("<Q", 100) + b'{"__metadata__": {}}', "safetensors"),
    (b"<!DOCTYPE html><html>", "web page"),
    (b"version https://git-lfs.github.com/spec/v1\noid sha256:ab", "Git LFS pointer"),
    (b"PK\x03\x04rest", "zip"),
    (b"ggjt\x00\x00\x00\x03", "pre-GGUF"),
    (b"", "empty"),
    (b"GGUF" + struct.pack("<I", 1) + b"\x00" * 16, "version 1"),
    (b"GGUF" + struct.pack("<I", 7) + b"\x00" * 16, "newer"),
])
def test_a_file_that_is_not_gguf_says_what_it_probably_is(tmp_path, content, needle):
    p = tmp_path / "x.gguf"
    p.write_bytes(content)
    with pytest.raises(gguf.GGUFError) as exc:
        gguf.read_tensors(str(p))
    assert needle in str(exc.value)


def test_a_truncated_header_says_so(tmp_path):
    whole = build_gguf(MODEL, kv=KV)
    p = tmp_path / "cut.gguf"
    p.write_bytes(whole[:60])
    with pytest.raises(gguf.GGUFTruncated) as exc:
        gguf.read_tensors(str(p))
    assert "incomplete download" in str(exc.value)


def test_read_gguf_works_on_any_file_object():
    kv, tensors = gguf.read_gguf(io.BytesIO(build_gguf(MODEL, kv=KV)))
    assert kv["general.architecture"] == "llama"
    assert [t["type"] for t in tensors] == ["Q8_0", "Q4_0", "F32", "Q8_0"]
    assert kv["_data_start"] % 32 == 0


# ------------------------------------------------------- sizing from tensor offsets

def test_an_unknown_type_is_sized_from_the_offsets(tmp_path):
    # Type 99 does not exist. Its tensor still sits between two others, so the gap
    # between their offsets says how big it is, to within the alignment padding.
    data = build_gguf([("blk.0.a", [256], Q4_0), ("blk.0.b", [100], 99),
                       ("blk.0.c", [256], Q4_0)], sizes=[None, 70, None])
    kv, tensors = gguf.read_gguf(io.BytesIO(data))
    assert tensors[1]["bytes"] is None
    types = gguf.size_from_offsets(tensors, len(data) - kv["_data_start"])
    assert types == ["type99"]
    assert 70 <= tensors[1]["bytes"] < 70 + 32
    assert tensors[1]["bytes_from_offsets"] is True
    assert tensors[0]["bytes"] == 144  # known sizes are never touched


def test_the_last_tensor_runs_to_the_end_of_the_data(tmp_path):
    data = build_gguf([("blk.0.a", [256], Q4_0), ("blk.0.b", [100], 99)],
                      sizes=[None, 64])
    kv, tensors = gguf.read_gguf(io.BytesIO(data))
    gguf.size_from_offsets(tensors, len(data) - kv["_data_start"])
    assert tensors[1]["bytes"] == 64
    # Without the data size the last tensor stays unsized rather than guessed.
    kv, tensors = gguf.read_gguf(io.BytesIO(data))
    assert gguf.size_from_offsets(tensors, None) == []
    assert tensors[1]["bytes"] is None


def test_inspect_reports_a_type_newer_than_the_tool(tmp_path, capsys):
    p = tmp_path / "future.gguf"
    p.write_bytes(build_gguf([("blk.0.a", [256], Q4_0), ("blk.0.b", [512], 99),
                              ("output.weight", [256], Q8_0)], sizes=[None, 96, None]))
    assert cli.main(["inspect", str(p)]) == 0
    out = capsys.readouterr().out
    assert "type99 is newer than" in out
    assert "per token" in out and "not computable" not in out


# ------------------------------------------------------------------- local, more

def test_inspect_json_is_machine_readable(tmp_path, capsys):
    p = tmp_path / "m-Q4_0.gguf"
    p.write_bytes(build_gguf(MODEL, kv=KV))
    assert cli.main(["inspect", str(p), "--json", "--peak-bw", "50"]) == 0
    doc = json.loads(capsys.readouterr().out)
    m = doc["models"][0]
    assert doc["command"] == "inspect"
    assert m["label"] == "Q4_0" and m["on_label_pct"] == 100.0
    assert m["streamed_bytes"] == 9216 + 256 + 8704  # layer, norm, head; no embedding
    assert abs(m["ceiling_tok_s"] - 50e9 / m["streamed_bytes"]) < 1e-6
    assert "tensors" not in m  # only with --tensors


def test_inspect_prints_the_ceiling_for_a_given_bandwidth(tmp_path, capsys):
    p = tmp_path / "m.gguf"
    p.write_bytes(build_gguf(MODEL, kv=KV))
    cli.main(["inspect", str(p), "--peak-bw", "10"])
    assert "an upper bound on decode" in capsys.readouterr().out


def _write_split(tmp_path, parts):
    stem = "big-Q4_0"
    paths = []
    for i, tensors in enumerate(parts, 1):
        p = tmp_path / f"{stem}-{i:05d}-of-{len(parts):05d}.gguf"
        p.write_bytes(build_gguf(tensors, kv=KV))
        paths.append(str(p))
    return paths


def test_a_split_model_is_read_part_by_part(tmp_path):
    # Each part has its own tensor table. Reading only the first, as 0.2.2 did, described
    # a fraction of the model.
    paths = _write_split(tmp_path, [MODEL[:2], MODEL[2:]])
    r = cli.inspect_model(paths[0])
    assert r["parts"] == 2
    assert r["n_tensors"] == 4
    assert r["file_bytes"] == sum(os.path.getsize(p) for p in paths)
    assert r["head_type"] == "Q8_0"


def test_a_split_model_with_a_missing_part_says_which(tmp_path, capsys):
    paths = _write_split(tmp_path, [MODEL[:2], MODEL[2:]])
    os.remove(paths[1])
    assert cli.main(["inspect", paths[0]]) == 2
    assert "part 2 is missing" in capsys.readouterr().err


# ------------------------------------------------------------------- remote

def test_hub_references_resolve_to_file_urls():
    assert remote.resolve("hf:Qwen/Repo-GGUF/model-q4_k_m.gguf")[0] == (
        "https://huggingface.co/Qwen/Repo-GGUF/resolve/main/model-q4_k_m.gguf")
    assert remote.resolve("hf:a/b@v1.0/sub/dir/m.gguf")[0] == (
        "https://huggingface.co/a/b/resolve/v1.0/sub/dir/m.gguf")
    # The link a browser shows is a page; the tool fetches the file it stands for.
    assert remote.resolve("https://huggingface.co/a/b/blob/main/m.gguf")[0] == (
        "https://huggingface.co/a/b/resolve/main/m.gguf")
    with pytest.raises(remote.RemoteError):
        remote.resolve("hf:just-a-name")
    assert remote.is_remote("hf:a/b/c.gguf") and not remote.is_remote("model.gguf")


class _Resp(io.BytesIO):
    def __init__(self, data, status, headers):
        super().__init__(data)
        self.status = status
        self.headers = headers

    def getcode(self):
        return self.status


class FakeServer:
    """Serves files from memory and honours ``Range: bytes=a-b`` like a file store."""

    def __init__(self, files, honour_range=True):
        self.files = files
        self.honour_range = honour_range
        self.requests = []

    def __call__(self, req, timeout=None):
        self.requests.append(req)
        name = req.full_url.rsplit("/", 1)[-1]
        if name not in self.files:
            raise urllib.error.HTTPError(req.full_url, 404, "Not Found",
                                         email.message.Message(), None)
        data = self.files[name]
        headers = email.message.Message()
        if not self.honour_range:
            return _Resp(data, 200, headers)
        lo, hi = (int(x) for x in req.get_header("Range")[len("bytes="):].split("-"))
        if lo >= len(data):
            return _Resp(b"", 416, headers)
        hi = min(hi, len(data) - 1)
        headers["Content-Range"] = f"bytes {lo}-{hi}/{len(data)}"
        return _Resp(data[lo:hi + 1], 206, headers)


def _patch_open(monkeypatch, server):
    orig = remote.RangeReader.__init__

    def init(self, url, token=None, timeout=30.0, opener=None):
        orig(self, url, token=token, timeout=timeout, opener=server)
    monkeypatch.setattr(remote.RangeReader, "__init__", init)


def test_remote_inspect_matches_the_local_file(tmp_path, monkeypatch):
    data = build_gguf(MODEL, kv=KV)
    local = tmp_path / "m.gguf"
    local.write_bytes(data)
    server = FakeServer({"m.gguf": data})
    _patch_open(monkeypatch, server)
    r_remote = cli.inspect_model("hf:o/r/m.gguf")
    r_local = cli.inspect_model(str(local))
    for key in ("label", "streamed_bytes", "head_bytes", "on_label_pct", "by_type",
                "file_bytes", "n_tensors"):
        assert r_remote[key] == r_local[key], key
    assert r_remote["header_bytes_fetched"] == len(data)  # a tiny file is one block


def test_remote_reads_only_the_header_of_a_big_file(monkeypatch):
    # Pad the data section to 3 blocks: the parser must stop after the first one.
    data = build_gguf(MODEL, kv=KV) + b"\x00" * (3 * remote.BLOCK)
    server = FakeServer({"m.gguf": data})
    _patch_open(monkeypatch, server)
    r = cli.inspect_model("https://example.org/models/m.gguf")
    assert len(server.requests) == 1
    assert r["header_bytes_fetched"] == remote.BLOCK
    assert r["file_bytes"] == len(data)


def test_remote_split_model_reads_every_part(monkeypatch):
    files = {
        "big-00001-of-00002.gguf": build_gguf(MODEL[:2], kv=KV),
        "big-00002-of-00002.gguf": build_gguf(MODEL[2:], kv=KV),
    }
    _patch_open(monkeypatch, FakeServer(files))
    r = cli.inspect_model("hf:o/r/big-00001-of-00002.gguf")
    assert r["parts"] == 2 and r["n_tensors"] == 4
    assert r["file_bytes"] == sum(len(v) for v in files.values())


def test_remote_404_points_at_the_repository_listing(monkeypatch, capsys):
    _patch_open(monkeypatch, FakeServer({}))
    assert cli.main(["inspect", "hf:owner/repo/missing.gguf"]) == 2
    err = capsys.readouterr().err
    assert "404" in err and "https://huggingface.co/owner/repo/tree/main" in err


def test_a_server_that_ignores_ranges_is_refused_not_downloaded(monkeypatch, capsys):
    _patch_open(monkeypatch, FakeServer({"m.gguf": build_gguf(MODEL, kv=KV)},
                                        honour_range=False))
    assert cli.main(["inspect", "https://example.org/m.gguf"]) == 2
    assert "ignored the byte-range request" in capsys.readouterr().err


def test_gated_repository_message_mentions_the_token(monkeypatch):
    def deny(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized",
                                     email.message.Message(), None)
    reader = remote.RangeReader("https://huggingface.co/o/r/resolve/main/m.gguf",
                                opener=deny)
    with pytest.raises(remote.RemoteError) as exc:
        reader.read(4)
    assert "HF_TOKEN" in str(exc.value)


def test_the_token_goes_to_the_hub_only_and_never_through_a_redirect():
    seen = []

    def capture(req, timeout=None):
        seen.append(req)
        raise urllib.error.HTTPError(req.full_url, 404, "x", email.message.Message(), None)

    hub = remote.RangeReader("https://huggingface.co/o/r/resolve/main/m.gguf",
                             token="hf_secret", opener=capture)
    with pytest.raises(remote.RemoteError):
        hub.read(4)
    # Unredirected headers are not copied onto the request a redirect produces.
    assert seen[0].unredirected_hdrs.get("Authorization") == "Bearer hf_secret"
    assert "Authorization" not in seen[0].headers

    other = remote.RangeReader("https://example.org/m.gguf", token="hf_secret",
                               opener=capture)
    with pytest.raises(remote.RemoteError):
        other.read(4)
    assert "Authorization" not in seen[1].unredirected_hdrs
    assert "Authorization" not in seen[1].headers


def test_a_dropped_connection_is_retried(monkeypatch):
    monkeypatch.setattr(remote.time, "sleep", lambda s: None)
    data = build_gguf(MODEL, kv=KV)
    good = FakeServer({"m.gguf": data})
    calls = {"n": 0}

    def flaky(req, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.URLError(ConnectionResetError(10054, "reset"))
        return good(req, timeout)

    reader = remote.RangeReader("https://example.org/m.gguf", opener=flaky)
    assert reader.read(4) == b"GGUF"
    assert calls["n"] == 2
