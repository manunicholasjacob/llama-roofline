"""Malformed headers end in GGUFError, never in an exception the callers do not catch.

``describe``, ``read_tensors``, ``inspect`` and the browser inspector catch GGUFError
(and OSError, struct.error). A corrupt array count used to reach ``f.seek`` unchecked,
which raises ValueError on a local file and OverflowError on a BytesIO, and on a
RangeReader silently put the data section exabytes past the end of the file.
"""

import email.message
import io
import struct

import pytest

from llama_roofline import cli, gguf, remote

ARRAY, U64, F32 = 9, 10, 6


def _s(x):
    b = x.encode()
    return struct.pack("<Q", len(b)) + b


def bad_array(count, etype=U64, trailing_kv=False):
    """GGUF v3, 0 tensors: an ARRAY of ``etype`` claiming ``count`` elements and holding none.

    With ``trailing_kv`` a second key follows the array, so the next read finds the end.
    """
    n_kv = 2 if trailing_kv else 1
    data = (b"GGUF" + struct.pack("<IQQ", 3, 0, n_kv) + _s("tokenizer.ggml.scores")
            + struct.pack("<I", ARRAY) + struct.pack("<IQ", etype, count))
    if trailing_kv:
        data += _s("general.architecture") + struct.pack("<I", 8) + _s("llama")
    return data


class _Resp(io.BytesIO):
    def __init__(self, data, headers):
        super().__init__(data)
        self.status = 206
        self.headers = headers


def serve(data):
    """A fake opener: answers every Range request from ``data``, like a file store."""
    def opener(req, timeout=None):
        lo, hi = (int(x) for x in req.get_header("Range")[len("bytes="):].split("-"))
        hi = min(hi, len(data) - 1)
        headers = email.message.Message()
        headers["Content-Range"] = f"bytes {lo}-{hi}/{len(data)}"
        return _Resp(data[lo:hi + 1], headers)
    return opener


HUGE = 2 ** 62           # 8 * 2**62 bytes: past what seek() accepts on any file object
PAST_END = 2 ** 20       # a seek that succeeds but lands past the end of a tiny file


def _write(tmp_path, data):
    p = tmp_path / "bad.gguf"
    p.write_bytes(data)
    return str(p)


def test_describe_reports_a_huge_array_count(tmp_path):
    out = gguf.describe(_write(tmp_path, bad_array(HUGE)))  # used to raise ValueError
    assert out["metadata_ok"] is False
    assert "implausible array length" in out["metadata_error"]


def test_read_tensors_raises_gguf_error_for_a_huge_array_count(tmp_path):
    with pytest.raises(gguf.GGUFError, match="implausible array length"):
        gguf.read_tensors(_write(tmp_path, bad_array(HUGE)))


def test_bytesio_huge_array_count_is_a_gguf_error():
    # The browser inspector parses a BytesIO; this used to raise OverflowError.
    with pytest.raises(gguf.GGUFError, match="implausible array length"):
        gguf.read_gguf(io.BytesIO(bad_array(HUGE)))


class _Ssize32BytesIO(io.BytesIO):
    """BytesIO as Pyodide (wasm32) builds it: positions are a 32-bit ssize_t."""

    def seek(self, pos, whence=0):
        target = pos + (self.tell() if whence == 1 else 0)
        if target > 2 ** 31 - 1:
            raise OverflowError("Python int too large to convert to C ssize_t")
        return super().seek(pos, whence)


def test_seek_overflow_below_the_bound_is_still_a_gguf_error():
    # 2**30 float64s = 8 GiB: under the 2**62 bound, over a 32-bit ssize_t.
    with pytest.raises(gguf.GGUFError, match="implausible array length"):
        gguf.read_gguf(_Ssize32BytesIO(bad_array(2 ** 30, etype=12)))


@pytest.mark.parametrize("trailing_kv", [False, True])
def test_skip_past_the_end_is_truncation(tmp_path, trailing_kv):
    # The last key's array runs past the end of the file. Before, with nothing read
    # after it, the parse "succeeded" with a data section beyond the end of the file.
    data = bad_array(PAST_END, etype=F32, trailing_kv=trailing_kv)
    with pytest.raises(gguf.GGUFTruncated):
        gguf.read_tensors(_write(tmp_path, data))
    with pytest.raises(gguf.GGUFTruncated):
        gguf.read_gguf(io.BytesIO(data))
    assert "ends inside its own header" in gguf.describe(_write(tmp_path, data))["metadata_error"]


def test_valid_skipped_array_at_the_end_still_parses():
    # The end-of-header check must not reject an array that ends exactly at EOF.
    data = bad_array(3, etype=F32) + struct.pack("<3f", 0.0, 1.0, 2.0)
    kv, tensors = gguf.read_gguf(io.BytesIO(data))
    assert kv["tokenizer.ggml.scores"]["len"] == 3
    assert tensors == []
    assert kv["_data_start"] == -(-len(data) // 32) * 32


@pytest.mark.parametrize("count,exc", [(HUGE, gguf.GGUFError), (PAST_END, gguf.GGUFTruncated)])
def test_range_reader_takes_the_same_path(count, exc):
    data = bad_array(count, etype=F32 if count == PAST_END else U64)
    reader = remote.RangeReader("https://example.org/bad.gguf", opener=serve(data))
    with pytest.raises(exc):
        gguf.read_gguf(reader)


def test_remote_inspect_reports_a_clean_error(monkeypatch):
    orig = remote.RangeReader.__init__

    def init(self, url, token=None, timeout=30.0, opener=None):
        orig(self, url, token=token, timeout=timeout, opener=serve(bad_array(HUGE)))
    monkeypatch.setattr(remote.RangeReader, "__init__", init)
    with pytest.raises(cli.InspectError, match="implausible array length"):
        cli.inspect_model("https://example.org/bad.gguf")


def test_local_inspect_reports_a_clean_error(tmp_path):
    with pytest.raises(cli.InspectError, match="implausible array length"):
        cli.inspect_model(_write(tmp_path, bad_array(HUGE)))
