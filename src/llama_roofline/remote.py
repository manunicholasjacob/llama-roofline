"""Read the header of a GGUF file over HTTP without downloading the model.

A GGUF file puts its metadata and its tensor table at the front, and those are all that
``inspect`` needs. HTTP range requests fetch just those bytes, usually a few MB of a file
that may be tens of GB, so a model can be checked before anyone commits to downloading it.

Standard library only, like the rest of the analysis path.

References accepted:

    hf:OWNER/REPO/FILE.gguf              a file on the Hugging Face Hub, main branch
    hf:OWNER/REPO@REVISION/FILE.gguf     a branch, tag or commit
    https://huggingface.co/OWNER/REPO/blob/main/FILE.gguf   (or /resolve/; both work)
    https://any.host/path/model.gguf     any server that honours Range requests

For a gated or private Hub repository, set ``HF_TOKEN``. The token is sent to
huggingface.co only and is not forwarded on the redirect to the file store.
"""

from __future__ import annotations

import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Dict, Optional, Tuple

from . import __version__

HF_HOST = "huggingface.co"
# The first fetch covers the whole header of most models; a 150k-token vocabulary needs
# about 5 MB. Larger headers are read in further blocks of the same size.
BLOCK = 4 * 1024 * 1024
# Refuse to keep reading past this: a header this large means something is wrong.
MAX_HEADER_BYTES = 512 * 1024 * 1024
# Extra attempts after a dropped connection or a 5xx answer.
RETRIES = 2

_HF_REF = re.compile(r"^(?P<owner>[^/@]+)/(?P<repo>[^/@]+)(?:@(?P<rev>[^/]+))?/(?P<file>.+)$")


class RemoteError(Exception):
    """A remote reference that could not be read, with a message fit for the user."""


def is_remote(ref: str) -> bool:
    return ref.startswith("hf:") or ref.startswith("https://") or ref.startswith("http://")


def resolve(ref: str) -> Tuple[str, str]:
    """Turn a reference into (download URL, short display name)."""
    if ref.startswith("hf:"):
        m = _HF_REF.match(ref[3:])
        if not m:
            raise RemoteError(
                f"cannot read {ref!r}: expected hf:OWNER/REPO/FILE.gguf, for example "
                "hf:Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_k_m.gguf")
        rev = m.group("rev") or "main"
        path = "/".join(urllib.parse.quote(p) for p in m.group("file").split("/"))
        url = (f"https://{HF_HOST}/{m.group('owner')}/{m.group('repo')}/resolve/"
               f"{urllib.parse.quote(rev, safe='')}/{path}")
        return url, ref
    parsed = urllib.parse.urlparse(ref)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise RemoteError(f"not a URL: {ref}")
    url = ref
    if parsed.netloc == HF_HOST and "/blob/" in parsed.path:
        # The page a browser shows; /resolve/ is the file itself.
        url = ref.replace("/blob/", "/resolve/", 1)
    return url, os.path.basename(parsed.path) or ref


def hf_repo_url(url: str) -> Optional[str]:
    """For a Hub file URL, the page listing the repository's files."""
    parsed = urllib.parse.urlparse(url)
    if parsed.netloc != HF_HOST:
        return None
    parts = parsed.path.strip("/").split("/")
    if len(parts) >= 4 and parts[2] == "resolve":
        return f"https://{HF_HOST}/{parts[0]}/{parts[1]}/tree/{parts[3]}"
    return None


class RangeReader:
    """A read-only, seekable file object over HTTP range requests.

    Fetches aligned blocks on demand and keeps them, so the parser's many small reads
    cost one request per block, and a seek over a skipped array costs nothing.
    """

    def __init__(self, url: str, token: Optional[str] = None, timeout: float = 30.0,
                 opener=None):
        self.url = url
        self.token = token
        self.timeout = timeout
        self.size: Optional[int] = None
        self.requests = 0
        self.bytes_fetched = 0
        self._pos = 0
        self._blocks: Dict[int, bytes] = {}
        self._open = opener or urllib.request.urlopen

    # -- the parts of the file protocol the GGUF parser uses
    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            self._pos = offset
        elif whence == 1:
            self._pos += offset
        elif whence == 2:
            if self.size is None:
                self._block(0)
            self._pos = (self.size or 0) + offset
        if self._pos < 0:
            raise ValueError("negative seek position")
        return self._pos

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            raise ValueError("RangeReader reads a bounded number of bytes only")
        out = bytearray()
        while n > 0:
            if self.size is not None and self._pos >= self.size:
                break
            if self._pos >= MAX_HEADER_BYTES:
                raise RemoteError(
                    f"read more than {MAX_HEADER_BYTES // (1024 * 1024)} MB without reaching "
                    "the end of the header; this does not look like a GGUF file")
            idx = self._pos // BLOCK
            block = self._block(idx)
            start = self._pos - idx * BLOCK
            chunk = block[start:start + n]
            if not chunk:
                break
            out += chunk
            self._pos += len(chunk)
            n -= len(chunk)
        return bytes(out)

    def close(self) -> None:
        self._blocks.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- HTTP
    def _block(self, idx: int) -> bytes:
        if idx in self._blocks:
            return self._blocks[idx]
        lo = idx * BLOCK
        hi = lo + BLOCK - 1
        if self.size is not None:
            hi = min(hi, self.size - 1)
            if lo > hi:
                return b""
        req = urllib.request.Request(self.url, headers={
            "Range": f"bytes={lo}-{hi}",
            "User-Agent": f"llama-roofline/{__version__}",
        })
        if self.token and urllib.parse.urlparse(self.url).netloc == HF_HOST:
            # Unredirected: the Hub answers with a redirect to its file store, and the
            # token has no business travelling there.
            req.add_unredirected_header("Authorization", f"Bearer {self.token}")
        resp = None
        for attempt in range(RETRIES + 1):
            try:
                resp = self._open(req, timeout=self.timeout)
                break
            except urllib.error.HTTPError as exc:
                if exc.code >= 500 and attempt < RETRIES:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                raise RemoteError(self._http_message(exc.code)) from None
            except (urllib.error.URLError, OSError) as exc:
                # A reset during the TLS handshake is common enough on busy CDNs that one
                # failed connection should not end the run.
                if attempt < RETRIES:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                reason = getattr(exc, "reason", exc)
                raise RemoteError(f"could not reach "
                                  f"{urllib.parse.urlparse(self.url).netloc}: "
                                  f"{reason}") from None
        with resp:
            status = getattr(resp, "status", None) or resp.getcode()
            if status == 416:
                return b""
            if status != 206:
                raise RemoteError(
                    f"the server ignored the byte-range request (HTTP {status}), so the "
                    "header cannot be read without downloading the whole file. Download "
                    "it and run inspect on the local copy")
            total = _total_from_content_range(resp.headers.get("Content-Range", ""))
            if total is not None:
                self.size = total
            data = resp.read(hi - lo + 1)
        self.requests += 1
        self.bytes_fetched += len(data)
        self._blocks[idx] = data
        return data

    def _http_message(self, code: int) -> str:
        where = hf_repo_url(self.url)
        if code in (401, 403):
            if urllib.parse.urlparse(self.url).netloc == HF_HOST:
                hint = ("This repository is gated or private. Accept its terms on the model "
                        "page, then set HF_TOKEN to a read token"
                        + (" (HF_TOKEN is set, but was refused)" if self.token else "")
                        + ".")
            else:
                hint = "The server refused access."
            return f"HTTP {code} for {self.url}. {hint}"
        if code == 404:
            return (f"HTTP 404: no such file at {self.url}."
                    + (f" The files in that repository are listed at {where}" if where
                       else ""))
        return f"HTTP {code} for {self.url}"


def _total_from_content_range(value: str) -> Optional[int]:
    # "bytes 0-4194303/987654321"
    m = re.match(r"^\s*bytes\s+\d+-\d+/(\d+)\s*$", value or "")
    return int(m.group(1)) if m else None


def open_remote(ref: str, timeout: float = 30.0) -> Tuple[RangeReader, str]:
    """A RangeReader for ``ref`` and the name to print for it."""
    url, name = resolve(ref)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    return RangeReader(url, token=token, timeout=timeout), name
