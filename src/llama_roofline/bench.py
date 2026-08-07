"""Thin, defensive wrapper around llama.cpp's ``llama-bench``.

We shell out to the user's own build rather than binding libllama: llama-bench is the
one measurement path in llama.cpp that is non-interactive, reports prefill and decode
separately, and already does warmup + repetitions. Anything else (llama-cli in
particular) can block on interactive input and is unusable from a script.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from typing import Any, Dict, List, Optional

BINARY_NAMES = ["llama-bench", "llama-bench.exe"]

# Places a llama.cpp build commonly lands, relative to a checkout or install prefix.
_SEARCH_SUBDIRS = [
    "", "bin", "build/bin", "build/bin/Release", "build", "llama.cpp/build/bin",
]
_SEARCH_ROOTS = [
    os.path.expanduser("~/llama.cpp"),
    os.path.expanduser("~/llm/llama.cpp"),
    os.path.expanduser("~/src/llama.cpp"),
    "/usr/local", "/usr", "/opt/llama.cpp", "/opt/homebrew",
    "C:/llama.cpp", "C:/llmpc",
]


class BenchError(RuntimeError):
    pass


class BenchNotFound(BenchError):
    pass


def find_llama_bench(explicit: Optional[str] = None) -> str:
    """Locate llama-bench: explicit path, then $LLAMA_BENCH, then PATH, then usual spots."""
    if explicit:
        cand = os.path.abspath(os.path.expanduser(explicit))
        if os.path.isdir(cand):
            for name in BINARY_NAMES:
                hit = os.path.join(cand, name)
                if os.path.isfile(hit):
                    return hit
            raise BenchNotFound(f"no llama-bench binary inside {cand}")
        if os.path.isfile(cand):
            return cand
        raise BenchNotFound(f"--llama-bench path does not exist: {explicit}")

    env = os.environ.get("LLAMA_BENCH")
    if env and os.path.isfile(os.path.expanduser(env)):
        return os.path.abspath(os.path.expanduser(env))

    for name in BINARY_NAMES:
        hit = shutil.which(name)
        if hit:
            return hit

    for root in _SEARCH_ROOTS:
        for sub in _SEARCH_SUBDIRS:
            for name in BINARY_NAMES:
                hit = os.path.join(root, sub, name)
                if os.path.isfile(hit):
                    return os.path.abspath(hit)

    raise BenchNotFound(
        "could not find `llama-bench`.\n"
        "  * build it:  cmake -B build && cmake --build build --target llama-bench -j\n"
        "  * then point at it: llama-roofline run --llama-bench /path/to/llama-bench ...\n"
        "  * or set the LLAMA_BENCH environment variable."
    )


# Backend-loading chatter that llama.cpp prints on every start. It buries the real error.
_NOISE_PREFIXES = ("load_backend:", "load_tensors:", "ggml_", "build:", "main:",
                   "register_backend", "register_device")


def summarize_stderr(stderr: str, limit: int = 2) -> str:
    """Pull the actual failure out of llama-bench's stderr, discarding startup banners."""
    lines = [ln.strip() for ln in stderr.strip().splitlines() if ln.strip()]
    signal = [ln for ln in lines
              if not ln.startswith(_NOISE_PREFIXES) and "loaded " not in ln]
    chosen = [ln for ln in signal if "error" in ln.lower() or "failed" in ln.lower()]
    if not chosen:
        chosen = signal or lines
    out = " | ".join(chosen[-limit:])
    return out if out else "no error message on stderr"


def parse_bench_json(raw: str) -> Dict[str, Any]:
    """Extract prefill/decode throughput and run metadata from llama-bench JSON output.

    llama-bench emits one record per test. A record with ``n_prompt>0, n_gen==0`` is
    prefill (prompt processing); ``n_gen>0, n_prompt==0`` is decode (token generation).
    """
    raw = raw.strip()
    if not raw:
        raise BenchError("llama-bench produced no output")
    # Some builds print backend-loading banners before the JSON; take the first array.
    start = raw.find("[")
    if start == -1:
        raise BenchError(f"llama-bench output is not JSON: {raw[:200]}")
    try:
        rows = json.loads(raw[start:])
    except json.JSONDecodeError as exc:
        raise BenchError(f"could not parse llama-bench JSON: {exc}") from exc
    if not isinstance(rows, list) or not rows:
        raise BenchError("llama-bench JSON contained no records")

    out: Dict[str, Any] = {
        "prefill_ts": None, "decode_ts": None,
        "prefill_stddev_ts": None, "decode_stddev_ts": None,
    }
    for row in rows:
        if not isinstance(row, dict):
            continue
        n_prompt = row.get("n_prompt", 0) or 0
        n_gen = row.get("n_gen", 0) or 0
        ts = row.get("avg_ts")
        sd = row.get("stddev_ts")
        if n_prompt > 0 and n_gen == 0:
            out["prefill_ts"], out["prefill_stddev_ts"] = ts, sd
        elif n_gen > 0 and n_prompt == 0:
            out["decode_ts"], out["decode_stddev_ts"] = ts, sd

    first = rows[0]
    for key, dest in (
        ("model_size", "model_size"), ("model_n_params", "model_n_params"),
        ("model_type", "model_type"), ("build_commit", "build_commit"),
        ("build_number", "build_number"), ("cpu_info", "cpu_info"),
        ("gpu_info", "gpu_info"), ("backends", "backends"),
        ("n_gpu_layers", "n_gpu_layers"), ("type_k", "type_k"), ("type_v", "type_v"),
    ):
        if key in first:
            out[dest] = first[key]
    return out


def bench_model(binary: str, model_path: str, threads: int,
                n_prompt: int = 128, n_gen: int = 128, reps: int = 3,
                depth: Optional[int] = None, gpu_layers: Optional[int] = None,
                extra_args: Optional[List[str]] = None,
                timeout: Optional[float] = 3600.0) -> Dict[str, Any]:
    """Run one llama-bench invocation and return parsed prefill/decode throughput.

    ``n_prompt`` must be > 0: with ``-p 0`` some llama.cpp builds report a spuriously
    low generation rate, so we always run a real (small) prompt alongside generation.

    ``depth`` maps to llama-bench's ``-d``: generate with that many tokens already in the
    KV cache. Note there is no context-size flag to set here; llama-bench sizes the context
    from n_prompt, n_gen and n_depth itself.
    """
    if n_prompt <= 0:
        raise ValueError("n_prompt must be > 0 (llama-bench reports bogus tg with -p 0)")

    cmd = [binary, "-m", model_path, "-t", str(threads),
           "-p", str(n_prompt), "-n", str(n_gen), "-r", str(reps), "-o", "json"]
    if depth:
        cmd += ["-d", str(depth)]
    if gpu_layers is not None:
        cmd += ["-ngl", str(gpu_layers)]
    if extra_args:
        cmd += list(extra_args)

    try:
        proc = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL, timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise BenchError(f"llama-bench timed out after {timeout}s on {model_path}") from exc
    except OSError as exc:
        raise BenchError(f"could not execute {binary}: {exc}") from exc

    if proc.returncode != 0:
        raise BenchError(
            f"llama-bench exited {proc.returncode}: "
            + summarize_stderr(proc.stderr.decode("utf-8", errors="replace"))
        )

    result = parse_bench_json(proc.stdout.decode("utf-8", errors="replace"))
    result["threads"] = threads
    result["n_prompt"] = n_prompt
    result["n_gen"] = n_gen
    result["reps"] = reps
    result["cmd"] = cmd
    return result


def bench_version(binary: str) -> Dict[str, Any]:
    """Best-effort build identification, without running a model."""
    info: Dict[str, Any] = {"path": binary, "build_commit": None, "build_number": None}
    try:
        proc = subprocess.run([binary, "--help"], stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                              timeout=30)
        text = proc.stdout.decode("utf-8", errors="replace")
    except (OSError, subprocess.SubprocessError):
        return info
    m = re.search(r"version:\s*(\d+)\s*\(([0-9a-f]+)\)", text)
    if m:
        info["build_number"], info["build_commit"] = int(m.group(1)), m.group(2)
    return info
