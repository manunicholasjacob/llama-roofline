import json
import os

import pytest

from llama_roofline import bench

PREFILL = {
    "build_commit": "0e4a03622", "build_number": 10154,
    "cpu_info": "12th Gen Intel(R) Core(TM) i7-12700H", "gpu_info": "", "backends": "CPU",
    "model_filename": "m.gguf", "model_type": "qwen2 1B Q4_K - Medium",
    "model_size": 332659200, "model_n_params": 494032768, "n_threads": 8,
    "n_gpu_layers": 0, "type_k": "f16", "type_v": "f16",
    "n_prompt": 128, "n_gen": 0, "avg_ts": 361.089045, "stddev_ts": 1.5,
}
DECODE = dict(PREFILL, n_prompt=0, n_gen=128, avg_ts=85.64, stddev_ts=0.4)


def test_parse_splits_prefill_and_decode():
    out = bench.parse_bench_json(json.dumps([PREFILL, DECODE]))
    assert out["prefill_ts"] == pytest.approx(361.089045)
    assert out["decode_ts"] == pytest.approx(85.64)
    assert out["decode_stddev_ts"] == pytest.approx(0.4)
    assert out["model_size"] == 332659200
    assert out["build_number"] == 10154
    assert out["backends"] == "CPU"


def test_parse_tolerates_leading_banner():
    raw = ("load_backend: loaded RPC backend from ggml-rpc.dll\n"
           "load_backend: loaded CPU backend from ggml-cpu-alderlake.dll\n"
           + json.dumps([PREFILL, DECODE]))
    out = bench.parse_bench_json(raw)
    assert out["decode_ts"] == pytest.approx(85.64)


def test_parse_decode_only():
    out = bench.parse_bench_json(json.dumps([DECODE]))
    assert out["prefill_ts"] is None
    assert out["decode_ts"] == pytest.approx(85.64)


def test_parse_rejects_garbage():
    with pytest.raises(bench.BenchError):
        bench.parse_bench_json("")
    with pytest.raises(bench.BenchError):
        bench.parse_bench_json("segmentation fault")
    with pytest.raises(bench.BenchError):
        bench.parse_bench_json("[")
    with pytest.raises(bench.BenchError):
        bench.parse_bench_json("[]")


def test_zero_prompt_is_rejected():
    # -p 0 makes some llama.cpp builds report a spuriously low generation rate.
    with pytest.raises(ValueError):
        bench.bench_model("llama-bench", "m.gguf", threads=4, n_prompt=0)


def test_find_llama_bench_explicit_file(tmp_path):
    exe = tmp_path / "llama-bench"
    exe.write_text("#!/bin/sh\n")
    assert bench.find_llama_bench(str(exe)) == os.path.abspath(str(exe))


def test_find_llama_bench_explicit_dir(tmp_path):
    exe = tmp_path / "llama-bench.exe"
    exe.write_text("stub")
    assert bench.find_llama_bench(str(tmp_path)) == str(exe)


def test_find_llama_bench_missing_path_raises(tmp_path):
    with pytest.raises(bench.BenchNotFound):
        bench.find_llama_bench(str(tmp_path / "nope"))
    with pytest.raises(bench.BenchNotFound):
        bench.find_llama_bench(str(tmp_path))  # empty dir


def test_find_llama_bench_env(tmp_path, monkeypatch):
    exe = tmp_path / "lb"
    exe.write_text("stub")
    monkeypatch.setenv("LLAMA_BENCH", str(exe))
    assert bench.find_llama_bench(None) == os.path.abspath(str(exe))
