import json
import os

import pytest

from llama_roofline import cli


def touch(p, size=16):
    p.write_bytes(b"\x00" * size)
    return str(p)


def test_expand_files_dirs_and_globs(tmp_path):
    a = touch(tmp_path / "a.gguf")
    b = touch(tmp_path / "b.gguf")
    (tmp_path / "notes.txt").write_text("ignore me")
    assert cli.expand_model_args([str(tmp_path)]) == [os.path.abspath(a), os.path.abspath(b)]
    assert cli.expand_model_args([str(tmp_path / "*.gguf")]) == [os.path.abspath(a),
                                                                 os.path.abspath(b)]
    assert cli.expand_model_args([a, a, b]) == [os.path.abspath(a), os.path.abspath(b)]


def test_shards_collapse_to_first_part(tmp_path):
    touch(tmp_path / "big-00001-of-00003.gguf", 100)
    touch(tmp_path / "big-00002-of-00003.gguf", 200)
    touch(tmp_path / "big-00003-of-00003.gguf", 300)
    found = cli.expand_model_args([str(tmp_path)])
    assert len(found) == 1
    assert found[0].endswith("big-00001-of-00003.gguf")
    # ...and the size accounts for every shard, not just the first.
    assert cli.total_model_bytes(found[0]) == 600


def test_total_bytes_of_incomplete_shard_set_is_unknown(tmp_path):
    p = touch(tmp_path / "part-00001-of-00002.gguf", 100)
    assert cli.total_model_bytes(p) is None


def test_total_bytes_plain_file(tmp_path):
    p = touch(tmp_path / "solo.gguf", 4242)
    assert cli.total_model_bytes(p) == 4242
    assert cli.total_model_bytes(str(tmp_path / "gone.gguf")) is None


def test_run_without_models_exits_cleanly(tmp_path, capsys):
    rc = cli.main(["run", "--models", str(tmp_path / "nothing-here.gguf")])
    assert rc == 2
    assert "model not found" in capsys.readouterr().err


def test_report_roundtrip(tmp_path, capsys):
    results = {
        "schema_version": 1, "tool_version": "0.1.0",
        "system": {"cpu": "Test CPU", "logical_cores": 8, "ram_bytes": 16_000_000_000,
                   "os": "Linux", "machine": "x86_64"},
        "llama_cpp": {"build_number": 10154, "build_commit": "abc1234"},
        "membw": {"peak_read_GBs": 40.0, "source": "measured", "best_kernel": "dot",
                  "best_threads": 8, "working_set_mb": 512},
        "analysis": {
            "peak_read_GBs": 40.0, "verdict": "memory-bound",
            "bw_util_pct": {"min": 80.0, "median": 85.0, "max": 90.0},
            "fit": {"bw_eff_GBs": 34.0, "r2": 0.997, "n_points": 2, "models": ["a", "b"]},
            "models": [
                {"name": "a", "ok": True, "quant": "Q4_K_M", "bytes_per_token": 400_000_000,
                 "decode_ts": 85.0, "prefill_ts": 236.0, "decode_threads": 8,
                 "decode_bw_GBs": 34.0, "bw_util_pct": 85.0, "runs": []},
                {"name": "b", "ok": True, "quant": "Q4_K_M", "bytes_per_token": 800_000_000,
                 "decode_ts": 42.5, "prefill_ts": 120.0, "decode_threads": 8,
                 "decode_bw_GBs": 34.0, "bw_util_pct": 85.0, "runs": []},
            ],
            "warnings": [], "frontier": [],
        },
    }
    path = tmp_path / "roofline.json"
    path.write_text(json.dumps(results))

    assert cli.main(["report", str(path)]) == 0
    out = capsys.readouterr().out
    assert "memory-bandwidth-bound" in out
    assert "34.00 GB/s" in out

    md = tmp_path / "r.md"
    assert cli.main(["report", str(path), "--markdown", str(md)]) == 0
    text = md.read_text()
    assert "| model |" in text
    assert "memory-bound" in text


def test_report_missing_file(tmp_path, capsys):
    assert cli.main(["report", str(tmp_path / "absent.json")]) == 2
    assert "could not read" in capsys.readouterr().err


def test_parser_defaults():
    args = cli.build_parser().parse_args(["run", "--models", "x.gguf"])
    assert args.n_prompt == 128 and args.n_gen == 128 and args.reps == 3
    assert args.gpu_layers == 0          # a CPU/system-RAM roofline by default
    assert args.depth is None
    assert args.command == "run"


def test_ctx_flag_is_gone(capsys):
    """--ctx passed llama-bench -c, which modern builds reject outright."""
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["run", "--models", "x.gguf", "--ctx", "4096"])
    assert "unrecognized arguments" in capsys.readouterr().err


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    assert "llama-roofline" in capsys.readouterr().out


def _minimal_results(tmp_path):
    results = {
        "schema_version": 1, "tool_version": "0.0.1",
        "system": {"cpu": "Test CPU", "logical_cores": 8},
        "analysis": {
            "peak_read_GBs": 40.0,
            "models": [
                {"name": "a", "bytes_per_token": 400_000_000,
                 "runs": [{"threads": 4, "decode_ts": 30.0, "prefill_ts": 100.0},
                          {"threads": 8, "decode_ts": 25.0, "prefill_ts": 180.0}]},
                {"name": "b", "bytes_per_token": 800_000_000,
                 "runs": [{"threads": 4, "decode_ts": 15.0, "prefill_ts": 60.0},
                          {"threads": 8, "decode_ts": 12.0, "prefill_ts": 110.0}]},
            ],
            "warnings": [], "verdict": "stale", "fit": None,
        },
    }
    path = tmp_path / "roofline.json"
    path.write_text(json.dumps(results))
    return path


def test_reanalyze_rewrites_analysis_but_not_measurements(tmp_path, capsys):
    path = _minimal_results(tmp_path)
    before = json.loads(path.read_text())["analysis"]["models"]

    assert cli.main(["report", str(path), "--reanalyze"]) == 0
    after = json.loads(path.read_text())

    assert after["analysis"]["fit"] is not None       # stale None was recomputed
    assert after["analysis"]["verdict"] != "stale"
    assert after["tool_version"] != "0.0.1"
    assert "reanalyzed_utc" in after
    # The raw llama-bench numbers must survive untouched.
    for old, new in zip(before, after["analysis"]["models"]):
        assert old["runs"] == new["runs"]
        assert old["bytes_per_token"] == new["bytes_per_token"]


def test_reanalyze_without_models_is_an_error(tmp_path, capsys):
    path = tmp_path / "empty.json"
    path.write_text(json.dumps({"schema_version": 1, "analysis": {}}))
    assert cli.main(["report", str(path), "--reanalyze"]) == 2
    assert "--reanalyze needs per-model runs" in capsys.readouterr().err


def test_run_exits_nonzero_when_every_model_fails(tmp_path, monkeypatch, capsys):
    """Files are still written, but a run that measured nothing is not a success."""
    from llama_roofline import bench as bench_mod

    model = tmp_path / "broken.gguf"
    model.write_bytes(b"not a gguf")
    fake_bin = tmp_path / "llama-bench"
    fake_bin.write_text("stub")

    def always_fails(*a, **k):
        raise bench_mod.BenchError("llama-bench exited 1: failed to load model")

    monkeypatch.setattr(bench_mod, "bench_model", always_fails)
    monkeypatch.setattr(bench_mod, "bench_version", lambda b: {"path": b})

    rc = cli.main(["run", "--models", str(model), "--llama-bench", str(fake_bin),
                   "--threads", "1", "--peak-bw", "40", "--no-plot",
                   "--out", str(tmp_path / "out")])
    assert rc == 1
    assert "no model produced a decode measurement" in capsys.readouterr().err
    assert (tmp_path / "out" / "roofline.json").exists()   # output is still kept
