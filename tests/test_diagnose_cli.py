"""The zero-configuration front door: discovery, argument handling, and the report.

No llama.cpp and no models are needed. Where a benchmark would run, a fake stands in,
because what is under test is the reporting and the plumbing, not llama-bench.
"""

import json
import os

from llama_roofline import cli, report


def flat(text: str) -> str:
    """One long line, so an assertion does not depend on where a paragraph wrapped."""
    return " ".join(text.split())


# ------------------------------------------------------------------- discovery

def test_discovery_finds_gguf_files_in_a_named_directory(tmp_path):
    (tmp_path / "a.gguf").write_bytes(b"x")
    (tmp_path / "notes.txt").write_bytes(b"x")
    found = cli.find_models([str(tmp_path)], defaults=False)
    assert [os.path.basename(f) for f in found] == ["a.gguf"]


def test_discovery_skips_the_later_shards_of_a_split_model(tmp_path):
    for i in (1, 2, 3):
        (tmp_path / f"big-{i:05d}-of-00003.gguf").write_bytes(b"x")
    found = cli.find_models([str(tmp_path)], defaults=False)
    assert [os.path.basename(f) for f in found] == ["big-00001-of-00003.gguf"]


def test_discovery_does_not_descend_forever(tmp_path):
    deep = tmp_path / "a" / "b" / "c" / "d"
    deep.mkdir(parents=True)
    (deep / "hidden.gguf").write_bytes(b"x")
    assert cli.find_models([str(tmp_path)], defaults=False) == []


def test_discovery_ignores_dot_directories(tmp_path):
    hidden = tmp_path / ".cache_of_junk"
    hidden.mkdir()
    (hidden / "a.gguf").write_bytes(b"x")
    assert cli.find_models([str(tmp_path)], defaults=False) == []


def test_spread_selection_takes_the_ends_and_the_middle(tmp_path):
    paths = []
    for i, size in enumerate([100, 200, 300, 4000, 50000]):
        p = tmp_path / f"m{i}.gguf"
        p.write_bytes(b"0" * size)
        paths.append(str(p))
    picked = cli.spread_selection(paths, 3)
    sizes = sorted(os.path.getsize(p) for p in picked)
    assert sizes == [100, 300, 50000]


def test_spread_selection_passes_a_short_list_through(tmp_path):
    p = tmp_path / "only.gguf"
    p.write_bytes(b"x")
    assert cli.spread_selection([str(p)], 3) == [str(p)]


# ------------------------------------------------------------------- the report

def _fake_results(util_pct=85.0, with_streamed=True):
    """A results dict shaped like a real diagnose run, with the numbers chosen."""
    model = {
        "name": "test-model", "quant": "Q4_K_M", "ok": True,
        "bytes_per_token": 500_000_000,
        "bytes_source": ("streamed tensors (GGUF tensor map)" if with_streamed
                         else "llama-bench model_size"),
        "decode_ts": 20.0, "decode_threads": 4, "prefill_ts": 100.0,
        "decode_bw_GBs": 10.0, "bw_util_pct": util_pct,
        "runs": [
            {"threads": 4, "decode_ts": 20.0, "prefill_ts": 100.0},
            {"threads": 16, "decode_ts": 16.0, "prefill_ts": 400.0},
        ],
    }
    if with_streamed:
        model["streamed"] = {
            "total_bytes": 700_000_000, "streamed_bytes": 500_000_000,
            "embedding_bytes": 200_000_000, "head_bytes": 150_000_000,
            "tied_embedding": False, "head_share_pct": 30.0, "n_tensors": 200,
        }
    return {
        "tool_version": "0.2.0", "command": "diagnose",
        "system": {"cpu": "Test CPU", "logical_cores": 16, "physical_cores": 8,
                   "ram_bytes": 32_000_000_000, "os": "Linux", "machine": "x86_64"},
        "llama_cpp": {"build_number": 1, "build_commit": "abc"},
        "membw": {"source": "measured", "best_kernel": "dot", "best_threads": 8},
        "analysis": {
            "models": [model], "peak_read_GBs": 12.0, "n_models_ok": 1,
            "warnings": [], "fit": None, "verdict": "memory-bound",
            "bw_util_pct": {"min": util_pct, "max": util_pct, "median": util_pct},
        },
    }


def test_a_bound_setup_gets_a_yes_and_a_token_ceiling():
    text = report.render_diagnosis(_fake_results(85.0))
    assert "Yes. Decode is memory-bandwidth-bound" in flat(text)
    # 12 GB/s over 500 MB per token is 24 tok/s.
    assert "about 24 tok/s" in flat(text)
    assert "You measured 20.0" in flat(text)


def test_an_unbound_setup_is_told_where_to_look_instead():
    text = report.render_diagnosis(_fake_results(20.0))
    assert "No. Decode is using only 20%" in flat(text)
    assert "paging off disk" in flat(text)


def test_the_report_says_when_more_threads_cost_throughput():
    text = report.render_diagnosis(_fake_results())
    assert "Threads: use 4" in flat(text)
    assert "20%" in flat(text)  # 16.0 against 20.0


def test_the_report_explains_the_gap_between_file_size_and_traffic():
    text = report.render_diagnosis(_fake_results(with_streamed=True))
    assert "reads 500 MB per token, not the 700 MB the file weighs" in flat(text)


def test_the_caveat_matches_the_byte_convention_that_was_used():
    streamed = report.render_diagnosis(_fake_results(with_streamed=True))
    assert "The token embedding is excluded" in flat(streamed)
    assert "ties the embedding to the output head" in flat(streamed)
    resident = report.render_diagnosis(_fake_results(with_streamed=False))
    assert "resident size" in flat(resident)


def test_diagnosis_is_plain_ascii():
    for renderer in (report.render_diagnosis, report.render_diagnosis_markdown):
        text = renderer(_fake_results())
        text.encode("ascii")
        assert "—" not in text


def test_markdown_diagnosis_is_pasteable():
    md = report.render_diagnosis_markdown(_fake_results())
    assert md.startswith("# llama-roofline diagnosis")
    assert "| model | quant |" in md
    assert "llama-roofline diagnose" in md


def test_a_run_where_nothing_measured_says_so_rather_than_inventing_a_verdict():
    results = _fake_results()
    results["analysis"]["models"] = []
    results["analysis"]["n_models_ok"] = 0
    text = report.render_diagnosis(results)
    assert "Nothing was measured" in flat(text)


# ------------------------------------------------------------------- arguments

def test_diagnose_without_models_and_without_any_on_disk_explains_itself(
        tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "find_models", lambda *a, **k: [])
    monkeypatch.chdir(tmp_path)
    rc = cli.main(["diagnose"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "no GGUF models found" in err
    assert "LLAMA_MODELS" in err


def test_diagnose_with_a_path_that_is_not_a_model_fails_clearly(tmp_path, capsys):
    rc = cli.main(["diagnose", str(tmp_path / "nothing-here.gguf")])
    assert rc == 2
    assert "none of those paths is a GGUF file" in capsys.readouterr().err


def test_advise_list_cores_needs_no_hardware(capsys):
    assert cli.main(["advise", "--list-cores"]) == 0
    out = capsys.readouterr().out
    assert "cortex-a76" in out and "gracemont" in out


def test_advise_rejects_a_core_it_has_no_measurements_for(capsys):
    assert cli.main(["advise", "--core", "apple-m3"]) == 2
    assert "no measurements for core" in capsys.readouterr().err


def test_advise_measure_refuses_a_single_file(tmp_path, capsys):
    p = tmp_path / "one.gguf"
    p.write_bytes(b"x")
    assert cli.main(["advise", "--measure", "--models", str(p)]) == 2
    assert "at least two files" in capsys.readouterr().err


def test_advise_writes_the_json_it_promises(tmp_path, capsys):
    out = tmp_path / "advice.json"
    assert cli.main(["advise", "--core", "cortex-a76", "--json", str(out)]) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["advice"][0]["core"] == "cortex-a76"


def test_inspect_reports_a_broken_file_without_crashing(tmp_path, capsys):
    p = tmp_path / "broken.gguf"
    p.write_bytes(b"not a gguf at all")
    assert cli.main(["inspect", str(p)]) == 2
    assert "could not read the tensor table" in capsys.readouterr().err
