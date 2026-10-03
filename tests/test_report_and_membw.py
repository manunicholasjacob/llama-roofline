import pytest

from llama_roofline import membw, report, roofline, sysinfo


def _results(verdict_bw, peak, n=3):
    models = []
    for i, mb in enumerate([400, 800, 1600][:n]):
        b = int(mb * 1e6)
        tok = verdict_bw * 1e9 / b
        models.append({
            "name": f"model{i}", "quant": "Q4_K_M", "bytes_per_token": b,
            "runs": [{"threads": t, "decode_ts": tok * (1.0 if t == 4 else 0.7),
                      "prefill_ts": 25.0 * t} for t in (1, 2, 4, 8)],
        })
    analysis = roofline.analyze(models, peak)
    return {
        "tool_version": "0.1.0",
        "system": {"cpu": "Test CPU", "logical_cores": 8, "physical_cores": 4,
                   "ram_bytes": 16_000_000_000, "os": "Linux", "machine": "x86_64"},
        "llama_cpp": {"build_number": 10154, "build_commit": "abc1234", "backends": "CPU"},
        "membw": {"source": "measured", "best_kernel": "dot", "best_threads": 8,
                  "working_set_mb": 512},
        "analysis": analysis,
    }


def test_render_memory_bound_report():
    text = report.render(_results(12.0, 14.0))
    assert "YES -- your decode is memory-bandwidth-bound" in text
    assert "decode tok/s  =  12.00 GB/s  /  model bytes" in text
    assert "Best decode thread count: 4" in text
    assert "Prefill is different" in text
    assert "CAVEATS" in text
    assert "Test CPU" in text


def test_render_not_bound_report_gives_troubleshooting():
    text = report.render(_results(4.0, 40.0))
    assert "NO -- something other than memory bandwidth" in text
    assert "thermal" in text


def test_render_without_ceiling_does_not_crash():
    text = report.render(_results(12.0, None))
    assert "ceiling not measured" in text
    assert "Not enough data to classify" in text


def test_render_handles_empty_analysis():
    text = report.render({"tool_version": "0.1.0", "system": {}, "analysis": {}})
    assert "report card" in text
    assert "no successful runs" in text


def test_markdown_report_is_a_table():
    md = report.render_markdown(_results(12.0, 14.0))
    assert md.startswith("# llama-roofline report")
    assert "| model | quant |" in md
    assert "decode tok/s = 12.00 GB/s / model_bytes" in md
    assert "llama-roofline" in md.splitlines()[-1]


def test_failed_model_is_listed():
    res = _results(12.0, 14.0)
    res["analysis"]["models"].append(
        {"name": "broken", "ok": False, "error": "out of memory", "runs": []})
    assert "! broken: out of memory" in report.render(res)


def test_wrap_never_loses_words():
    text = " ".join(f"word{i}" for i in range(40))
    lines = report._wrap(text, 20)
    assert " ".join(lines).split() == text.split()
    assert all(len(l) <= 20 for l in lines)


def test_choose_working_set_respects_small_ram():
    # A 2 GB Raspberry Pi must not get a 512 MB x2 working set.
    assert membw.choose_working_set_mb(None, ram_bytes=2_000_000_000) == 250
    assert membw.choose_working_set_mb(None, ram_bytes=64_000_000_000) == 512
    assert membw.choose_working_set_mb(1024, ram_bytes=2_000_000_000) == 1024
    assert membw.choose_working_set_mb(None, ram_bytes=100_000_000) >= membw.MIN_WORKING_SET_MB


def test_membw_measure_smoke():
    pytest.importorskip("numpy")
    m = membw.measure(working_set_mb=membw.MIN_WORKING_SET_MB, thread_counts=[1, 2], reps=1)
    assert m["peak_read_GBs"] > 0
    assert m["best_kernel"] in ("sum", "max", "dot")
    assert set(m["by_threads"]) == {"1", "2"}
    assert m["source"] == "measured"


def test_sysinfo_collect_is_json_safe():
    import json
    info = sysinfo.collect()
    json.dumps(info)
    assert info["logical_cores"] >= 1
    sweep = sysinfo.default_thread_sweep()
    assert sweep == sorted(set(sweep)) and sweep[0] == 1
    assert sysinfo.default_thread_sweep(max_threads=4)[-1] <= 4


def test_point_label_does_not_repeat_quant():
    plot = pytest.importorskip("llama_roofline.plot")
    # Quant already in the name: do not print it twice.
    assert plot._point_label({"name": "Qwen2.5-0.5B Q4_K_M", "quant": "Q4_K_M"}) \
        == "Qwen2.5-0.5B Q4_K_M"
    # Quant not in the name: add it on a second line.
    assert plot._point_label({"name": "qwen0.5b", "quant": "Q4_K_M"}) == "qwen0.5b\nQ4_K_M"
    # Long names are elided, not cut mid-token without a marker.
    long = plot._point_label({"name": "a-very-long-model-name-that-runs-on", "quant": None})
    assert long.endswith("...") and len(long) <= plot.MAX_LABEL
    assert plot._point_label({"name": "solo", "quant": None}) == "solo"


def test_markdown_header_facts_are_list_items():
    """Consecutive plain lines collapse into one paragraph when rendered; use a list."""
    lines = report.render_markdown(_results(12.0, 14.0)).splitlines()
    facts = [l for l in lines if l.startswith("- **")]
    assert any(l.startswith("- **Machine:**") for l in facts)
    assert any(l.startswith("- **Memory ceiling:**") for l in facts)
    assert any(l.startswith("- **Verdict:**") for l in facts)
    assert any(l.startswith("- **Roofline:**") for l in facts)


def test_markdown_advice_bullets_are_single_lines():
    """Terminal wrapping must not leak into the Markdown."""
    md = report.render_markdown(_results(12.0, 14.0))
    body = md.split("## What this means", 1)[1].split("## Caveats", 1)[0]
    bullets = [l for l in body.splitlines() if l.startswith("- ")]
    assert bullets
    assert all(not l.startswith("  ") for l in body.splitlines() if l.strip())


def test_standing_caveats_appear_in_both_renderers():
    res = _results(12.0, 14.0)
    text, md = report.render(res), report.render_markdown(res)
    for caveat in report.STANDING_CAVEATS:
        head = caveat.split(".")[0]
        assert head.split()[0] in text
        assert caveat in md


def test_ceiling_caveat_matches_how_the_ceiling_was_obtained():
    measured = _results(12.0, 14.0)
    measured["membw"] = {"source": "measured"}
    assert report.CEILING_CAVEAT_MEASURED in report.render_markdown(measured)
    assert report.CEILING_CAVEAT_SUPPLIED not in report.render_markdown(measured)

    supplied = _results(12.0, 14.0)
    supplied["membw"] = {"source": "user-supplied (--peak-bw)"}
    assert report.CEILING_CAVEAT_SUPPLIED in report.render_markdown(supplied)
    assert report.CEILING_CAVEAT_MEASURED not in report.render_markdown(supplied)
    assert report.CEILING_CAVEAT_SUPPLIED.split(".")[0][:40] in " ".join(
        report.render(supplied).split())


def test_no_ceiling_means_no_ceiling_caveat():
    none = _results(12.0, None)
    md = report.render_markdown(none)
    assert report.CEILING_CAVEAT_MEASURED not in md
    assert report.CEILING_CAVEAT_SUPPLIED not in md


def test_markdown_ceiling_labelled_supplied_when_not_measured():
    res = _results(12.0, 14.0)
    res["membw"] = {"source": "external STREAM benchmark"}
    assert "(supplied)" in report.render_markdown(res)


def test_nice_log_ticks_cover_the_range():
    plot = pytest.importorskip("llama_roofline.plot")
    # The awkward case: model sizes from 0.2 GB to 9 GB, where matplotlib's default
    # minor labels collide.
    ticks = plot._nice_log_ticks(0.2, 9.0)
    assert ticks[0] >= 0.2 and ticks[-1] <= 9.0
    assert len(ticks) >= 3
    assert ticks == sorted(ticks)
    # A narrow range still gets usable ticks rather than none.
    assert len(plot._nice_log_ticks(1.1, 1.4)) >= 1
    assert plot._plain(0.5) == "0.5" and plot._plain(100.0) == "100"


def test_membw_flags_unstable_measurement(monkeypatch):
    """Best-of-N hides constant contention; disagreeing repetitions must be reported."""
    pytest.importorskip("numpy")
    calls = {"n": 0}

    def flaky(fn, slices, nthreads, bytes_moved, reps):
        # One fast repetition among slow ones: best is 100, median is 10.
        calls["n"] += 1
        return [100.0, 10.0, 10.0, 10.0, 10.0]

    monkeypatch.setattr(membw, "_samples", flaky)
    m = membw.measure(working_set_mb=membw.MIN_WORKING_SET_MB, thread_counts=[1], reps=5)
    assert calls["n"] > 0
    assert m["unstable"] is True
    assert m["stability"] == pytest.approx(0.1)
    assert "unstable" in m["warning"] and "--peak-bw" in m["warning"]


def test_membw_stable_measurement_has_no_warning(monkeypatch):
    pytest.importorskip("numpy")
    monkeypatch.setattr(membw, "_samples",
                        lambda fn, s, nt, b, r: [50.0, 49.0, 50.5, 49.5, 50.0])
    m = membw.measure(working_set_mb=membw.MIN_WORKING_SET_MB, thread_counts=[1], reps=5)
    assert "unstable" not in m
    assert "warning" not in m
    assert m["stability"] > membw.STABILITY_THRESHOLD


def test_median_helper():
    assert membw._median([3.0, 1.0, 2.0]) == 2.0
    assert membw._median([4.0, 1.0, 2.0, 3.0]) == 2.5
    assert membw._median([]) == 0.0


@pytest.mark.parametrize("reps", [0, -1])
def test_membw_refuses_zero_repetitions(reps):
    """With no repetitions every kernel scored 0.0 and the result was a 0.00 GB/s ceiling
    with no kernel and no thread count, which reads like a measurement."""
    with pytest.raises(ValueError, match="reps must be at least 1"):
        membw.measure(working_set_mb=8, thread_counts=[1], reps=reps)
