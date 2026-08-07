import math

import pytest

from llama_roofline import roofline


def _model(name, mb, tok_s, threads=(1, 2, 4), quant="Q4_K_M", **kw):
    """A synthetic model whose decode peaks at the middle thread count."""
    peaks = {threads[len(threads) // 2]: tok_s}
    runs = []
    for i, t in enumerate(threads):
        dec = peaks.get(t, tok_s * (0.6 + 0.1 * i))
        runs.append({"threads": t, "decode_ts": dec, "prefill_ts": 20.0 * t})
    return {"name": name, "quant": quant, "bytes_per_token": int(mb * 1e6), "runs": runs, **kw}


def test_fit_recovers_exact_bandwidth():
    bw = 12.0e9
    sizes = [400e6, 800e6, 1600e6]
    toks = [bw / s for s in sizes]
    fit = roofline.fit_through_origin(sizes, toks)
    assert fit["bw_eff_GBs"] == pytest.approx(12.0, rel=1e-9)
    assert fit["r2"] == pytest.approx(1.0, abs=1e-9)
    assert fit["n_points"] == 3


def test_fit_rejects_bad_input():
    with pytest.raises(ValueError):
        roofline.fit_through_origin([], [])
    with pytest.raises(ValueError):
        roofline.fit_through_origin([1.0, 2.0], [1.0])


def test_summarize_picks_best_threads_and_knee():
    m = _model("m", 400, 30.0, threads=(1, 2, 4, 8))
    s = roofline.summarize_model(m)
    assert s["ok"] is True
    assert s["decode_threads"] == 4          # the injected peak
    assert s["decode_thread_knee"] <= 4
    assert s["decode_bw_GBs"] == pytest.approx(30.0 * 400e6 / 1e9)
    assert s["prefill_scaling"] == pytest.approx(8.0)   # prefill grows linearly here


def test_summarize_marks_failed_model():
    s = roofline.summarize_model({"name": "x", "bytes_per_token": 1, "runs": []})
    assert s["ok"] is False


def test_analyze_declares_memory_bound():
    bw = 12.0e9
    models = [_model(f"m{i}", mb, bw / (mb * 1e6)) for i, mb in enumerate((400, 800, 1600))]
    a = roofline.analyze(models, peak_read_GBs=14.0)
    assert a["verdict"] == "memory-bound"
    assert a["fit"]["bw_eff_GBs"] == pytest.approx(12.0, rel=1e-6)
    assert a["bw_util_pct"]["median"] == pytest.approx(100 * 12.0 / 14.0, rel=1e-6)
    assert a["frontier_span"]["size_ratio"] == pytest.approx(4.0)
    assert a["frontier_span"]["speed_ratio"] == pytest.approx(4.0)


def test_analyze_declares_not_bound_when_far_from_ceiling():
    bw = 4.0e9
    models = [_model(f"m{i}", mb, bw / (mb * 1e6)) for i, mb in enumerate((400, 800, 1600))]
    a = roofline.analyze(models, peak_read_GBs=40.0)
    assert a["verdict"] == "not bandwidth-limited"


def test_analyze_warns_when_decode_exceeds_ceiling():
    bw = 50.0e9
    models = [_model(f"m{i}", mb, bw / (mb * 1e6)) for i, mb in enumerate((400, 800))]
    a = roofline.analyze(models, peak_read_GBs=40.0)
    assert any("exceeded the microbenchmark ceiling" in w for w in a["warnings"])


def test_analyze_excludes_moe_from_fit():
    bw = 12.0e9
    dense = [_model(f"d{i}", mb, bw / (mb * 1e6)) for i, mb in enumerate((400, 800))]
    moe = _model("mixture", 4000, 25.0, is_moe=True)   # way off the dense line
    a = roofline.analyze(dense + [moe], peak_read_GBs=14.0)
    assert "mixture" not in a["fit"]["models"]
    assert a["fit"]["bw_eff_GBs"] == pytest.approx(12.0, rel=1e-6)
    assert any("mixture-of-experts" in w for w in a["warnings"])


def test_analyze_warns_on_single_model():
    a = roofline.analyze([_model("only", 400, 20.0)], peak_read_GBs=14.0)
    assert a["fit"] is None
    assert any("only one model" in w for w in a["warnings"])


def test_analyze_warns_on_poor_fit():
    models = [_model("a", 400, 30.0), _model("b", 800, 29.0), _model("c", 1600, 28.0)]
    a = roofline.analyze(models, peak_read_GBs=40.0)
    assert any("fit is poor" in w for w in a["warnings"])


def test_analyze_handles_no_ceiling():
    models = [_model(f"m{i}", mb, 10.0) for i, mb in enumerate((400, 800))]
    a = roofline.analyze(models, peak_read_GBs=None)
    assert a["verdict"] is None
    assert any("no memory-bandwidth ceiling" in w for w in a["warnings"])


def test_analyze_handles_all_failures():
    a = roofline.analyze([{"name": "x", "bytes_per_token": 1, "runs": []}], peak_read_GBs=14.0)
    assert a["fit"] is None
    assert any("nothing to analyze" in w for w in a["warnings"])


def test_r2_is_nan_for_identical_throughputs():
    fit = roofline.fit_through_origin([1e9, 1e9], [10.0, 10.0])
    assert math.isnan(fit["r2"])


def test_thread_knee_ignores_single_setting_models():
    """A model measured at one thread count must not vote on the global thread knee."""
    swept = [_model("a", 400, 30.0, threads=(1, 2, 4, 8)),
             _model("b", 800, 15.0, threads=(1, 2, 4, 8))]
    single = {"name": "quant-only", "quant": "Q2_K", "bytes_per_token": int(300e6),
              "runs": [{"threads": 20, "decode_ts": 40.0, "prefill_ts": 200.0}]}
    a = roofline.analyze(swept + [single], peak_read_GBs=40.0)
    assert a["decode_thread_knee"] != 20


def test_prefill_scaling_uses_median_not_max():
    """One thrashing single-thread run must not become the headline scaling number."""
    normal = []
    for i, mb in enumerate((400, 800, 1600)):
        normal.append({"name": f"m{i}", "bytes_per_token": int(mb * 1e6),
                       "runs": [{"threads": 1, "decode_ts": 10.0, "prefill_ts": 20.0},
                                {"threads": 8, "decode_ts": 20.0, "prefill_ts": 140.0}]})
    thrashing = {"name": "huge", "bytes_per_token": int(5000e6),
                 "runs": [{"threads": 1, "decode_ts": 0.9, "prefill_ts": 1.3},
                          {"threads": 8, "decode_ts": 11.0, "prefill_ts": 42.0}]}
    a = roofline.analyze(normal + [thrashing], peak_read_GBs=54.0)
    assert a["prefill_scaling"] == pytest.approx(7.0)      # median, not the 32x outlier
    assert a["prefill_scaling_range"][1] > 30              # the outlier is still recorded
