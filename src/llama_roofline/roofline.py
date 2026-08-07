"""The analysis: turn raw llama-bench numbers into a roofline.

The model, in one line:

    decode tok/s  =  BW_eff / bytes_per_token

For a dense transformer every generated token requires reading every weight once, so
bytes_per_token is essentially the model's resident size. Fitting a single BW_eff across
models of different sizes tests that claim: if it holds, the points fall on a hyperbola
and BW_eff comes out close to the machine's measured memory ceiling.

Pure standard library -- numpy is only needed to *measure* the ceiling, not to fit it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# How close to the measured ceiling counts as "the wall".
HARD_BOUND_PCT = 70.0
PARTLY_BOUND_PCT = 40.0
# Below this relative gain, extra threads are not worth it.
THREAD_KNEE_TOL = 0.02


def fit_through_origin(bytes_list: List[float], toks_list: List[float]) -> Dict[str, Any]:
    """Least-squares fit of tok/s = BW / bytes, i.e. y = BW * x with x = 1/bytes.

    Returns BW in bytes/s plus the coefficient of determination against the mean of y
    (the ordinary R^2, so it is comparable with a regression that had an intercept).
    """
    n = len(bytes_list)
    if n != len(toks_list) or n == 0:
        raise ValueError("need matching, non-empty byte and throughput lists")
    xs = [1.0 / b for b in bytes_list]
    ys = list(toks_list)
    sxy = sum(x * y for x, y in zip(xs, ys))
    sxx = sum(x * x for x in xs)
    if sxx == 0:
        raise ValueError("degenerate fit (zero model sizes)")
    bw = sxy / sxx
    ybar = sum(ys) / n
    ss_res = sum((y - bw * x) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - ybar) ** 2 for y in ys)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"bw_eff_GBs": bw / 1e9, "bw_eff_bytes_s": bw, "r2": r2, "n_points": n}


def summarize_model(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Collapse a model's thread sweep into its best decode/prefill operating points."""
    runs = [r for r in entry.get("runs", []) if r.get("decode_ts")]
    out: Dict[str, Any] = dict(entry)
    if not runs:
        out["ok"] = False
        return out

    best = max(runs, key=lambda r: r["decode_ts"])
    out["ok"] = True
    out["decode_ts"] = best["decode_ts"]
    out["decode_threads"] = best["threads"]
    out["decode_stddev_ts"] = best.get("decode_stddev_ts")

    prefill_runs = [r for r in entry.get("runs", []) if r.get("prefill_ts")]
    if prefill_runs:
        bp = max(prefill_runs, key=lambda r: r["prefill_ts"])
        out["prefill_ts"] = bp["prefill_ts"]
        out["prefill_threads"] = bp["threads"]
        lo = min(prefill_runs, key=lambda r: r["threads"])
        if lo["prefill_ts"] and lo["threads"] != bp["threads"]:
            out["prefill_scaling"] = bp["prefill_ts"] / lo["prefill_ts"]
            out["prefill_scaling_threads"] = [lo["threads"], bp["threads"]]

    b = out.get("bytes_per_token")
    if b:
        out["decode_bw_GBs"] = out["decode_ts"] * b / 1e9

    # Where does adding threads stop paying? First thread count within tolerance of best.
    by_threads = sorted(runs, key=lambda r: r["threads"])
    knee = by_threads[-1]["threads"]
    for r in by_threads:
        if r["decode_ts"] >= best["decode_ts"] * (1.0 - THREAD_KNEE_TOL):
            knee = r["threads"]
            break
    out["decode_thread_knee"] = knee
    top = by_threads[-1]
    if top["threads"] != best["threads"] and best["decode_ts"]:
        out["decode_loss_at_max_threads_pct"] = 100.0 * (1 - top["decode_ts"] / best["decode_ts"])
    return out


def analyze(models: List[Dict[str, Any]], peak_read_GBs: Optional[float]) -> Dict[str, Any]:
    """Full roofline analysis over a set of benchmarked models."""
    summaries = [summarize_model(m) for m in models]
    ok = [m for m in summaries if m.get("ok") and m.get("bytes_per_token")]

    analysis: Dict[str, Any] = {
        "models": summaries,
        "peak_read_GBs": peak_read_GBs,
        "n_models_ok": len(ok),
        "warnings": [],
        "fit": None,
        "verdict": None,
    }
    if not ok:
        analysis["warnings"].append("no model produced a decode measurement; nothing to analyze")
        return analysis

    # MoE models read only the active experts per token, so their bytes-per-token is
    # well below the file size. Including them would bend the fit; report separately.
    dense = [m for m in ok if not m.get("is_moe")]
    moe = [m for m in ok if m.get("is_moe")]
    if moe:
        analysis["warnings"].append(
            "excluded from the fit (mixture-of-experts -- only the active experts are read "
            "per token, so bytes-per-token is far below the model size): "
            + ", ".join(m["name"] for m in moe)
        )

    fit_set = dense or ok
    if len(fit_set) >= 2:
        analysis["fit"] = fit_through_origin(
            [m["bytes_per_token"] for m in fit_set],
            [m["decode_ts"] for m in fit_set],
        )
        analysis["fit"]["models"] = [m["name"] for m in fit_set]
    elif len(fit_set) == 1:
        analysis["warnings"].append(
            "only one model benchmarked -- the roofline slope cannot be fitted. "
            "Add a second model of a different size (or the same model at another quant) "
            "to test whether tok/s really scales as 1/bytes."
        )

    utils = []
    for m in summaries:
        if m.get("decode_bw_GBs") and peak_read_GBs:
            m["bw_util_pct"] = 100.0 * m["decode_bw_GBs"] / peak_read_GBs
            if not m.get("is_moe"):
                utils.append(m["bw_util_pct"])
    if utils:
        utils_sorted = sorted(utils)
        mid = len(utils_sorted) // 2
        median = (utils_sorted[mid] if len(utils_sorted) % 2
                  else 0.5 * (utils_sorted[mid - 1] + utils_sorted[mid]))
        analysis["bw_util_pct"] = {
            "min": utils_sorted[0], "max": utils_sorted[-1], "median": median,
        }
        if median >= HARD_BOUND_PCT:
            verdict = "memory-bound"
        elif median >= PARTLY_BOUND_PCT:
            verdict = "partly memory-bound"
        else:
            verdict = "not bandwidth-limited"
        analysis["verdict"] = verdict

        if utils_sorted[-1] > 100.0:
            analysis["warnings"].append(
                f"measured decode bandwidth exceeded the microbenchmark ceiling "
                f"(max {utils_sorted[-1]:.0f}%). The ceiling is a measured lower bound, so "
                f"treat the percentages as lower bounds too -- llama.cpp's hand-tuned kernels "
                f"can stream faster than numpy. Pass --peak-bw with a STREAM number for a "
                f"tighter ceiling."
            )
    elif peak_read_GBs is None:
        analysis["warnings"].append(
            "no memory-bandwidth ceiling available, so utilisation is unknown. "
            "Re-run without --skip-membw, or pass --peak-bw <GB/s>."
        )

    fit = analysis.get("fit")
    if fit and fit["n_points"] >= 3 and fit["r2"] == fit["r2"] and fit["r2"] < 0.90:
        analysis["warnings"].append(
            f"the 1/bytes fit is poor (R^2 = {fit['r2']:.3f}). Decode on this setup is not "
            f"cleanly bandwidth-bound -- likely causes: partial GPU offload, a mixture-of-experts "
            f"model, thermal throttling, or another process competing for memory."
        )

    # Quantization / size frontier, cheapest bytes first.
    frontier = sorted(ok, key=lambda m: m["bytes_per_token"])
    analysis["frontier"] = [
        {"name": m["name"], "quant": m.get("quant"), "bytes": m["bytes_per_token"],
         "decode_ts": m["decode_ts"]}
        for m in frontier
    ]
    if len(frontier) >= 2:
        lo, hi = frontier[0], frontier[-1]
        analysis["frontier_span"] = {
            "smallest": lo["name"], "largest": hi["name"],
            "size_ratio": hi["bytes_per_token"] / lo["bytes_per_token"],
            "speed_ratio": lo["decode_ts"] / hi["decode_ts"] if hi["decode_ts"] else None,
        }

    # Only models that actually swept threads have a meaningful knee; a model measured at
    # a single thread count would otherwise "vote" for whatever setting it happened to use.
    knees = [m["decode_thread_knee"] for m in ok
             if m.get("decode_thread_knee") and len(m.get("runs", [])) >= 2]
    if knees:
        analysis["decode_thread_knee"] = max(set(knees), key=knees.count)
    # Report the median, not the max. A single-thread run on a model near the RAM limit can
    # thrash and produce an absurd speedup ratio that is a measurement artefact, not thread
    # scaling; the median across models is robust to exactly that.
    scal = sorted(m["prefill_scaling"] for m in summaries if m.get("prefill_scaling"))
    if scal:
        mid = len(scal) // 2
        analysis["prefill_scaling"] = (scal[mid] if len(scal) % 2
                                       else 0.5 * (scal[mid - 1] + scal[mid]))
        analysis["prefill_scaling_range"] = [scal[0], scal[-1]]

    if any(m.get("n_gpu_layers") not in (0, None) and m.get("gpu_info") for m in summaries):
        analysis["warnings"].append(
            "at least one run used GPU offload. The ceiling measured here is *system RAM* "
            "bandwidth; offloaded layers stream from VRAM instead, so the utilisation figures "
            "do not apply. Re-run with --gpu-layers 0 for a clean CPU roofline."
        )
    return analysis
