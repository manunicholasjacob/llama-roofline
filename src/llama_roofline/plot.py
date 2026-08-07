"""Figure generation. matplotlib is an optional extra -- the tool works fully without it."""

from __future__ import annotations

from typing import Any, Dict, Optional

CEILING_COLOR = "#444444"
FIT_COLOR = "#c51b7d"
POINT_COLOR = "#2166ac"
PREFILL_COLOR = "#01665e"


MAX_LABEL = 22


class PlotUnavailable(RuntimeError):
    pass


def _point_label(model: dict) -> str:
    """Model name, plus the quant on a second line only if it is not already in the name."""
    name = model.get("name") or "?"
    quant = model.get("quant")
    if len(name) > MAX_LABEL:
        # ASCII rather than a horizontal ellipsis: no font or encoding surprises.
        name = name[:MAX_LABEL - 3].rstrip("-_ ") + "..."
    if quant and quant.lower() not in (model.get("name") or "").lower():
        return f"{name}\n{quant}"
    return name


_NICE = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10, 20, 30, 50,
         100, 200, 300, 500, 1000]


def _nice_log_ticks(lo: float, hi: float):
    """Readable ticks for a log axis spanning under a decade or several.

    matplotlib's default minor labels on a log axis collide badly over a range like
    0.3 to 9, which is exactly the range model sizes fall in.
    """
    ticks = [t for t in _NICE if lo <= t <= hi]
    if len(ticks) < 3:                       # very narrow range: fall back to the decades
        ticks = [t for t in _NICE if lo / 1.5 <= t <= hi * 1.5]
    return ticks


def _plain(value: float) -> str:
    if value >= 10:
        return f"{value:.0f}"
    if value >= 1:
        return f"{value:g}"
    return f"{value:g}"


def _apply_log_ticks(ax, axis: str, lo: float, hi: float):
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
    ticks = _nice_log_ticks(lo, hi)
    if not ticks:
        return
    target = ax.xaxis if axis == "x" else ax.yaxis
    target.set_major_locator(FixedLocator(ticks))
    target.set_major_formatter(FuncFormatter(lambda v, _: _plain(v)))
    target.set_minor_formatter(NullFormatter())


def _require_mpl():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise PlotUnavailable(
            "matplotlib is needed to draw the figure. Install it (pip install matplotlib) "
            "or pass --no-plot."
        ) from exc
    return plt


def roofline_figure(results: Dict[str, Any], out_path: str,
                    title: Optional[str] = None, dpi: int = 150) -> str:
    """Two panels: the roofline itself, and why threads help prefill but not decode."""
    plt = _require_mpl()
    analysis = results.get("analysis", {})
    models = [m for m in analysis.get("models", []) if m.get("ok") and m.get("bytes_per_token")]
    if not models:
        raise PlotUnavailable("no successful measurements to plot")

    peak = analysis.get("peak_read_GBs")
    fit = analysis.get("fit")
    sysinfo = results.get("system", {})

    plt.rcParams.update({
        "font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": dpi, "savefig.bbox": "tight",
    })
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    ax = axes[0]

    xs = [m["bytes_per_token"] / 1e9 for m in models]
    ys = [m["decode_ts"] for m in models]
    # Extra room on the right: the largest model's label extends past its point.
    lo, hi = min(xs) * 0.65, max(xs) * 1.9
    grid = [lo + (hi - lo) * i / 199.0 for i in range(200)]

    if peak:
        measured = (results.get("membw") or {}).get("source") == "measured"
        ax.plot(grid, [peak / g for g in grid], ls="--", lw=1.4, color=CEILING_COLOR,
                label=f"memory ceiling: {peak:.1f} GB/s "
                      f"({'measured' if measured else 'supplied'})")
    if fit:
        r2 = fit.get("r2")
        lab = f"fit: {fit['bw_eff_GBs']:.2f} GB/s / bytes"
        if r2 == r2:
            lab += f"  ($R^2$={r2:.3f})"
        ax.plot(grid, [fit["bw_eff_GBs"] / g for g in grid], lw=1.6, color=FIT_COLOR, label=lab)

    dense_x = [x for x, m in zip(xs, models) if not m.get("is_moe")]
    dense_y = [y for y, m in zip(ys, models) if not m.get("is_moe")]
    ax.scatter(dense_x, dense_y, s=68, color=POINT_COLOR, zorder=5, label="your models (decode)")
    moe_x = [x for x, m in zip(xs, models) if m.get("is_moe")]
    moe_y = [y for y, m in zip(ys, models) if m.get("is_moe")]
    if moe_x:
        ax.scatter(moe_x, moe_y, s=68, marker="D", facecolors="none",
                   edgecolors=POINT_COLOR, zorder=5, label="MoE (off the dense roofline)")
    # Points cluster tightly on a quantization sweep, so alternate the label side and
    # avoid repeating a quant that is already part of the model name.
    order = sorted(range(len(models)), key=lambda i: xs[i])
    for rank, i in enumerate(order):
        m = models[i]
        label = _point_label(m)
        right = rank % 2 == 0
        ax.annotate(label, (xs[i], ys[i]), fontsize=6.5,
                    xytext=(6, 5) if right else (-6, -12),
                    ha="left" if right else "right",
                    textcoords="offset points", color="#333333")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)   # annotations sit outside the data range; give them the room
    _apply_log_ticks(ax, "x", lo, hi)
    _apply_log_ticks(ax, "y", min(ys) * 0.8, max(ys) * 1.25)
    ax.set_xlabel("Model size (GB read per token)")
    ax.set_ylabel("Decode throughput (tok/s)")
    ax.set_title("(a) Decode rides the memory ceiling")
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")

    ax2 = axes[1]
    plotted = False
    all_threads = set()
    for m in models:
        runs = sorted((r for r in m.get("runs", []) if r.get("decode_ts")),
                      key=lambda r: r["threads"])
        if len(runs) < 2:
            continue
        t = [r["threads"] for r in runs]
        all_threads.update(t)
        d0 = runs[0]["decode_ts"]
        ax2.plot(t, [r["decode_ts"] / d0 for r in runs], marker="o", lw=1.4,
                 color=POINT_COLOR, alpha=0.75,
                 label="decode" if not plotted else None)
        pruns = [r for r in runs if r.get("prefill_ts")]
        if len(pruns) >= 2:
            p0 = pruns[0]["prefill_ts"]
            ax2.plot([r["threads"] for r in pruns], [r["prefill_ts"] / p0 for r in pruns],
                     marker="s", ls="--", lw=1.4, color=PREFILL_COLOR, alpha=0.75,
                     label="prefill" if not plotted else None)
        plotted = True
    if plotted:
        ax2.axhline(1.0, color="#999999", lw=0.8, ls=":")
        ax2.set_xticks(sorted(all_threads))   # thread counts are integers, not 1.5
        # A single-thread run on a model near the RAM limit can thrash and produce a huge
        # speedup ratio that flattens every other curve on a linear axis.
        top = max(ax2.get_ylim())
        if top > 12:
            ax2.set_yscale("log")
            _apply_log_ticks(ax2, "y", 0.9, top)
        ax2.set_xlabel("Threads")
        ax2.set_ylabel("Speedup vs. fewest threads")
        ax2.set_title("(b) Threads buy prefill, not decode")
        ax2.legend(frameon=False, fontsize=8)
    else:
        ax2.set_axis_off()
        ax2.text(0.5, 0.5, "run a thread sweep\n(--threads 1,2,4,8)",
                 ha="center", va="center", fontsize=9, color="#777777")

    head = title or (sysinfo.get("cpu") or "llama.cpp roofline")
    verdict = analysis.get("verdict")
    util = analysis.get("bw_util_pct")
    if verdict and util:
        head += f"  --  decode is {verdict} ({util['median']:.0f}% of memory ceiling)"
    fig.suptitle(head, fontsize=11, y=1.02)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
