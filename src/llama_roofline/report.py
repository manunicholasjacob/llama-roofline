"""The report card: what the numbers mean, in plain English.

The whole point of the tool is this file. A JSON of bandwidth figures helps nobody who
is not already thinking in rooflines; a paragraph that says "you are at 87% of your
memory ceiling, so a faster CPU will not help you -- a smaller model will" does.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

RULE = "=" * 72
THIN = "-" * 72

# Always true, regardless of what was measured.
STANDING_CAVEATS = [
    "bytes-per-token is taken as the model's resident size. That is exact for a dense "
    "transformer at short context and an overestimate once the KV cache grows large, or "
    "for MoE models.",
    "Numbers are throughput only. Nothing here measures output quality.",
]

CEILING_CAVEAT_MEASURED = (
    "The ceiling is what this machine sustained under a portable microbenchmark, not the "
    "spec sheet number. It is a floor on the truth, so the percentages are conservative."
)
CEILING_CAVEAT_SUPPLIED = (
    "The ceiling was supplied rather than measured by this tool, so every percentage here "
    "is only as good as that number."
)


def _caveats(results: Dict[str, Any]) -> List[str]:
    """Run-specific warnings first, then the caveats that always apply."""
    analysis = results.get("analysis", {}) or {}
    out = list(analysis.get("warnings", []))
    standing = list(STANDING_CAVEATS)
    if analysis.get("peak_read_GBs"):
        measured = (results.get("membw") or {}).get("source") == "measured"
        standing.insert(1, CEILING_CAVEAT_MEASURED if measured else CEILING_CAVEAT_SUPPLIED)
    return out + standing


def _gb(nbytes: Optional[float]) -> str:
    if not nbytes:
        return "?"
    if nbytes >= 1e9:
        return f"{nbytes / 1e9:.2f} GB"
    return f"{nbytes / 1e6:.0f} MB"


def _fmt_ram(nbytes: Optional[int]) -> str:
    return f"{nbytes / 1e9:.1f} GB" if nbytes else "unknown"


def _verdict_prose(analysis: Dict[str, Any]) -> List[str]:
    """Headline plus explanation, as unwrapped paragraphs. Renderers do the formatting."""
    verdict = analysis.get("verdict")
    util = analysis.get("bw_util_pct")
    if not verdict or not util:
        return ["Not enough data to classify this setup."]
    med = util["median"]
    if verdict == "memory-bound":
        return [
            "YES -- your decode is memory-bandwidth-bound.",
            f"Token generation is running at {med:.0f}% of the memory bandwidth this machine "
            f"can actually sustain. The CPU spends most of each token waiting for weights to "
            f"arrive from RAM, not computing.",
        ]
    if verdict == "partly memory-bound":
        return [
            "PARTLY -- bandwidth matters here, but it is not the only limit.",
            f"Decode reaches {med:.0f}% of the sustainable memory bandwidth. There is real "
            f"headroom left, so compute, threading or throttling is also holding you back.",
        ]
    return [
        "NO -- something other than memory bandwidth is limiting you.",
        f"Decode is only using {med:.0f}% of this machine's memory bandwidth. This usually "
        f"means threads, throttling, a slow build, or storage rather than DRAM. See the "
        f"suggestions below.",
    ]


def _advice(analysis: Dict[str, Any]) -> List[str]:
    """Tuning advice as unwrapped single-string bullets."""
    out: List[str] = []
    verdict = analysis.get("verdict")
    util = analysis.get("bw_util_pct") or {}
    med = util.get("median")

    if verdict == "memory-bound":
        out.append("Model size is your throughput dial. Halving the bytes you load roughly "
                   "doubles tok/s: a smaller model or a lower quant buys speed almost exactly "
                   "in proportion to the bytes it removes.")
        if med and med >= 80:
            out.append("A faster CPU will not help. You are close to the memory wall, and only "
                       "fewer bytes or faster RAM moves this number.")
    elif verdict == "not bandwidth-limited":
        out.append("Check, in this order: thread count (see below), CPU thermal or power "
                   "throttling, whether the model actually fits in RAM (if it is paging from "
                   "disk, decode collapses), and whether your llama.cpp build uses the right "
                   "SIMD kernels for this CPU.")

    span = analysis.get("frontier_span")
    if span and span.get("speed_ratio") and span["size_ratio"] > 1.05:
        out.append(f"Measured on your models: {span['smallest']} is {span['size_ratio']:.2f}x "
                   f"smaller than {span['largest']} and decodes {span['speed_ratio']:.2f}x "
                   f"faster. Bytes in, tokens out.")

    knee = analysis.get("decode_thread_knee")
    if knee:
        out.append(f"Best decode thread count: {knee}. Past that, extra threads add contention, "
                   f"not throughput: decode is waiting on memory, and more waiters do not make "
                   f"the memory faster.")
    losses = [m.get("decode_loss_at_max_threads_pct") for m in analysis.get("models", [])]
    losses = [x for x in losses if x and x > 2]
    if losses:
        out.append(f"Using every thread cost up to {max(losses):.0f}% of decode throughput "
                   f"versus the best setting. Set -t explicitly; do not let it default.")

    scaling = analysis.get("prefill_scaling")
    if scaling and scaling > 1.5:
        out.append(f"Prefill is different: it scaled {scaling:.1f}x with threads (median "
                   f"across your models). Prompt processing is compute-bound, so "
                   f"long-prompt workloads DO want all your cores even though generation "
                   f"does not.")

    if not out:
        out.append("Not enough measurements to give tuning advice. Benchmark two or more models "
                   "of different sizes.")
    return out


def _bullets(items: List[str], marker: str = "*", width: int = 68) -> List[str]:
    """Wrap prose into indented terminal bullets."""
    lines: List[str] = []
    for item in items:
        wrapped = _wrap(item, width)
        lines.append(f"  {marker} {wrapped[0]}")
        lines.extend(f"    {rest}" for rest in wrapped[1:])
    return lines


def _model_table(analysis: Dict[str, Any]) -> List[str]:
    rows = [m for m in analysis.get("models", []) if m.get("ok")]
    if not rows:
        return ["  (no successful runs)"]
    head = (f"  {'model':<26}{'quant':<9}{'size':>9}{'decode':>10}"
            f"{'prefill':>10}{'thr':>5}{'GB/s':>8}{'%ceil':>7}")
    out = [head, "  " + "-" * (len(head) - 2)]
    for m in sorted(rows, key=lambda r: r.get("bytes_per_token") or 0):
        util = m.get("bw_util_pct")
        out.append(
            f"  {m['name'][:25]:<26}{str(m.get('quant') or '?')[:8]:<9}"
            f"{_gb(m.get('bytes_per_token')):>9}"
            f"{m['decode_ts']:>9.1f}t"
            f"{(m.get('prefill_ts') or 0):>9.0f}t"
            f"{m.get('decode_threads', '?'):>5}"
            f"{(m.get('decode_bw_GBs') or 0):>8.1f}"
            f"{(f'{util:.0f}%' if util else '-'):>7}"
        )
    if any(m.get("is_moe") for m in rows):
        out.append("  (*) MoE model: bytes-per-token is below the file size; excluded from the fit.")
    return out


def render(results: Dict[str, Any]) -> str:
    """Render the full terminal report card."""
    analysis = results.get("analysis", {})
    sysinfo = results.get("system", {})
    mem = results.get("membw", {}) or {}
    L: List[str] = []

    L.append(RULE)
    L.append(f"  llama-roofline v{results.get('tool_version', '?')}  --  report card")
    L.append(RULE)

    cores = sysinfo.get("physical_cores")
    logical = sysinfo.get("logical_cores")
    core_str = (f"{logical} logical" + (f" / {cores} physical" if cores else "")) if logical else "?"
    L.append(f"  Machine   : {sysinfo.get('cpu') or 'unknown CPU'}")
    L.append(f"              {core_str} cores, {_fmt_ram(sysinfo.get('ram_bytes'))} RAM, "
             f"{sysinfo.get('os', '?')} {sysinfo.get('machine', '')}".rstrip())
    peak = analysis.get("peak_read_GBs")
    if peak:
        if mem.get("source") == "measured":
            L.append(f"  Memory    : {peak:.1f} GB/s sustained read (measured: {mem.get('best_kernel')} "
                     f"kernel, {mem.get('best_threads')} threads, {mem.get('working_set_mb')} MB set)")
        else:
            L.append(f"  Memory    : {peak:.1f} GB/s "
                     f"({mem.get('source') or 'supplied via --peak-bw'})")
    else:
        L.append("  Memory    : ceiling not measured")
    build = results.get("llama_cpp", {})
    bnum = build.get("build_number")
    bcom = build.get("build_commit")
    L.append(f"  llama.cpp : build {bnum or '?'}" + (f" ({bcom})" if bcom else "")
             + (f", backends: {build['backends']}" if build.get("backends") else ""))
    L.append("")

    L.append("  IS YOUR DECODE MEMORY-BOUND?")
    L.append(THIN)
    prose = _verdict_prose(analysis)
    L.append(f"  {prose[0]}")
    for para in prose[1:]:
        L.append("")
        L.extend(f"  {line}" for line in _wrap(para, 70))
    L.append("")

    fit = analysis.get("fit")
    if fit:
        L.append("  THE ROOFLINE")
        L.append(THIN)
        L.append(f"    decode tok/s  =  {fit['bw_eff_GBs']:.2f} GB/s  /  model bytes")
        r2 = fit.get("r2")
        r2s = f"{r2:.4f}" if r2 == r2 else "n/a"
        L.append(f"    fitted across {fit['n_points']} models, R^2 = {r2s}")
        if peak:
            L.append(f"    that effective bandwidth is {100 * fit['bw_eff_GBs'] / peak:.0f}% "
                     f"of your {peak:.1f} GB/s ceiling")
        if r2 == r2 and r2 >= 0.95:
            L.append("    A fit this tight means one number -- bytes -- predicts your")
            L.append("    generation speed. You can size a model for a target tok/s.")
        L.append("")

    L.append("  WHAT TO DO ABOUT IT")
    L.append(THIN)
    L.extend(_bullets(_advice(analysis)))
    L.append("")

    L.append("  YOUR MODELS")
    L.append(THIN)
    L.extend(_model_table(analysis))
    L.append("")

    failed = [m for m in analysis.get("models", []) if not m.get("ok")]
    if failed:
        L.append("  FAILED")
        L.append(THIN)
        for m in failed:
            L.append(f"  ! {m['name']}: {m.get('error', 'no decode measurement')}")
        L.append("")

    L.append("  CAVEATS")
    L.append(THIN)
    n_warn = len(analysis.get("warnings", []))
    caveats = _caveats(results)
    L.extend(_bullets(caveats[:n_warn], marker="!"))
    L.extend(_bullets(caveats[n_warn:]))
    L.append("")
    L.append(RULE)
    return "\n".join(L)


def _wrap(text: str, width: int) -> List[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}" if cur else w
    if cur:
        lines.append(cur)
    return lines or [""]


def render_markdown(results: Dict[str, Any]) -> str:
    """A shareable Markdown version -- for pasting into an issue, gist or forum post."""
    analysis = results.get("analysis", {})
    sysinfo = results.get("system", {})
    fit = analysis.get("fit")
    util = analysis.get("bw_util_pct") or {}
    peak = analysis.get("peak_read_GBs")

    # Each fact is its own list item: consecutive plain lines would collapse into one
    # paragraph when the Markdown is rendered.
    L = ["# llama-roofline report", ""]
    L.append(f"- **Machine:** {sysinfo.get('cpu') or 'unknown CPU'}, "
             f"{sysinfo.get('logical_cores', '?')} logical cores, "
             f"{_fmt_ram(sysinfo.get('ram_bytes'))} RAM, "
             f"{sysinfo.get('os', '?')} {sysinfo.get('machine', '')}".rstrip())
    if peak:
        measured = (results.get("membw") or {}).get("source") == "measured"
        L.append(f"- **Memory ceiling:** {peak:.1f} GB/s sustained read "
                 f"({'measured' if measured else 'supplied'})")
    verdict = analysis.get("verdict")
    if verdict and util:
        L.append(f"- **Verdict:** decode is **{verdict}** "
                 f"({util['median']:.0f}% of the memory ceiling, "
                 f"range {util['min']:.0f}-{util['max']:.0f}%)")
    if fit:
        r2 = fit.get("r2")
        line = f"- **Roofline:** `decode tok/s = {fit['bw_eff_GBs']:.2f} GB/s / model_bytes`"
        if r2 == r2:
            line += f" (R^2 = {r2:.4f}, n = {fit['n_points']})"
        L.append(line)
    build = results.get("llama_cpp", {})
    if build.get("build_number"):
        L.append(f"- **llama.cpp:** build {build['build_number']}"
                 + (f" (`{build['build_commit']}`)" if build.get("build_commit") else ""))
    L.append("")
    L.append("| model | quant | size | decode tok/s | prefill tok/s | threads | GB/s | % ceiling |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for m in sorted((m for m in analysis.get("models", []) if m.get("ok")),
                    key=lambda r: r.get("bytes_per_token") or 0):
        u = m.get("bw_util_pct")
        L.append(
            f"| {m['name']}{' *(MoE)*' if m.get('is_moe') else ''} | {m.get('quant') or '?'} "
            f"| {_gb(m.get('bytes_per_token'))} | {m['decode_ts']:.1f} "
            f"| {(m.get('prefill_ts') or 0):.0f} | {m.get('decode_threads', '?')} "
            f"| {(m.get('decode_bw_GBs') or 0):.1f} | {f'{u:.0f}%' if u else '-'} |"
        )
    L.append("")
    L.append("## What this means")
    L.append("")
    for para in _verdict_prose(analysis):
        L.append(para)
        L.append("")
    for a in _advice(analysis):
        L.append(f"- {a}")
    L.append("")
    L.append("## Caveats")
    L.append("")
    for w in _caveats(results):
        L.append(f"- {w}")
    L.append("")
    L.append("---")
    L.append("Generated by [llama-roofline](https://github.com/manunicholasjacob/llama-roofline).")
    return "\n".join(L)
