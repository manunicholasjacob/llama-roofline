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

# Used instead of the first caveat above when the tensor table was parsed and the
# embedding table excluded, which is the more accurate accounting of decode traffic.
STREAMED_CAVEAT = (
    "bytes-per-token is the sum of every tensor read on each token: the repeating layers "
    "and the output head. The token embedding is excluded, because decode looks up one "
    "row of it rather than streaming it, unless the model ties the embedding to the "
    "output head, in which case it is streamed in full and counted. That is exact for a "
    "dense transformer at short context and an overestimate once the KV cache grows "
    "large, or for MoE models."
)

# A fitted bandwidth from this tool is not comparable with one from a paper unless the
# operating point is stated. summarize_model() takes each model at its best decode thread
# count across the sweep, so the fit sits on the upper envelope. The same seven models at a
# fixed thread count give 35.73 GB/s against this tool's 37.65 on the same laptop. Both are
# right and they answer different questions, so every surface that prints the number prints
# this too.
FIT_OPERATING_POINT = "each model at its own best thread count"

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
    if any((m.get("bytes_source") or "").startswith("streamed")
           for m in analysis.get("models", [])):
        standing[0] = STREAMED_CAVEAT
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
        L.append(f"    fitted across {fit['n_points']} models, R^2 = {r2s},")
        L.append(f"    taking {FIT_OPERATING_POINT}")
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
            line += f" (R^2 = {r2:.4f}, n = {fit['n_points']}, {FIT_OPERATING_POINT})"
        else:
            line += f" ({FIT_OPERATING_POINT})"
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


# --------------------------------------------------------------------------- diagnosis

# Same thresholds the analysis uses to classify, restated here because the diagnosis
# talks about one model rather than the median across them.
HARD_BOUND_PCT_TEXT = 70.0
PARTLY_BOUND_PCT_TEXT = 40.0


def _tok_ceiling(model: Dict[str, Any], peak_GBs: Optional[float]) -> Optional[float]:
    """Tokens per second this model could reach if decode hit the memory ceiling."""
    b = model.get("bytes_per_token")
    if not b or not peak_GBs:
        return None
    return peak_GBs * 1e9 / b


def _headline(model: Dict[str, Any], analysis: Dict[str, Any]) -> List[str]:
    """Two or three sentences that answer the question the user actually asked."""
    util = model.get("bw_util_pct")
    peak = analysis.get("peak_read_GBs")
    out: List[str] = []
    if util is None:
        out.append("Decode measured, but with no memory ceiling to compare it against "
                   "there is no verdict. Re-run without --skip-membw.")
        return out

    ceiling = _tok_ceiling(model, peak)
    if util >= HARD_BOUND_PCT_TEXT:
        out.append(f"Yes. Decode is memory-bandwidth-bound, at {util:.0f}% of what this "
                   f"machine sustains.")
        out.append("Every token reads the whole model out of RAM, and at this utilisation "
                   "the cores are mostly waiting for it. A faster CPU changes nothing "
                   "here. Fewer bytes does.")
    elif util >= PARTLY_BOUND_PCT_TEXT:
        out.append(f"Partly. Decode reaches {util:.0f}% of the sustainable memory "
                   f"bandwidth, so bandwidth matters but is not the whole story.")
        out.append("Something else is taking the rest: thread count, throttling, or a "
                   "build without the right kernels for this CPU.")
    else:
        out.append(f"No. Decode is using only {util:.0f}% of this machine's memory "
                   f"bandwidth, so bandwidth is not what is holding you back.")
        out.append("Look at thread count first, then thermal or power throttling, then "
                   "whether the model actually fits in RAM. A model paging off disk "
                   "produces exactly this shape.")
    if ceiling:
        out.append(f"For {model['name']} the ceiling on this machine is about "
                   f"{ceiling:.0f} tok/s. You measured {model['decode_ts']:.1f} at "
                   f"{model.get('decode_threads', '?')} threads.")
    return out


def _thread_verdict(model: Dict[str, Any]) -> Optional[str]:
    runs = [r for r in model.get("runs", []) if r.get("decode_ts")]
    if len(runs) < 2:
        return None
    best = max(runs, key=lambda r: r["decode_ts"])
    top = max(runs, key=lambda r: r["threads"])
    knee = model.get("decode_thread_knee") or best["threads"]
    if top["threads"] == best["threads"]:
        return (f"Threads: decode was still improving at {best['threads']}, the highest "
                f"count measured, so try more if you have them.")
    loss = 100.0 * (1 - top["decode_ts"] / best["decode_ts"])
    if loss < 2:
        return (f"Threads: {knee} is enough. Going to {top['threads']} changed decode by "
                f"under 2%, so the extra cores are doing nothing for generation.")
    return (f"Threads: use {best['threads']}. Running {top['threads']} cost "
            f"{loss:.0f}% of decode throughput, because decode is waiting on memory and "
            f"more waiters do not make memory faster.")


def _streamed_note(model: Dict[str, Any]) -> Optional[str]:
    s = model.get("streamed")
    if not s:
        return None
    total, streamed = s["total_bytes"], s["streamed_bytes"]
    if not total or not streamed:
        return None
    gap = 100.0 * (1 - streamed / total)
    if gap < 3:
        return None
    return (f"This model reads {streamed / 1e6:.0f} MB per token, not the "
            f"{total / 1e6:.0f} MB the file weighs: the token embedding is a row lookup, "
            f"so {gap:.0f}% of the file never moves during generation. The output head "
            f"alone is {s['head_share_pct']:.0f}% of what does.")


def _fmt_bytes(n: Optional[float]) -> str:
    if not n:
        return "?"
    return f"{n / 1e9:.2f} GB" if n >= 1e9 else f"{n / 1e6:.0f} MB"


def render_diagnosis(results: Dict[str, Any]) -> str:
    """The zero-configuration report: what is limiting this setup, in plain words."""
    analysis = results.get("analysis", {}) or {}
    sysi = results.get("system", {}) or {}
    mem = results.get("membw", {}) or {}
    models = [m for m in analysis.get("models", []) if m.get("ok")]
    peak = analysis.get("peak_read_GBs")

    L: List[str] = [RULE]
    L.append(f"  llama-roofline v{results.get('tool_version', '?')}  --  diagnosis")
    L.append(RULE)
    cores = sysi.get("logical_cores")
    phys = sysi.get("physical_cores")
    L.append(f"  Machine   : {sysi.get('cpu') or 'unknown CPU'}")
    L.append(f"              {cores or '?'} logical"
             + (f" / {phys} physical" if phys else "") + " cores, "
             + _fmt_ram(sysi.get("ram_bytes")) + " RAM, "
             + f"{sysi.get('os', '?')} {sysi.get('machine', '')}".rstrip())
    if peak:
        if mem.get("source") == "measured":
            L.append(f"  Memory    : {peak:.1f} GB/s sustained read (measured: "
                     f"{mem.get('best_kernel')} kernel, {mem.get('best_threads')} threads)")
        else:
            L.append(f"  Memory    : {peak:.1f} GB/s ({mem.get('source', 'supplied')})")
    else:
        L.append("  Memory    : ceiling not measured, so there is nothing to compare against")
    build = results.get("llama_cpp", {}) or {}
    if build.get("build_number"):
        L.append(f"  llama.cpp : build {build['build_number']}"
                 + (f" ({build['build_commit']})" if build.get("build_commit") else "")
                 + (f", backends: {build['backends']}" if build.get("backends") else ""))
    L.append("")

    if not models:
        L.append("  Nothing was measured. Every model failed to benchmark; the errors are")
        L.append("  above and in the results JSON.")
        L.append(RULE)
        return "\n".join(L)

    primary = max(models, key=lambda m: m.get("bw_util_pct") or 0)
    L.append("  THE ANSWER")
    L.append(THIN)
    for para in _headline(primary, analysis):
        L.extend(f"  {ln}" for ln in _wrap(para, 70))
        L.append("")

    L.append("  WHAT TO CHANGE")
    L.append(THIN)
    advice: List[str] = []
    tv = _thread_verdict(primary)
    if tv:
        advice.append(tv)
    util = primary.get("bw_util_pct")
    if util and util >= HARD_BOUND_PCT_TEXT:
        advice.append("Size is the only dial that moves this. Halving the bytes read per "
                      "token roughly doubles tok/s, which is why a smaller model or a "
                      "lower quantization buys speed almost exactly in proportion to the "
                      "bytes it removes.")
        advice.append("Not every format of the same size costs the same, though, and "
                      "which one wins depends on your core. Run `llama-roofline advise`.")
    span = analysis.get("frontier_span")
    if span and span.get("speed_ratio") and span["size_ratio"] > 1.05:
        advice.append(f"Measured here: {span['smallest']} is {span['size_ratio']:.2f}x "
                      f"smaller than {span['largest']} and decodes "
                      f"{span['speed_ratio']:.2f}x faster.")
    scaling = analysis.get("prefill_scaling")
    if scaling and scaling > 1.5:
        advice.append(f"Prompt processing is the opposite case: it scaled {scaling:.1f}x "
                      f"with threads. If your workload is long prompts and short answers, "
                      f"you do want the cores.")
    sn = _streamed_note(primary)
    if sn:
        advice.append(sn)
    L.extend(_bullets(advice))
    L.append("")

    L.append("  WHAT WAS MEASURED")
    L.append(THIN)
    L.extend(_model_table(analysis))
    L.append("")

    fit = analysis.get("fit")
    if fit:
        r2 = fit.get("r2")
        L.append("  THE ROOFLINE")
        L.append(THIN)
        L.append(f"    decode tok/s  =  {fit['bw_eff_GBs']:.2f} GB/s  /  bytes per token")
        L.append(f"    fitted across {fit['n_points']} models, "
                 f"R^2 = {f'{r2:.4f}' if r2 == r2 else 'n/a'},")
        L.append(f"    taking {FIT_OPERATING_POINT}")
        if r2 == r2 and r2 >= 0.95:
            L.append("    One number predicts your generation speed, so you can size a")
            L.append("    model for a target tok/s instead of guessing.")
        L.append("")

    failed = [m for m in analysis.get("models", []) if not m.get("ok")]
    if failed:
        L.append("  DID NOT RUN")
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


def render_diagnosis_markdown(results: Dict[str, Any]) -> str:
    """The same diagnosis, shaped for pasting into an issue or a forum thread."""
    analysis = results.get("analysis", {}) or {}
    sysi = results.get("system", {}) or {}
    models = [m for m in analysis.get("models", []) if m.get("ok")]
    peak = analysis.get("peak_read_GBs")

    L = ["# llama-roofline diagnosis", ""]
    L.append(f"- **Machine:** {sysi.get('cpu') or 'unknown CPU'}, "
             f"{sysi.get('logical_cores', '?')} logical cores, "
             f"{_fmt_ram(sysi.get('ram_bytes'))} RAM, "
             f"{sysi.get('os', '?')} {sysi.get('machine', '')}".rstrip())
    if peak:
        measured = (results.get("membw") or {}).get("source") == "measured"
        L.append(f"- **Memory ceiling:** {peak:.1f} GB/s sustained read "
                 f"({'measured' if measured else 'supplied'})")
    build = results.get("llama_cpp", {}) or {}
    if build.get("build_number"):
        L.append(f"- **llama.cpp:** build {build['build_number']}"
                 + (f" (`{build['build_commit']}`)" if build.get("build_commit") else ""))
    L.append("")
    if not models:
        L.append("Nothing was measured; every model failed to benchmark.")
        return "\n".join(L)

    primary = max(models, key=lambda m: m.get("bw_util_pct") or 0)
    L.append("## The answer")
    L.append("")
    for para in _headline(primary, analysis):
        L.append(para)
        L.append("")

    L.append("## What was measured")
    L.append("")
    L.append("| model | quant | bytes/token | decode tok/s | prefill tok/s | threads "
             "| GB/s | % of ceiling |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for m in sorted(models, key=lambda r: r.get("bytes_per_token") or 0):
        u = m.get("bw_util_pct")
        L.append(
            f"| {m['name']} | {m.get('quant') or '?'} | {_gb(m.get('bytes_per_token'))} "
            f"| {m['decode_ts']:.1f} | {(m.get('prefill_ts') or 0):.0f} "
            f"| {m.get('decode_threads', '?')} | {(m.get('decode_bw_GBs') or 0):.1f} "
            f"| {f'{u:.0f}%' if u else '-'} |")
    L.append("")
    bs = primary.get("bytes_source")
    if bs:
        L.append(f"Bytes per token from: {bs}.")
        L.append("")

    advice = []
    tv = _thread_verdict(primary)
    if tv:
        advice.append(tv)
    sn = _streamed_note(primary)
    if sn:
        advice.append(sn)
    if advice:
        L.append("## What to change")
        L.append("")
        for a in advice:
            L.append(f"- {a}")
        L.append("")

    fit = analysis.get("fit")
    if fit:
        r2 = fit.get("r2")
        L.append(f"**Roofline:** `decode tok/s = {fit['bw_eff_GBs']:.2f} GB/s / "
                 f"bytes_per_token`"
                 + (f" (R^2 = {r2:.4f}, n = {fit['n_points']}, {FIT_OPERATING_POINT})"
                    if r2 == r2 else f" ({FIT_OPERATING_POINT})"))
        L.append("")
    L.append("## Caveats")
    L.append("")
    for w in _caveats(results):
        L.append(f"- {w}")
    L.append("")
    L.append("---")
    L.append("Generated by [llama-roofline]"
             "(https://github.com/manunicholasjacob/llama-roofline) with "
             "`llama-roofline diagnose`.")
    return "\n".join(L)
