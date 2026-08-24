"""Which quantization format should you run, on the core you actually have.

The usual answer to that question is a single global ranking, most often "Q4_K_M is the
balanced choice". Measurement says there is no single ranking. The same eight GGUF files,
built from the same FP16 source, order differently on an Arm Cortex-A76, an Intel
Golden Cove P-core and an Intel Gracemont E-core, and the ordering moves again with the
thread count and with the model scale.

So this module refuses to give one answer. It looks up the core, gives that core's
ranking, says how the numbers were obtained, and on silicon nobody measured it says so
and shows how to measure it.

The matrix in ``data/format_matrix.csv`` holds one row per measurement, with the file it
came from in the ``source`` column. Nothing is averaged across cores.
"""

from __future__ import annotations

import csv
import os
from typing import Any, Dict, List, Optional, Tuple

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
MATRIX_PATH = os.path.join(DATA_DIR, "format_matrix.csv")

# Formats within this fraction of the fastest are treated as the same speed, so quality
# and energy decide between them instead of measurement noise.
SPEED_TIE_PCT = 5.0
# Perplexity differences below this are inside the paired noise of the corpus used, so
# they do not decide anything.
PPL_TIE = 0.05
# Below this share of on-label bytes, a file is mostly not the format its name claims and
# any comparison involving it is partly measuring llama-quantize's substitutions.
ON_LABEL_CONFOUNDED = 60.0

# The study these numbers come from, quoted once so every report can point at it.
SOURCE_NOTE = (
    "Qwen2.5-0.5B-Instruct and Qwen2.5-1.5B-Instruct, quantized from the official FP16 "
    "GGUF, llama-bench -p 128 -n 128 with 5 repetitions. Raspberry Pi 5 energy is the "
    "PMIC rail sum from a separate identical run, uncalibrated, so relative only. "
    "Perplexity is 100 chunks of 512 tokens from a public-domain corpus, measured at "
    "0.5B on the same artifacts. Data: https://doi.org/10.5281/zenodo.21938812"
)

CONFIDENCE_TEXT = {
    "exact": "measured on this machine",
    "same-uarch": "measured on the same core design in a different chip",
    "related": "EXTRAPOLATED from a related core, not measured on yours",
    "unknown": "not measured, and not close to anything measured",
    "forced": "you named this core, so it may not be the one in this machine",
}


class MatrixMissing(RuntimeError):
    """The shipped measurement matrix could not be read."""


# --------------------------------------------------------------------------- loading

def _num(v):
    if v is None or v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def load_matrix(path: Optional[str] = None) -> List[Dict[str, Any]]:
    path = path or MATRIX_PATH
    try:
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except OSError as exc:
        raise MatrixMissing(f"could not read the format matrix at {path}: {exc}") from exc
    for r in rows:
        for k in ("params_b", "tok_s", "streamed_MiB", "file_MiB", "streamed_GBs",
                  "mJ_per_token", "perplexity", "repack_coverage_pct",
                  "on_label_pct", "bits_per_weight"):
            r[k] = _num(r.get(k))
        r["threads"] = int(r["threads"])
    if not rows:
        raise MatrixMissing(f"the format matrix at {path} is empty")
    return rows


def cores(matrix: List[Dict[str, Any]]) -> List[str]:
    seen = []
    for r in matrix:
        if r["core"] not in seen:
            seen.append(r["core"])
    return seen


def scales(matrix: List[Dict[str, Any]], core: str) -> List[float]:
    return sorted({r["params_b"] for r in matrix if r["core"] == core})


def thread_counts(matrix: List[Dict[str, Any]], core: str, params_b: float) -> List[int]:
    return sorted({r["threads"] for r in matrix
                   if r["core"] == core and r["params_b"] == params_b})


def nearest_scale(matrix: List[Dict[str, Any]], core: str,
                  params_b: Optional[float]) -> float:
    avail = scales(matrix, core)
    if params_b is None:
        return avail[0]
    return min(avail, key=lambda s: abs(s - params_b))


# --------------------------------------------------------------------------- ranking

def rows_for(matrix, core: str, params_b: float, threads: int) -> List[Dict[str, Any]]:
    sel = [r for r in matrix
           if r["core"] == core and r["params_b"] == params_b and r["threads"] == threads]
    return sorted(sel, key=lambda r: -r["tok_s"])


def best_operating_point(matrix, core: str, params_b: float) -> Optional[Tuple[int, Dict]]:
    """The (thread count, row) with the highest decode rate anywhere on this core.

    Decode saturates early, so the fastest thread count is often not the highest one,
    and telling someone to use every core can cost them throughput.
    """
    sel = [r for r in matrix if r["core"] == core and r["params_b"] == params_b]
    if not sel:
        return None
    best = max(sel, key=lambda r: r["tok_s"])
    return best["threads"], best


def pareto(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Formats that nothing else beats on both speed and quality.

    A format that is slower *and* worse than another has no case for being chosen, and
    saying so is more useful than ranking it.
    """
    keep = []
    for r in rows:
        if r.get("perplexity") is None:
            keep.append(r)
            continue
        dominated = any(
            o is not r and o.get("perplexity") is not None
            and o["tok_s"] >= r["tok_s"] and o["perplexity"] <= r["perplexity"]
            and (o["tok_s"] > r["tok_s"] or o["perplexity"] < r["perplexity"])
            for o in rows
        )
        if not dominated:
            keep.append(r)
    return keep


def envelope_deficit(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    """How far each format sits below the best streamed-byte rate on this core.

    Streamed bytes are the traffic decode actually generates, so a format that moves the
    same bytes more slowly is losing time to unpacking, not to memory.
    """
    vals = [r["streamed_GBs"] for r in rows if r.get("streamed_GBs")]
    if not vals:
        return {}
    top = max(vals)
    return {r["format"]: 100.0 * (1 - r["streamed_GBs"] / top)
            for r in rows if r.get("streamed_GBs")}


MATCHED_BYTE_TOL = 1.03


def matched_byte_group(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The largest set of formats that stream within 3% of the same bytes.

    Comparing Q8_0 against Q4_0 measures size, which everyone already knows about.
    Comparing formats that move the same bytes is the only way to see what the format
    itself costs, and it is the comparison the ranking rests on.
    """
    sized = [r for r in rows if r.get("streamed_MiB")]
    if len(sized) < 2:
        return []
    sized.sort(key=lambda r: r["streamed_MiB"])
    best: List[Dict[str, Any]] = []
    for i in range(len(sized)):
        group = [sized[i]]
        for j in range(i + 1, len(sized)):
            if sized[j]["streamed_MiB"] / sized[i]["streamed_MiB"] <= MATCHED_BYTE_TOL:
                group.append(sized[j])
            else:
                break
        if len(group) > len(best):
            best = group
    return best if len(best) >= 3 else []


def matched_byte_spread(matrix, core: str, params_b: float) -> Dict[int, Dict[str, Any]]:
    """How much the format choice is worth at each measured thread count.

    Traffic is held constant across the group, so whatever is left is compute.
    """
    out: Dict[int, Dict[str, Any]] = {}
    for t in thread_counts(matrix, core, params_b):
        rows = rows_for(matrix, core, params_b, t)
        group = matched_byte_group(rows)
        if not group:
            continue
        fastest = max(group, key=lambda r: r["tok_s"])
        slowest = min(group, key=lambda r: r["tok_s"])
        out[t] = {
            "spread_pct": 100.0 * (fastest["tok_s"] / slowest["tok_s"] - 1),
            "fastest": fastest["format"], "slowest": slowest["format"],
            "formats": [r["format"] for r in group],
            "streamed_MiB": round(sum(r["streamed_MiB"] for r in group) / len(group)),
        }
    return out


def advise_core(matrix, core: str, params_b: Optional[float] = None,
                threads: Optional[int] = None) -> Dict[str, Any]:
    """Everything worth saying about one core, as data."""
    scale = nearest_scale(matrix, core, params_b)
    avail_threads = thread_counts(matrix, core, scale)
    if not avail_threads:
        return {"core": core, "error": "no rows for this core"}

    if threads is None:
        point = best_operating_point(matrix, core, scale)
        chosen = point[0] if point else avail_threads[0]
        thread_choice = "fastest measured"
    else:
        chosen = min(avail_threads, key=lambda t: abs(t - threads))
        thread_choice = ("as requested" if chosen == threads
                         else f"nearest measured to {threads}")

    rows = rows_for(matrix, core, scale, chosen)
    deficits = envelope_deficit(rows)
    front = pareto(rows)

    out: Dict[str, Any] = {
        "core": core,
        "core_label": rows[0]["core_label"],
        "model": rows[0]["model"],
        "params_b": scale,
        "threads": chosen,
        "threads_available": avail_threads,
        "thread_choice": thread_choice,
        "normalization": rows[0]["normalization"],
        "rows": rows,
        "pareto": front,
        "envelope_deficit_pct": deficits,
        "source": rows[0]["source"],
    }

    out["fastest"] = rows[0]
    tie = [r for r in rows if r["tok_s"] >= rows[0]["tok_s"] * (1 - SPEED_TIE_PCT / 100)]
    with_ppl = [r for r in tie if r.get("perplexity") is not None]
    if with_ppl:
        floor = min(r["perplexity"] for r in with_ppl)
        # IQ4_NL and IQ4_XS land 0.001 perplexity apart on this corpus, which is nothing.
        # Picking between them on that margin would be picking on noise, so anything
        # inside PPL_TIE of the best quality is re-sorted on energy, then on speed.
        close = [r for r in with_ppl if r["perplexity"] <= floor + PPL_TIE]
        out["best_quality_at_speed"] = min(
            close, key=lambda r: (r.get("mJ_per_token") if r.get("mJ_per_token")
                                  else 0.0, -r["tok_s"]))
    else:
        out["best_quality_at_speed"] = None
    energy = [r for r in rows if r.get("mJ_per_token") is not None]
    out["lowest_energy"] = min(energy, key=lambda r: r["mJ_per_token"]) if energy else None
    out["has_energy"] = bool(energy)

    # Which of these files are mostly not the format they are named after. At 0.5B most
    # of them are not, which changes what a ranking between them means.
    mislabelled = [(r["format"], r["on_label_pct"], r["bits_per_weight"]) for r in rows
                   if r.get("on_label_pct") is not None
                   and r["on_label_pct"] < ON_LABEL_CONFOUNDED]
    out["mislabelled"] = sorted(mislabelled, key=lambda x: x[1])

    # Worst value for the bytes it moves, which is the format to warn about.
    if deficits:
        worst = max(deficits, key=deficits.get)
        if deficits[worst] >= 10.0:
            out["off_envelope"] = {"format": worst, "deficit_pct": deficits[worst]}

    # Does the ranking survive a change of thread count on this same core?
    orders = {}
    for t in avail_threads:
        orders[t] = [r["format"] for r in rows_for(matrix, core, scale, t)]
    out["ranking_by_threads"] = orders
    out["ranking_stable_across_threads"] = len({tuple(v[:3]) for v in orders.values()}) == 1
    out["matched_byte_spread"] = matched_byte_spread(matrix, core, scale)
    return out


def cross_core_facts(matrix) -> List[str]:
    """The findings that held on every core measured, computed from the matrix itself."""
    facts: List[str] = []
    scale = 0.5

    # First, because it is a property of the files rather than of a benchmark, and because
    # it changes how every number below should be read.
    at_scale = {}
    for r in matrix:
        if r["params_b"] == scale and r.get("on_label_pct") is not None:
            at_scale[r["format"]] = (r["on_label_pct"], r["bits_per_weight"])
    if at_scale:
        honest = [f for f, (pct, _) in at_scale.items() if pct >= 90]
        empty = [f for f, (pct, _) in at_scale.items() if pct < 1]
        bigger = {f: bpw for f, (pct, bpw) in at_scale.items()}
        facts.append(
            f"The label does not describe the file. Of the {len(at_scale)} formats measured "
            f"at {scale}B, built from one FP16 source, only {len(honest)} contain the type "
            f"their name claims: " + ", ".join(sorted(honest)) + ". "
            + (f"{' and '.join(sorted(empty))} contain none of it at all. "
               if empty else "")
            + (f"Q4_K_M is {at_scale['Q4_K_M'][0]:.0f}% Q4_K and stores "
               f"{bigger['Q4_K_M']:.2f} bits per weight against a nominal 4.5. "
               if "Q4_K_M" in at_scale else "")
            + "llama-quantize substitutes per tensor when a shape does not divide by the "
              "block size, and says nothing the filename can carry. At 1.5B the same "
              "recipes land at 58 to 100% on-label, so this is a small-model effect and it "
              "is worst exactly where people benchmark quickly.")

    # 1. Where the community default lands, per core, at its own best thread count.
    lines = []
    for core in cores(matrix):
        point = best_operating_point(matrix, core, scale)
        if not point:
            continue
        rows = rows_for(matrix, core, scale, point[0])
        d = envelope_deficit(rows)
        if "Q4_K_M" in d:
            lines.append(f"{rows[0]['core_label']} {d['Q4_K_M']:.0f}%")
    if lines:
        facts.append(
            "Q4_K_M, the format most model cards recommend, sits below the streamed-byte "
            "envelope on every core measured at 0.5B: " + "; ".join(lines) + ". At 1.5B it "
            "recovers, because the runtime's kernel coverage for its tensor types recovers "
            "and the 256-divisibility fallback that inflated it disappears."
        )

    # 2. The energy and quality trade, where energy exists.
    a76 = rows_for(matrix, "cortex-a76", scale, 2)
    by_fmt = {r["format"]: r for r in a76}
    if {"Q4_K_M", "IQ4_XS"} <= set(by_fmt) and by_fmt["Q4_K_M"].get("mJ_per_token"):
        km, xs = by_fmt["Q4_K_M"], by_fmt["IQ4_XS"]
        prem = 100.0 * (km["mJ_per_token"] / xs["mJ_per_token"] - 1)
        dppl = xs["perplexity"] - km["perplexity"]
        facts.append(
            f"On the Pi 5 that costs energy: Q4_K_M spends {km['mJ_per_token']:.0f} mJ per "
            f"token against IQ4_XS's {xs['mJ_per_token']:.0f}, a {prem:.0f}% premium, for "
            f"{dppl:.2f} perplexity points of quality."
        )
    if {"Q6_K", "Q4_K_M"} <= set(by_fmt) and by_fmt["Q6_K"].get("mJ_per_token"):
        q6, km = by_fmt["Q6_K"], by_fmt["Q4_K_M"]
        facts.append(
            f"Also on the Pi 5, Q6_K reaches better quality than Q4_K_M "
            f"({q6['perplexity']:.2f} against {km['perplexity']:.2f}) at the same energy "
            f"per token ({q6['mJ_per_token']:.0f} against {km['mJ_per_token']:.0f} mJ)."
        )

    facts.append(
        "The label does not describe the file. Requantizing from a Q8_0 intermediate "
        "instead of FP16 leaves the same name on a file with a different per-tensor type "
        "map, and those files decoded up to 38% slower on the Cortex-A76. Check what you "
        "downloaded with `llama-roofline inspect <model.gguf>`."
    )

    # 3. Whether the top choice actually differs between cores.
    tops = {}
    for core in cores(matrix):
        point = best_operating_point(matrix, core, scale)
        if point:
            tops[point[1]["core_label"]] = (point[1]["format"], point[0])
    if len(set(v[0] for v in tops.values())) > 1:
        pieces = [f"{lab} wants {fmt} at {t} threads" for lab, (fmt, t) in tops.items()]
        facts.append("The fastest choice is not the same on every core: " + "; ".join(pieces)
                     + ". A ranking measured on a laptop is not advice for a phone.")
    return facts


# --------------------------------------------------------------------------- rendering

RULE = "=" * 72
THIN = "-" * 72

HYBRID_NOTE = (
    "A hybrid chip runs the same binary on two microarchitectures. Which one your decode "
    "lands on is the scheduler's choice unless you pin it, and the two do not agree about "
    "formats. Both rankings are below. Without pinning, a small thread count usually ends "
    "up on the performance cores."
)

LIMITS = [
    "One model family at two scales. Qwen2.5 at 0.5B and 1.5B, so nothing here is "
    "verified on a 7B or on a different architecture, and the 0.5B and 1.5B rankings "
    "already differ from each other.",
    "Three cores. Cortex-A76, Golden Cove, Gracemont. Everything else is either an "
    "extrapolation, and labelled as one, or absent.",
    "Energy on one board only, from an uncalibrated rail sum, so treat those numbers as "
    "ratios between formats rather than absolute joules.",
    "Perplexity on 51 thousand tokens of one corpus. Format deltas exceed the paired "
    "noise, but perplexity is not downstream task accuracy.",
    "Nothing here measures your own files. Run `llama-roofline advise --measure` to get "
    "this table for the GGUFs on your disk.",
]


def _wrap(text: str, width: int = 70) -> List[str]:
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


def _bullets(items, marker="*", width=66, indent="  ") -> List[str]:
    out = []
    for item in items:
        w = _wrap(item, width)
        out.append(f"{indent}{marker} {w[0]}")
        out.extend(f"{indent}  {rest}" for rest in w[1:])
    return out


def _para(text: str, indent: str = "  ", width: int = 68) -> List[str]:
    return [f"{indent}{ln}" for ln in _wrap(text, width)]


def _table(adv: Dict[str, Any]) -> List[str]:
    rows = adv["rows"]
    deficits = adv["envelope_deficit_pct"]
    front = {id(r) for r in adv["pareto"]}
    has_energy = adv["has_energy"]
    streamed = adv["normalization"] == "streamed_bytes"

    head = f"    {'format':<9}{'tok/s':>8}{'vs best':>9}"
    head += f"{'GB/s':>8}{'off env':>9}" if streamed else f"{'file MiB':>10}"
    head += f"{'ppl':>8}"
    if has_energy:
        head += f"{'mJ/tok':>9}"
    out = [head, "    " + "-" * (len(head) - 4)]
    top = rows[0]["tok_s"]
    for r in rows:
        rel = 100.0 * (r["tok_s"] / top - 1)
        out_line = (f"    {r['format']:<9}{r['tok_s']:>8.1f}"
                    f"{('best' if rel == 0 else f'{rel:+.0f}%'):>9}")
        if streamed:
            out_line += f"{(r.get('streamed_GBs') or 0):>8.1f}"
            d = deficits.get(r["format"])
            out_line += f"{(f'-{d:.0f}%' if d and d >= 1 else '-'):>9}"
        else:
            out_line += f"{(r.get('file_MiB') or 0):>10.0f}"
        out_line += (f"{r['perplexity']:>8.2f}" if r.get("perplexity") else f"{'-':>8}")
        if has_energy:
            out_line += (f"{r['mJ_per_token']:>9.0f}" if r.get("mJ_per_token")
                         else f"{'-':>9}")
        if id(r) in front:
            out_line += "  <"
        out.append(out_line)
    out.append("")
    out.append("    A < marks a format nothing else beats on both speed and quality.")
    if streamed:
        out.append("    GB/s is over bytes actually streamed per token rather than file")
        out.append("    size, so 'off env' is time lost to unpacking, not to memory.")
    return out


def render_core(adv: Dict[str, Any], confidence: str = "exact", why: str = "") -> List[str]:
    """The report block for one core."""
    L: List[str] = []
    L.append(f"  {adv['core_label'].upper()}, {adv['threads']} THREADS, "
             f"{adv['params_b']}B CLASS")
    L.append(THIN)
    evidence = CONFIDENCE_TEXT.get(confidence, confidence)
    L.extend(_para(f"Evidence: {evidence}."
                   + (f" {why[0].upper()}{why[1:]}." if why else ""),
                   indent="    ", width=66))
    if adv["thread_choice"] != "as requested":
        L.append(f"    Threads: {adv['threads']} ({adv['thread_choice']}; measured at "
                 f"{', '.join(str(t) for t in adv['threads_available'])})")
    L.append("")
    L.extend(_table(adv))
    L.append("")

    picks = []
    f = adv["fastest"]
    picks.append(f"Fastest: {f['format']} at {f['tok_s']:.1f} tok/s.")
    q = adv.get("best_quality_at_speed")
    if q and q["format"] != f["format"]:
        picks.append(f"Best quality without giving up speed: {q['format']}, perplexity "
                     f"{q['perplexity']:.2f} against {f['format']}'s "
                     f"{f['perplexity']:.2f}, still within {SPEED_TIE_PCT:.0f}% of the "
                     f"fastest.")
    elif q:
        picks.append(f"{f['format']} is also the best quality inside that speed band. "
                     f"That does not happen on every core.")
    e = adv.get("lowest_energy")
    if e:
        picks.append(f"Lowest energy per token: {e['format']} at {e['mJ_per_token']:.0f} mJ.")
    else:
        picks.append("Energy: not measured here. Package power needs a rail sensor and "
                     "this platform does not expose one.")
    off = adv.get("off_envelope")
    if off:
        picks.append(f"Worst value for its bytes: {off['format']}, "
                     f"{off['deficit_pct']:.0f}% below the streaming envelope on this "
                     f"core. It moves the bytes. It spends longer unpacking them.")
    if not adv["ranking_stable_across_threads"]:
        bits = [f"{t} threads: {v[0]}" for t, v in sorted(adv["ranking_by_threads"].items())]
        picks.append("The winner changes with thread count on this core (" +
                     ", ".join(bits) + "), so set -t deliberately rather than letting it "
                     "default.")
    mislabelled = adv.get("mislabelled") or []
    if mislabelled:
        worst = mislabelled[0]
        picks.append(
            "Read this table knowing what is in the files. "
            + ", ".join(f"{f} {p:.0f}%" for f, p, _ in mislabelled[:4])
            + " on-label by repeating-layer bytes, so a ranking between them is partly a "
              "ranking of what llama-quantize substituted rather than of the formats. "
            + (f"{worst[0]} contains no {worst[0].rsplit('_', 1)[0]} tensors at all. "
               if worst[1] < 1 else "")
            + "Run `llama-roofline inspect` on any file before quoting a comparison "
              "that involves it.")

    spread = adv.get("matched_byte_spread") or {}
    if spread:
        group = next(iter(spread.values()))
        bits = [f"{v['spread_pct']:.0f}% at {t} threads" for t, v in sorted(spread.items())]
        picks.append(
            "Holding bytes constant, format choice alone is worth " + ", ".join(bits) +
            f" here. That comparison uses {', '.join(group['formats'])}, all of which "
            f"stream about {group['streamed_MiB']} MiB per token, so the difference is "
            f"unpacking work rather than traffic."
        )
    L.extend(_bullets(picks, indent="    "))
    return L


def _unknown_silicon(detection: Dict[str, Any], matrix) -> List[str]:
    L = ["  NO MEASUREMENT FOR THIS SILICON", THIN]
    L.extend(_para(
        "The rankings that ship with this tool were measured on an Arm Cortex-A76, an "
        "Intel Golden Cove P-core and an Intel Gracemont E-core. This machine is none of "
        "those and is not a close relative of one. Quoting those numbers at you would be "
        "a guess wearing the clothes of advice."))
    L.append("")
    L.append("  MEASURE IT YOURSELF")
    L.append(THIN)
    L.append("    llama-roofline advise --measure --models ~/models/*.gguf")
    L.append("")
    L.extend(_para(
        "That benchmarks the formats already on this machine and prints the same table "
        "from your own numbers. Several quantizations of one model is the useful case. "
        "Different models tell you about size, not about format."))
    L.append("")
    L.extend(_para(
        "If you do not have several quantizations of one model, this prints the commands "
        "that build a comparable set from an unquantized one, and explains why the source "
        "has to be unquantized:"))
    L.append("")
    L.append("    llama-roofline advise --plan ~/models/<model>-f16.gguf")
    L.append("")
    L.extend(_para(
        "Whatever it says about this machine is a result nobody has, because this core "
        "has not been measured. " + GALLERY_URL))
    L.append("")
    L.append("  WHAT IS STILL WORTH KNOWING")
    L.append(THIN)
    L.extend(_bullets(cross_core_facts(matrix)))
    L.append("")
    return L


def render(detection: Dict[str, Any], advices, matrix, tool_version: str = "") -> str:
    """Full terminal report. ``advices`` pairs each detected core with its advice."""
    from . import silicon

    L = [RULE, f"  llama-roofline {tool_version}  --  which quantization should you run?",
         RULE]
    machine = _wrap(silicon.summary_line(detection), 60)
    L.append(f"  Machine : {machine[0]}")
    L.extend(f"            {ln}" for ln in machine[1:])
    if detection.get("detail"):
        L.extend(_para(detection["detail"], indent="            ", width=58))
    L.append("")

    if not advices:
        L.extend(_unknown_silicon(detection, matrix))
        L.append(RULE)
        return "\n".join(L)

    if len(advices) > 1:
        L.append("  TWO KINDS OF CORE, AND THEY WANT DIFFERENT FORMATS")
        L.append(THIN)
        L.extend(_para(HYBRID_NOTE))
        L.append("")

    for core_info, adv in advices:
        L.extend(render_core(adv, core_info.get("confidence", "exact"),
                             core_info.get("why", "")))
        L.append("")

    L.append("  WHAT HELD ON EVERY CORE MEASURED")
    L.append(THIN)
    L.extend(_bullets(cross_core_facts(matrix)))
    L.append("")
    L.append("  WHAT THIS IS NOT")
    L.append(THIN)
    L.extend(_bullets(LIMITS))
    L.append("")
    L.append("  WHERE THE NUMBERS CAME FROM")
    L.append(THIN)
    L.extend(_para(SOURCE_NOTE))
    L.append("")
    L.append(RULE)
    return "\n".join(L)


def render_markdown(detection: Dict[str, Any], advices, matrix,
                    tool_version: str = "") -> str:
    from . import silicon

    L = ["# Which quantization format to run", ""]
    L.append(f"- **Machine:** {silicon.summary_line(detection)}")
    if detection.get("detail"):
        L.append(f"- **Note:** {detection['detail']}")
    L.append(f"- **Advisor:** llama-roofline {tool_version}")
    L.append("")
    if not advices:
        L.append("No measurements exist for this silicon, so this tool has no ranking to "
                 "offer. Run `llama-roofline advise --measure --models <your gguf files>` "
                 "to produce one from this machine.")
        L.append("")
    for core_info, adv in advices:
        L.append(f"## {adv['core_label']}, {adv['threads']} threads, "
                 f"{adv['params_b']}B class")
        L.append("")
        L.append(f"Evidence: "
                 f"{CONFIDENCE_TEXT.get(core_info.get('confidence', 'exact'))}."
                 + (f" {core_info.get('why', '')}" if core_info.get("why") else ""))
        L.append("")
        streamed = adv["normalization"] == "streamed_bytes"
        cols = ["format", "tok/s"]
        cols += ["GB/s streamed", "below envelope"] if streamed else ["file MiB"]
        cols += ["perplexity"]
        if adv["has_energy"]:
            cols.append("mJ/token")
        L.append("| " + " | ".join(cols) + " |")
        L.append("|" + "|".join(["---"] + ["---:"] * (len(cols) - 1)) + "|")
        d = adv["envelope_deficit_pct"]
        for r in adv["rows"]:
            cells = [r["format"], f"{r['tok_s']:.1f}"]
            if streamed:
                cells.append(f"{r.get('streamed_GBs') or 0:.1f}")
                dd = d.get(r["format"])
                cells.append(f"{dd:.0f}%" if dd and dd >= 1 else "-")
            else:
                cells.append(f"{r.get('file_MiB') or 0:.0f}")
            cells.append(f"{r['perplexity']:.2f}" if r.get("perplexity") else "-")
            if adv["has_energy"]:
                cells.append(f"{r['mJ_per_token']:.0f}" if r.get("mJ_per_token") else "-")
            L.append("| " + " | ".join(cells) + " |")
        L.append("")
        f = adv["fastest"]
        L.append(f"Fastest here is **{f['format']}** at {f['tok_s']:.1f} tok/s.")
        q = adv.get("best_quality_at_speed")
        if q and q["format"] != f["format"]:
            L.append(f"**{q['format']}** reaches better quality (perplexity "
                     f"{q['perplexity']:.2f} against {f['perplexity']:.2f}) inside "
                     f"{SPEED_TIE_PCT:.0f}% of that speed.")
        e = adv.get("lowest_energy")
        if e:
            L.append(f"Lowest energy per token is **{e['format']}** at "
                     f"{e['mJ_per_token']:.0f} mJ.")
        L.append("")
    L.append("## What held on every core measured")
    L.append("")
    for fact in cross_core_facts(matrix):
        L.append(f"- {fact}")
    L.append("")
    L.append("## What this is not")
    L.append("")
    for lim in LIMITS:
        L.append(f"- {lim}")
    L.append("")
    L.append("## Where the numbers came from")
    L.append("")
    L.append(SOURCE_NOTE)
    L.append("")
    L.append("---")
    L.append("Generated by [llama-roofline]"
             "(https://github.com/manunicholasjacob/llama-roofline).")
    return "\n".join(L)


# ------------------------------------------------------------------- measured locally

def _group_by_threads(measured: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    out: Dict[int, List[Dict[str, Any]]] = {}
    for r in measured:
        out.setdefault(r["threads"], []).append(r)
    for t in out:
        out[t].sort(key=lambda r: -r["tok_s"])
    return out


def local_matched_spread(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The same matched-byte comparison, over whatever the user happens to have."""
    sized = [r for r in rows if r.get("streamed_MiB")]
    if len(sized) < 2:
        return None
    sized.sort(key=lambda r: r["streamed_MiB"])
    best: List[Dict[str, Any]] = []
    for i in range(len(sized)):
        group = [sized[i]]
        for j in range(i + 1, len(sized)):
            if sized[j]["streamed_MiB"] / sized[i]["streamed_MiB"] <= MATCHED_BYTE_TOL:
                group.append(sized[j])
            else:
                break
        if len(group) > len(best):
            best = group
    if len(best) < 2:
        return None
    fast = max(best, key=lambda r: r["tok_s"])
    slow = min(best, key=lambda r: r["tok_s"])
    return {
        "spread_pct": 100.0 * (fast["tok_s"] / slow["tok_s"] - 1),
        "fastest": fast["format"], "slowest": slow["format"],
        "formats": [r["format"] for r in best],
        "streamed_MiB": round(sum(r["streamed_MiB"] for r in best) / len(best)),
    }


MEASURED_CAVEATS = [
    "These are your files on your machine, so they are the only numbers here that are "
    "certainly about you. They are also throughput only: nothing was measured about "
    "output quality or energy.",
    "A comparison between files of different sizes mostly measures size. The line "
    "marked 'same bytes' below is the one that isolates the format, and it only appears "
    "when two or more of your files stream within 3% of the same bytes per token.",
    "Run this on an idle machine. Another process competing for memory depresses every "
    "number and the ordering can move with it.",
]


def render_measured(system: Dict[str, Any], measured: List[Dict[str, Any]],
                    threads: List[int]) -> str:
    """Report the ranking measured on this machine, in the same shape as the shipped one."""
    L = [RULE, "  measured on this machine", RULE]
    L.append(f"  Machine : {system.get('cpu') or 'unknown CPU'}, "
             f"{system.get('logical_cores', '?')} logical cores")
    L.append("")
    groups = _group_by_threads(measured)
    for t in sorted(groups):
        rows = groups[t]
        L.append(f"  {t} THREAD{'S' if t != 1 else ''}")
        L.append(THIN)
        has_stream = any(r.get("streamed_GBs") for r in rows)
        head = f"    {'file':<28}{'format':<9}{'tok/s':>8}{'vs best':>9}"
        if has_stream:
            head += f"{'MiB/tok':>9}{'GB/s':>8}"
        L.append(head)
        L.append("    " + "-" * (len(head) - 4))
        top = rows[0]["tok_s"]
        for r in rows:
            rel = 100.0 * (r["tok_s"] / top - 1)
            line = (f"    {r['name'][:27]:<28}{str(r['format'])[:8]:<9}{r['tok_s']:>8.1f}"
                    f"{('best' if rel == 0 else f'{rel:+.0f}%'):>9}")
            if has_stream:
                line += (f"{(r.get('streamed_MiB') or 0):>9.0f}"
                         f"{(r.get('streamed_GBs') or 0):>8.1f}")
            L.append(line)
        spread = local_matched_spread(rows)
        if spread:
            L.append("")
            L.extend(_para(
                f"Same bytes: {', '.join(spread['formats'])} all stream about "
                f"{spread['streamed_MiB']} MiB per token, and {spread['fastest']} is "
                f"{spread['spread_pct']:.0f}% faster than {spread['slowest']} anyway. "
                f"That difference is unpacking work, not memory traffic.",
                indent="    ", width=66))
        L.append("")

    winners = {t: g[0]["format"] for t, g in groups.items()}
    if len(set(winners.values())) > 1:
        bits = [f"{t} threads: {f}" for t, f in sorted(winners.items())]
        L.extend(_para("The fastest file changes with thread count here (" +
                       ", ".join(bits) + "), so pick -t and the format together."))
        L.append("")
    if not any(local_matched_spread(rows) for rows in groups.values()):
        L.append("  THIS IS A SIZE COMPARISON, NOT A FORMAT COMPARISON")
        L.append(THIN)
        L.extend(_para(NO_MATCHED_BYTES))
        L.append("")
    L.append("  READ THIS BEFORE QUOTING IT")
    L.append(THIN)
    L.extend(_bullets(MEASURED_CAVEATS))
    L.append("")
    L.append("  WHERE TO SEND IT")
    L.append(THIN)
    L.extend(_para(
        "Three CPU cores have been measured for the shipped rankings. If this machine is "
        "not one of them, or if it is and your ordering disagrees, that is worth more "
        "than another confirmation. Re-run with --markdown result.md and open an issue:"))
    L.append("")
    L.append(f"    {GALLERY_URL}")
    L.append("")
    L.append(RULE)
    return "\n".join(L)


def render_measured_markdown(system: Dict[str, Any], measured: List[Dict[str, Any]],
                             threads: List[int]) -> str:
    L = ["# Quantization formats measured on this machine", ""]
    L.append(f"- **Machine:** {system.get('cpu') or 'unknown CPU'}, "
             f"{system.get('logical_cores', '?')} logical cores")
    L.append("")
    groups = _group_by_threads(measured)
    for t in sorted(groups):
        rows = groups[t]
        has_stream = any(r.get("streamed_GBs") for r in rows)
        L.append(f"## {t} thread{'s' if t != 1 else ''}")
        L.append("")
        cols = ["file", "format", "tok/s"]
        if has_stream:
            cols += ["MiB per token", "GB/s streamed"]
        L.append("| " + " | ".join(cols) + " |")
        L.append("|" + "|".join(["---"] + ["---:"] * (len(cols) - 1)) + "|")
        for r in rows:
            cells = [r["name"], str(r["format"]), f"{r['tok_s']:.1f}"]
            if has_stream:
                cells += [f"{r.get('streamed_MiB') or 0:.0f}",
                          f"{r.get('streamed_GBs') or 0:.1f}"]
            L.append("| " + " | ".join(cells) + " |")
        spread = local_matched_spread(rows)
        if spread:
            L.append("")
            L.append(f"{', '.join(spread['formats'])} stream about "
                     f"{spread['streamed_MiB']} MiB per token each, and "
                     f"**{spread['fastest']}** is {spread['spread_pct']:.0f}% faster than "
                     f"{spread['slowest']} at the same traffic.")
        L.append("")
    L.append("## Read this before quoting it")
    L.append("")
    for c in MEASURED_CAVEATS:
        L.append(f"- {c}")
    L.append("")
    L.append("---")
    L.append("Generated by [llama-roofline]"
             "(https://github.com/manunicholasjacob/llama-roofline) with "
             "`llama-roofline advise --measure`.")
    return "\n".join(L)


# ---------------------------------------------------------------- the comparison plan

GALLERY_URL = ("https://github.com/manunicholasjacob/llama-roofline/issues/new"
               "?template=results-gallery.md")

# The set the shipped study used. Q4_0, IQ4_NL, IQ4_XS and Q3_K_M land within 3% of the
# same streamed bytes at 0.5B, which is what makes a format comparison a format
# comparison rather than a size comparison. Q4_K_M is in because it is the community
# default and the one most likely to surprise; Q6_K and Q8_0 anchor the top of the range.
PLAN_FORMATS = ["Q4_0", "IQ4_NL", "IQ4_XS", "Q3_K_M", "Q4_K_M", "Q6_K", "Q8_0"]

UNQUANTIZED = ("F32", "F16", "BF16")


def quantize_plan(source_path: str, quant_label: Optional[str],
                  formats: Optional[List[str]] = None) -> List[str]:
    """Commands that build a comparable format set from one unquantized model.

    The insistence on an unquantized source is the whole point. Requantizing from a
    Q8_0 intermediate is a single extra flag and it produces files that carry the same
    format label, the same nominal bit width, and a different per-tensor type map. On
    the Cortex-A76 those files decoded up to 38% slower than their FP16-sourced
    counterparts. A comparison built on them measures the intermediate, not the format.
    """
    import os as _os

    formats = formats or PLAN_FORMATS
    stem = _os.path.splitext(_os.path.basename(source_path))[0]
    for suffix in ("-fp16", "-f16", ".fp16", ".f16", "-bf16", ".bf16",
                   "-fp32", "-f32", ".fp32", ".f32"):
        if stem.lower().endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    directory = _os.path.dirname(_os.path.abspath(source_path))

    def out_path(name):
        # Match the separator the caller typed, so the commands can be pasted back.
        joined = _os.path.join(directory, name)
        return joined.replace("\\", "/") if "/" in source_path else joined

    L: List[str] = [RULE, "  a format comparison you can actually run", RULE]
    L.extend(_para(f"Source: {_os.path.basename(source_path)}"
                   + (f", which reports itself as {quant_label}" if quant_label else "")))
    L.append("")

    if quant_label and quant_label.upper() not in UNQUANTIZED:
        L.append("  THIS FILE IS THE WRONG STARTING POINT")
        L.append(THIN)
        L.extend(_para(
            f"It is already quantized to {quant_label}. Building the comparison set from "
            f"it needs llama-quantize's --allow-requantize flag, and the files that come "
            f"out carry the same format labels as a proper set while holding different "
            f"per-tensor types. On the Cortex-A76 those requantized files decoded up to "
            f"38% slower than their FP16-sourced counterparts under an identical name. "
            f"A comparison built on them measures what the file was made from rather "
            f"than what the format costs."))
        L.append("")
        L.extend(_para(
            "Download the F16 or BF16 GGUF of the same model instead, or convert it from "
            "the original weights with llama.cpp's convert_hf_to_gguf.py, and run this "
            "again pointing at that file. It is usually the largest file in the same "
            "repository and it is the only one every other format can be derived from "
            "honestly."))
        L.append("")
        L.append(RULE)
        return L

    L.append("  1. BUILD THE SET")
    L.append(THIN)
    L.extend(_para(
        "Each command takes about a minute for a small model. llama-quantize comes from "
        "the same llama.cpp build as llama-bench.", width=66))
    L.append("")
    for fmt in formats:
        out = out_path(f"{stem}-{fmt}.gguf")
        L.append(f"    llama-quantize {source_path} {out} {fmt}")
    L.append("")
    L.append("  2. MEASURE THEM")
    L.append(THIN)
    L.append("    llama-roofline advise --measure --models "
             + out_path(stem) + "-*.gguf")
    L.append("")
    L.extend(_para(
        "That benchmarks each file across a thread sweep and prints the ranking for this "
        "machine, with a line isolating the formats that stream the same bytes so the "
        "comparison is about the format rather than about the size."))
    L.append("")
    L.append("  3. SEND IT")
    L.append(THIN)
    L.extend(_para(
        "The rankings that ship with this tool cover three CPU cores. If yours is not one "
        "of them, or if it is and your ordering disagrees, that is the result worth "
        "having. Add --markdown result.md to step 2 and open an issue with it:"))
    L.append("")
    L.append(f"    {GALLERY_URL}")
    L.append("")
    L.append(RULE)
    return L


NO_MATCHED_BYTES = (
    "None of the files measured stream within 3% of the same bytes per token, so the "
    "table above is mostly telling you about size, which you already knew. To find out "
    "what the format itself costs, build several formats from one unquantized model and "
    "measure those: llama-roofline advise --plan <model-f16.gguf> prints the commands."
)
