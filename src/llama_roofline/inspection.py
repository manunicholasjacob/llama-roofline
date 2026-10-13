"""What ``inspect`` says about a GGUF file: the summary and its text report.

Kept apart from the command line, and dependent on nothing but ``gguf.py`` and the
standard library, so that the browser version of the inspector (Pyodide, on the docs
site and the Hugging Face Space) runs this exact file and prints exactly what the CLI
prints. Copy it there next to ``gguf.py``; both work as flat modules too.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

try:  # inside the package
    from . import gguf
except ImportError:  # vendored next to gguf.py, as in the browser inspector
    import gguf  # type: ignore[no-redef]

RULE = "=" * 72
THIN = "-" * 72

# (metadata, tensors, size of that part in bytes or None)
Part = Tuple[Dict[str, Any], List[Dict[str, Any]], Optional[int]]


def summarize(parts: Sequence[Part], name: str, source: str,
              fetched: Optional[int] = None) -> Dict[str, Any]:
    """Everything ``inspect`` reports about one model, from its parsed header(s).

    ``parts`` holds one entry per file of a split model (one entry otherwise). Tensors
    of a type this version cannot size are sized from their offsets when the part's
    size is known.
    """
    kv0 = parts[0][0]
    tensors: List[Dict[str, Any]] = []
    size: Optional[int] = 0
    for kv, part_tensors, part_size in parts:
        data_bytes = (part_size - kv["_data_start"]
                      if part_size is not None and "_data_start" in kv else None)
        gguf.size_from_offsets(part_tensors, data_bytes, kv.get("_alignment", 32))
        tensors.extend(part_tensors)
        size = None if (size is None or part_size is None) else size + part_size

    desc = gguf.describe_metadata(kv0, source, size)
    stream = gguf.streamed_bytes(tensors)
    comp = gguf.composition(tensors, desc.get("quant"))
    known = {n for n, _, _ in gguf.GGML_TYPES.values()}
    if comp and comp.get("expected_type") not in known and any(
            t.get("bytes_from_offsets") for t in tensors):
        # The label names a type this version cannot recognise in the tensor table, so
        # "0% on-label" would be a statement about this tool rather than the file.
        comp = dict(comp, on_label_pct=None, on_label_bytes=None)

    by_type: Dict[str, Dict[str, Any]] = {}
    for t in tensors:
        slot = by_type.setdefault(t["type"], {"tensors": 0, "bytes": 0})
        slot["tensors"] += 1
        slot["bytes"] += t["bytes"] or 0
    head_type = None
    for t in tensors:
        if t["name"].startswith("output.weight") or (
                stream and stream["tied_embedding"] and t["name"] == "token_embd.weight"):
            head_type = t["type"]
            break
    return {
        "name": name,
        "source": source,
        "label": desc.get("quant"),
        "architecture": desc.get("arch"),
        "file_bytes": size,
        "parts": len(parts),
        "n_tensors": len(tensors),
        "is_moe": bool(desc.get("is_moe")),
        "expected_type": comp.get("expected_type") if comp else None,
        "on_label_pct": comp.get("on_label_pct") if comp else None,
        "bits_per_weight": comp.get("bits_per_weight") if comp else None,
        "repeating_by_type_pct": comp.get("by_type_pct") if comp else None,
        "streamed_bytes": stream["streamed_bytes"] if stream else None,
        "total_tensor_bytes": stream["total_bytes"] if stream else None,
        "head_bytes": stream["head_bytes"] if stream else None,
        "head_share_pct": stream["head_share_pct"] if stream else None,
        "tied_embedding": stream["tied_embedding"] if stream else None,
        "head_type": head_type,
        "by_type": dict(sorted(by_type.items(), key=lambda kv: -kv[1]["bytes"])),
        "unsized_types": gguf.unsized_types(tensors),
        "types_sized_from_offsets": sorted({t["type"] for t in tensors
                                            if t.get("bytes_from_offsets")}),
        "header_bytes_fetched": fetched,
        "tensors": tensors,
    }


def wrap(text: str, width: int) -> List[str]:
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


def _field(out: List[str], label: str, text: str) -> None:
    """One labelled line of the report, wrapped under its own label."""
    lines = wrap(text, 70 - 15)
    out.append(f"  {label:<11}: {lines[0]}")
    out.extend(" " * 15 + line for line in lines[1:])


def render_text(r: Dict[str, Any], version: str, show_tensors: bool = False,
                peak_bw: Optional[float] = None) -> str:
    """The terminal report for one summary."""
    out: List[str] = []
    size = r["file_bytes"]
    out += [RULE, f"  {r['name']}", RULE]
    out.append(f"  label      : {r['label'] or 'unknown'}"
               + (f"   architecture: {r['architecture']}" if r["architecture"] else ""))
    # The headline. A label is a recipe, and the recipe substitutes silently, so the
    # first thing worth knowing about a file is how much of it is what it says.
    if r["on_label_pct"] is not None:
        out.append(f"  actually   : {r['on_label_pct']:.0f}% {r['expected_type']} "
                   f"in the repeating layers, {r['bits_per_weight']:.2f} bits per weight")
    parts = f", {r['parts']} parts" if r["parts"] > 1 else ""
    if size:
        out.append(f"  file       : {size / 1e6:.0f} MB, {r['n_tensors']} tensors{parts}")
    else:
        out.append(f"  file       : size not reported, {r['n_tensors']} tensors{parts}")
    if r["header_bytes_fetched"] is not None:
        out.append(f"  fetched    : {r['header_bytes_fetched'] / 1e6:.1f} MB of header; "
                   "the weights were not downloaded")
    if r["streamed_bytes"]:
        out.append(f"  per token  : {r['streamed_bytes'] / 1e6:.0f} MB read "
                   f"({100 * r['streamed_bytes'] / r['total_tensor_bytes']:.0f}% of the "
                   f"file)")
        out.append(f"  output head: {r['head_bytes'] / 1e6:.0f} MB, "
                   f"{r['head_share_pct']:.0f}% of what is read per token"
                   + ("   (tied to the embedding)" if r["tied_embedding"] else ""))
        if peak_bw:
            ceiling = peak_bw * 1e9 / r["streamed_bytes"]
            _field(out, "ceiling", f"{ceiling:.1f} tok/s at {peak_bw:g} GB/s, an upper "
                                   "bound on decode")
        if r["types_sized_from_offsets"]:
            _field(out, "note", f"{', '.join(r['types_sized_from_offsets'])} is newer than "
                                f"llama-roofline {version}. Its sizes come from the tensor "
                                "offsets, exact to within the alignment padding.")
        if r["is_moe"]:
            _field(out, "note", "mixture-of-experts. Only the routed experts are read per "
                                "token, so the per-token figure is an upper bound.")
    else:
        unknown = ", ".join(r["unsized_types"]) or "a type it cannot size"
        _field(out, "per token", f"not computable. This file uses {unknown}, which "
                                 f"llama-roofline {version} cannot size. Try `pip install "
                                 "-U llama-roofline`; if the type is still unknown, please "
                                 "report it: https://github.com/manunicholasjacob/"
                                 "llama-roofline/issues/new?template=gguf-type.md")
    out += ["", "  TENSOR TYPES", THIN, f"    {'type':<10}{'tensors':>9}{'MB':>10}{'share':>8}"]
    denom = size or r["total_tensor_bytes"]
    for name, slot in r["by_type"].items():
        if name in r["unsized_types"]:
            mb, share = f"{'?':>10}", f"{'?':>8}"
        else:
            mb = f"{slot['bytes'] / 1e6:>10.0f}"
            share = f"{100.0 * slot['bytes'] / denom:>7.0f}%" if denom else f"{'?':>8}"
        out.append(f"    {name:<10}{slot['tensors']:>9}{mb}{share}")
    out.append("")

    if r["on_label_pct"] is not None:
        share = r["on_label_pct"]
        expected = r["expected_type"]
        pct = r["repeating_by_type_pct"]
        ranked = [t for t in pct if t != expected]
        out += ["  WHAT THE LABEL MEANS HERE", THIN]
        if share >= 90:
            lead = (f"This file is what it says: {share:.0f}% of its repeating-layer "
                    f"bytes are {expected}.")
        elif share < 1:
            lead = (f"This file contains no {expected} tensors at all. Its repeating "
                    f"layers are "
                    + ", ".join(f"{t} {pct[t]:.0f}%" for t in ranked[:3]) + ".")
        else:
            lead = (f"Only {share:.0f}% of this file's repeating-layer bytes are "
                    f"{expected}. " + ("Most of it is " if share < 50 else "The rest is ")
                    + ", ".join(f"{t} {pct[t]:.0f}%" for t in ranked[:3]) + ".")
        text = (lead + (f" The output head is {r['head_type']}." if r["head_type"] else "")
                + f" It stores {r['bits_per_weight']:.2f} bits per weight."
                + " A GGUF label names a recipe rather than a type. llama-quantize"
                  " substitutes per tensor when a shape does not divide by the block"
                  " size. None of that is visible from the filename, and at small"
                  " model sizes the substitution can take over the file.")
        out.extend(f"  {line}" for line in wrap(text, 68))
        out.append("")
    if show_tensors:
        out += ["  EVERY TENSOR", THIN]
        for t in r["tensors"]:
            dims = "x".join(str(d) for d in t["dims"])
            mb = f"{t['bytes'] / 1e6:>8.1f} MB" if t["bytes"] is not None else f"{'?':>8} MB"
            out.append(f"    {t['name']:<40}{t['type']:<10}{dims:<20}{mb}")
        out.append("")
    return "\n".join(out)
