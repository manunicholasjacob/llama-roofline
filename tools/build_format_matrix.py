#!/usr/bin/env python3
"""Build the quantization-format matrix that ships with the advisor.

Every row is a measurement. Nothing here is modelled, averaged across cores, or typed
from a summary: throughput comes out of the raw campaign files, energy out of the Pi's
PMIC records, perplexity out of the perplexity run, and streamed bytes out of the
per-tensor type map that the study parsed for each artifact.

The source data lives in a private research repository, so this script is kept for
provenance rather than for the user to run. Point --raw at that repository's
paper16-format-tax/data directory to regenerate:

    python tools/build_format_matrix.py --raw <...>/paper16-format-tax/data
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, "src"))
from llama_roofline import gguf  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, os.pardir, "src", "llama_roofline", "data", "format_matrix.csv")

MIB = 1024 * 1024

# Bytes actually read per generated token: repeating layers plus the output head. The
# embedding table is a row lookup, not a stream, so file size overstates traffic, and it
# overstates it differently per format. These come from the study's per-tensor type-map
# parse of each artifact (paper16-format-tax, Table 1 and pi5_canonical_quant.log).
PUBLISHED_STREAMED_MIB = {
    "Q4_0": 330, "IQ4_XS": 330, "IQ4_NL": 332, "Q3_K_M": 334,
    "Q2_K": 317, "Q4_K_M": 374, "Q6_K": 477, "Q8_0": 501,
}

# Filled by --artifacts. Parsing the files gives streamed bytes at both scales, which the
# study only did at 0.5B, and gives the two columns that separate a fallback effect from a
# format effect: how much of a file is the type its name claims, and what it really stores
# per weight.
PARSED: dict = {}


def parse_artifacts(directories):
    """Read every GGUF under the given directories into PARSED, keyed by (scale, format)."""
    for scale, directory in directories:
        for path in sorted(glob.glob(os.path.join(directory, "*.gguf"))):
            stem = os.path.basename(path)
            match = re.search(r"-([A-Za-z0-9_]+?)(?:\.fp16src)?\.gguf$", stem)
            if not match:
                continue
            fmt = match.group(1).upper()
            tensors = gguf.read_tensors(path)
            stream = gguf.streamed_bytes(tensors)
            comp = gguf.composition(tensors, fmt)
            if not stream or not comp:
                print(f"  ! could not size {stem}", file=sys.stderr)
                continue
            PARSED[(scale, fmt)] = {
                "streamed_MiB": stream["streamed_bytes"] / MIB,
                "on_label_pct": comp["on_label_pct"],
                "bits_per_weight": comp["bits_per_weight"],
            }
    # The 0.5B streamed figures are a published result. If the parser disagrees with them
    # the parser is wrong, or the artifacts are not the ones the study used, and either
    # way the matrix must not be built.
    for fmt, published in PUBLISHED_STREAMED_MIB.items():
        got = PARSED.get((0.5, fmt))
        if not got:
            continue
        drift = abs(got["streamed_MiB"] - published) / published
        if drift > 0.01:
            raise SystemExit(
                f"parsed streamed size for {fmt} is {got['streamed_MiB']:.1f} MiB against "
                f"the published {published} MiB, a {100 * drift:.1f}% difference. Refusing "
                f"to build a matrix on artifacts that are not the measured ones.")
    print(f"parsed {len(PARSED)} artifacts, 0.5B streamed sizes agree with the study")


def streamed_mib(scale, fmt):
    hit = PARSED.get((scale, fmt))
    if hit:
        return hit["streamed_MiB"]
    return PUBLISHED_STREAMED_MIB.get(fmt) if scale == 0.5 else None

# One line per core, describing the machine the rows were taken on.
CORES = {
    "cortex-a76": {
        "label": "Arm Cortex-A76",
        "machine": "Raspberry Pi 5, 4 cores at 2.4 GHz, 2 GB LPDDR4X, 64-bit Linux",
        "isa": "aarch64",
    },
    "golden-cove": {
        "label": "Intel Golden Cove (P-core)",
        "machine": "i7-12700H, llama-bench pinned by affinity mask to the 6 P-cores",
        "isa": "x86_64",
    },
    "gracemont": {
        "label": "Intel Gracemont (E-core)",
        "machine": "i7-12700H, llama-bench pinned by affinity mask to the 8 E-cores",
        "isa": "x86_64",
    },
}

FIELDS = [
    "core", "core_label", "isa", "model", "params_b", "threads", "format",
    "tok_s", "streamed_MiB", "file_MiB", "streamed_GBs", "mJ_per_token", "perplexity",
    "on_label_pct", "bits_per_weight",
    "repack_coverage_pct", "normalization", "basis", "source",
]

# Runtime repack coverage: the share of repeating-layer bytes llama.cpp rewrites into an
# architecture-tuned kernel at load. It is the mechanism behind the ranking, and it is the
# one column a user can check on their own build (llama-bench -v prints it).
COVERAGE_X86 = {"Q4_0": 100, "IQ4_NL": 94, "Q3_K_M": 91, "Q2_K": 75,
                "IQ4_XS": 70, "Q4_K_M": 12, "Q6_K": 0, "Q8_0": 0}


def read_ppl(path):
    out = {}
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            out[row["model"].strip()] = float(row["ppl"])
    return out


def read_pi_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


_X86_RE = re.compile(
    r"^([PE])\s+(\S+)\s+(\d+)\s+([\d.]+)\s+(\d+)\s+([\d.]+)\s*$")
_FMT_RE = re.compile(r"qwen[\d.]+b-([A-Za-z0-9_]+?)(?:\.fp16src)?\.gguf")


def read_x86_summary(path):
    """Parse the fixed-width summary tables the x86 arms wrote."""
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            m = _X86_RE.match(line.rstrip("\n"))
            if not m:
                continue
            core = "golden-cove" if m.group(1) == "P" else "gracemont"
            fmt = _FMT_RE.search(m.group(2))
            if not fmt:
                continue
            rows.append({
                "core": core, "format": fmt.group(1).upper(),
                "threads": int(m.group(3)), "tok_s": float(m.group(4)),
                "file_bytes": int(m.group(5)),
            })
    return rows


def emit(rows, core, model, params_b, threads, fmt, tok_s, file_bytes,
         mJ=None, ppl=None, source="", normalization="streamed_bytes"):
    mib = streamed_mib(params_b, fmt)
    parsed = PARSED.get((params_b, fmt), {})
    if mib is None:
        normalization = "file_bytes"
    gbs = round(tok_s * mib * MIB / 1e9, 3) if mib else None
    rows.append({
        "core": core,
        "core_label": CORES[core]["label"],
        "isa": CORES[core]["isa"],
        "model": model,
        "params_b": params_b,
        "threads": threads,
        "format": fmt,
        "tok_s": round(tok_s, 3),
        "streamed_MiB": round(mib, 1) if mib else "",
        "file_MiB": round(file_bytes / MIB, 1) if file_bytes else "",
        "streamed_GBs": gbs if gbs else "",
        "mJ_per_token": round(mJ, 2) if mJ is not None else "",
        "perplexity": ppl if ppl is not None else "",
        "on_label_pct": (round(parsed["on_label_pct"], 1)
                         if parsed.get("on_label_pct") is not None else ""),
        "bits_per_weight": (round(parsed["bits_per_weight"], 2)
                            if parsed.get("bits_per_weight") is not None else ""),
        "repack_coverage_pct": (COVERAGE_X86.get(fmt, "")
                                if CORES[core]["isa"] == "x86_64" and params_b == 0.5
                                else ""),
        "normalization": normalization,
        "basis": "measured",
        "source": source,
    })


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="paper16-format-tax/data directory")
    ap.add_argument("--artifacts-05b", help="directory holding the 0.5B GGUFs")
    ap.add_argument("--artifacts-15b", help="directory holding the 1.5B GGUFs")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args(argv)

    directories = []
    if args.artifacts_05b:
        directories.append((0.5, args.artifacts_05b))
    if args.artifacts_15b:
        directories.append((1.5, args.artifacts_15b))
    if directories:
        parse_artifacts(directories)

    raw = args.raw
    need = ["pi5_canonical.jsonl", "ppl_results.txt", "summary_fp16src_P.txt",
            "summary_fp16src_E.txt", "pi5_15b.jsonl", "summary_15b_P.txt",
            "summary_15b_E.txt"]
    missing = [n for n in need if not os.path.isfile(os.path.join(raw, n))]
    if missing:
        print(f"missing raw inputs in {raw}: {', '.join(missing)}", file=sys.stderr)
        return 2

    ppl = read_ppl(os.path.join(raw, "ppl_results.txt"))
    rows = []

    # --- Cortex-A76, 0.5B: the only arm with energy, because it is the only board with
    # a PMIC we can read. Power comes from a separate identical run, never the timed one.
    for r in read_pi_jsonl(os.path.join(raw, "pi5_canonical.jsonl")):
        emit(rows, "cortex-a76", "Qwen2.5-0.5B-Instruct", 0.5, r["threads"], r["tag"],
             r["tok_s"], r["bytes"], mJ=r.get("mJ_per_tok"), ppl=ppl.get(r["tag"]),
             source="paper16-format-tax/data/pi5_canonical.jsonl")

    # --- x86, 0.5B, one microarchitecture at a time on one memory system.
    for name in ("summary_fp16src_P.txt", "summary_fp16src_E.txt"):
        for r in read_x86_summary(os.path.join(raw, name)):
            emit(rows, r["core"], "Qwen2.5-0.5B-Instruct", 0.5, r["threads"], r["format"],
                 r["tok_s"], r["file_bytes"], ppl=ppl.get(r["format"]),
                 source=f"paper16-format-tax/data/{name}")

    # --- 1.5B, the out-of-sample arm. Tensor maps were not parsed at this scale, so
    # these rows normalize by file size and carry no streamed-byte figure. Keeping the
    # distinction in the data stops the advisor comparing the two silently.
    for r in read_pi_jsonl(os.path.join(raw, "pi5_15b.jsonl")):
        emit(rows, "cortex-a76", "Qwen2.5-1.5B-Instruct", 1.5, r["threads"], r["tag"],
             r["tok_s"], r["bytes"], mJ=r.get("mJ_per_tok"),
             source="paper16-format-tax/data/pi5_15b.jsonl"
                    + ("; streamed bytes parsed from the artifacts" if PARSED else ""))
    for name in ("summary_15b_P.txt", "summary_15b_E.txt"):
        for r in read_x86_summary(os.path.join(raw, name)):
            emit(rows, r["core"], "Qwen2.5-1.5B-Instruct", 1.5, r["threads"], r["format"],
                 r["tok_s"], r["file_bytes"],
                 source=f"paper16-format-tax/data/{name}"
                        + ("; streamed bytes parsed from the artifacts" if PARSED else ""))

    rows.sort(key=lambda r: (r["params_b"], r["core"], r["threads"], -r["tok_s"]))
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
