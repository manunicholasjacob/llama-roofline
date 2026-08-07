"""Command-line interface.

    llama-roofline run --models ~/models/*.gguf
    llama-roofline membw
    llama-roofline report results.json --markdown report.md
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional

from . import SCHEMA_VERSION, __version__, bench, gguf, membw, report, roofline, sysinfo

SHARD_RE = re.compile(r"^(?P<stem>.+)-(?P<idx>\d{5})-of-(?P<total>\d{5})\.gguf$", re.IGNORECASE)


def _eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


# --------------------------------------------------------------------------- models

def expand_model_args(patterns: List[str]) -> List[str]:
    """Turn files, directories and globs into a de-duplicated list of GGUF paths.

    Multi-part models are represented by their first shard only: llama.cpp loads the
    remaining shards automatically, and benchmarking each part separately is meaningless.
    """
    found: List[str] = []
    for pat in patterns:
        pat = os.path.expanduser(pat)
        if os.path.isdir(pat):
            found.extend(sorted(glob.glob(os.path.join(pat, "*.gguf"))))
        elif any(ch in pat for ch in "*?["):
            found.extend(sorted(glob.glob(pat)))
        else:
            found.append(pat)

    out: List[str] = []
    seen = set()
    for p in found:
        ap = os.path.abspath(p)
        m = SHARD_RE.match(os.path.basename(ap))
        if m and m.group("idx") != "00001":
            continue  # non-first shard: skip, the first one stands for the whole model
        if ap not in seen:
            seen.add(ap)
            out.append(ap)
    return out


def total_model_bytes(path: str) -> Optional[int]:
    """File size, summing every shard of a multi-part model."""
    m = SHARD_RE.match(os.path.basename(path))
    if not m:
        try:
            return os.path.getsize(path)
        except OSError:
            return None
    total = 0
    stem, n = m.group("stem"), int(m.group("total"))
    for i in range(1, n + 1):
        shard = os.path.join(os.path.dirname(path), f"{stem}-{i:05d}-of-{n:05d}.gguf")
        try:
            total += os.path.getsize(shard)
        except OSError:
            return None
    return total


# --------------------------------------------------------------------------- run

def cmd_run(args: argparse.Namespace) -> int:
    quiet = args.quiet

    def say(*a):
        if not quiet:
            print(*a, flush=True)

    paths = expand_model_args(args.models)
    missing = [p for p in paths if not os.path.isfile(p)]
    for p in missing:
        _eprint(f"error: model not found: {p}")
    paths = [p for p in paths if os.path.isfile(p)]
    if not paths:
        _eprint("error: no GGUF models to benchmark. Pass --models with files, a directory, "
                "or a glob.")
        return 2

    try:
        binary = bench.find_llama_bench(args.llama_bench)
    except bench.BenchNotFound as exc:
        _eprint(f"error: {exc}")
        return 3
    say(f"llama-bench : {binary}")

    system = sysinfo.collect()
    threads = ([int(t) for t in args.threads.split(",") if t.strip()]
               if args.threads else sysinfo.default_thread_sweep())
    threads = sorted(set(t for t in threads if t >= 1))
    say(f"machine     : {system.get('cpu')} ({system.get('logical_cores')} logical cores)")
    say(f"threads     : {threads}")
    say(f"models      : {len(paths)}")

    # ---- ceiling
    mem: Dict[str, Any] = {}
    peak: Optional[float] = args.peak_bw
    if args.peak_bw:
        mem = {"peak_read_GBs": args.peak_bw, "source": "user-supplied (--peak-bw)"}
        say(f"\nceiling     : {args.peak_bw:.1f} GB/s (supplied)")
    elif args.skip_membw:
        say("\nceiling     : skipped (--skip-membw); utilisation will be unavailable")
    else:
        say("\n[1/2] measuring memory bandwidth ceiling ...")
        try:
            mem = membw.measure(
                working_set_mb=args.membw_mb, reps=args.membw_reps,
                ram_bytes=system.get("ram_bytes"),
                progress=(None if quiet else
                          lambda nt, row: print(
                              f"      {nt:>3} threads: read {max(row['sum'], row['max'], row['dot']):6.1f} GB/s"
                              f"   copy {row['copy']:6.1f} GB/s", flush=True)),
            )
            peak = mem["peak_read_GBs"]
            say(f"      -> ceiling {peak:.1f} GB/s "
                f"({mem['best_kernel']} kernel, {mem['best_threads']} threads)")
            if mem.get("warning"):
                _eprint(f"warning: {mem['warning']}")
        except membw.NumpyMissing as exc:
            _eprint(f"warning: {exc}")

    # ---- benchmarks
    say(f"\n[2/2] benchmarking {len(paths)} model(s) x {len(threads)} thread setting(s) ...")
    models: List[Dict[str, Any]] = []
    build_info: Dict[str, Any] = {}
    t_start = time.time()

    for path in paths:
        desc = gguf.describe(path)
        file_bytes = total_model_bytes(path) or desc.get("file_bytes")
        entry: Dict[str, Any] = {
            "name": os.path.splitext(os.path.basename(path))[0],
            "path": path,
            "file_bytes": file_bytes,
            "quant": desc.get("quant"),
            "arch": desc.get("arch"),
            "n_params": desc.get("n_params"),
            "is_moe": desc.get("is_moe", False),
            "n_expert": desc.get("n_expert"),
            "gguf_metadata_ok": desc.get("metadata_ok"),
            "runs": [],
        }
        say(f"\n  {entry['name']}  ({(file_bytes or 0) / 1e6:.0f} MB"
            + (f", {desc['quant']}" if desc.get("quant") else "")
            + (", MoE" if entry["is_moe"] else "") + ")")

        for t in threads:
            try:
                r = bench.bench_model(
                    binary, path, threads=t, n_prompt=args.n_prompt, n_gen=args.n_gen,
                    reps=args.reps, depth=args.depth, gpu_layers=args.gpu_layers,
                    extra_args=args.bench_arg, timeout=args.timeout,
                )
            except bench.BenchError as exc:
                _eprint(f"    t={t}: FAILED -- {exc}")
                entry.setdefault("error", str(exc))
                continue
            entry["runs"].append(r)
            if not build_info and r.get("build_number"):
                build_info = {k: r.get(k) for k in
                              ("build_number", "build_commit", "backends", "gpu_info", "cpu_info")}
            # llama-bench reports the resident tensor size; prefer it over the file size.
            if r.get("model_size") and not entry.get("model_size"):
                entry["model_size"] = r["model_size"]
            if r.get("model_n_params") and not entry.get("n_params"):
                entry["n_params"] = r["model_n_params"]
            if r.get("model_type") and not entry.get("model_type"):
                entry["model_type"] = r["model_type"]
            for k in ("n_gpu_layers", "gpu_info"):
                if k in r:
                    entry[k] = r[k]

            dec, pre = r.get("decode_ts"), r.get("prefill_ts")
            bw = (dec * (entry.get("model_size") or file_bytes or 0) / 1e9) if dec else None
            line = (f"    t={t:<3} decode {dec:7.2f} tok/s   prefill {pre or 0:8.1f} tok/s")
            if bw:
                line += f"   {bw:6.2f} GB/s"
                if peak:
                    line += f"  ({100 * bw / peak:.0f}% of ceiling)"
            say(line)

        entry["bytes_per_token"] = entry.get("model_size") or file_bytes
        entry["bytes_source"] = "llama-bench model_size" if entry.get("model_size") else "file size"
        models.append(entry)

    analysis = roofline.analyze(models, peak)
    if mem.get("warning"):
        analysis["warnings"].insert(0, mem["warning"])
    results: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "tool_version": __version__,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_s": round(time.time() - t_start, 1),
        "system": system,
        "llama_cpp": {**bench.bench_version(binary), **build_info},
        "settings": {
            "threads": threads, "n_prompt": args.n_prompt, "n_gen": args.n_gen,
            "reps": args.reps, "depth": args.depth, "gpu_layers": args.gpu_layers,
            "extra_bench_args": args.bench_arg,
        },
        "membw": mem,
        "analysis": analysis,
    }

    outdir = args.out
    os.makedirs(outdir, exist_ok=True)
    json_path = args.json or os.path.join(outdir, "roofline.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    text = report.render(results)
    print("\n" + text)
    with open(os.path.join(outdir, "report.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    md_path = args.markdown or os.path.join(outdir, "report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(report.render_markdown(results) + "\n")

    written = [json_path, os.path.join(outdir, "report.txt"), md_path]
    if not args.no_plot:
        try:
            from . import plot
            png = plot.roofline_figure(results, os.path.join(outdir, "roofline.png"))
            written.append(png)
        except Exception as exc:  # plotting must never sink a good measurement
            _eprint(f"note: plot skipped ({exc})")

    print("  wrote:")
    for w in written:
        print(f"    {w}")

    # Files were still written (the failures are worth keeping), but a run where nothing
    # could be measured is not a success, and a script should be able to tell.
    if analysis.get("n_models_ok", 0) == 0:
        _eprint("\nerror: no model produced a decode measurement. "
                "Check that the GGUF files load with your llama-bench build.")
        return 1
    return 0


# --------------------------------------------------------------------------- membw

def cmd_membw(args: argparse.Namespace) -> int:
    system = sysinfo.collect()
    print(f"machine: {system.get('cpu')} ({system.get('logical_cores')} logical cores, "
          f"{(system.get('ram_bytes') or 0) / 1e9:.1f} GB RAM)")
    try:
        mem = membw.measure(working_set_mb=args.membw_mb, reps=args.membw_reps,
                            ram_bytes=system.get("ram_bytes"),
                            progress=lambda nt, row: print(
                                f"  {nt:>3} threads   sum {row['sum']:6.1f}   max {row['max']:6.1f}"
                                f"   dot {row['dot']:6.1f}   copy {row['copy']:6.1f}  GB/s",
                                flush=True))
    except membw.NumpyMissing as exc:
        _eprint(f"error: {exc}")
        return 4
    print(f"\nsustained read ceiling: {mem['peak_read_GBs']:.2f} GB/s "
          f"({mem['best_kernel']} kernel @ {mem['best_threads']} threads, "
          f"{mem['working_set_mb']} MB working set)")
    if mem.get("warning"):
        _eprint(f"\nwarning: {mem['warning']}")
    print("this is a measured lower bound on true peak; pass it to `run` with "
          f"--peak-bw {mem['peak_read_GBs']:.2f}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"system": system, "membw": mem}, f, indent=2)
        print(f"wrote {args.json}")
    return 0


# --------------------------------------------------------------------------- report

def cmd_report(args: argparse.Namespace) -> int:
    try:
        with open(args.results, "r", encoding="utf-8") as f:
            results = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        _eprint(f"error: could not read {args.results}: {exc}")
        return 2
    if results.get("schema_version") != SCHEMA_VERSION:
        _eprint(f"warning: results were written by schema v{results.get('schema_version')}, "
                f"this build expects v{SCHEMA_VERSION}")

    if args.reanalyze:
        old = results.get("analysis") or {}
        if not old.get("models"):
            _eprint("error: --reanalyze needs per-model runs, which these results do not have")
            return 2
        results["analysis"] = roofline.analyze(old["models"], old.get("peak_read_GBs"))
        results["tool_version"] = __version__
        results["reanalyzed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with open(args.results, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"reanalyzed with llama-roofline {__version__}, rewrote {args.results}")
    if args.markdown:
        with open(args.markdown, "w", encoding="utf-8") as f:
            f.write(report.render_markdown(results) + "\n")
        print(f"wrote {args.markdown}")
    if args.plot:
        from . import plot
        print(f"wrote {plot.roofline_figure(results, args.plot)}")
    if not args.markdown and not args.plot:
        print(report.render(results))
    return 0


# --------------------------------------------------------------------------- parser

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="llama-roofline",
        description="Is your llama.cpp decode memory-bandwidth-bound? Measure it.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  llama-roofline run --models ~/models/*.gguf\n"
            "  llama-roofline run --models a.gguf b.gguf --threads 1,2,4,8 --reps 5\n"
            "  llama-roofline membw\n"
            "  llama-roofline report out/roofline.json --markdown report.md\n"
        ),
    )
    p.add_argument("--version", action="version", version=f"llama-roofline {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="benchmark models and produce the roofline report")
    r.add_argument("--models", "-m", nargs="+", required=True,
                   metavar="GGUF", help="GGUF files, directories, or globs")
    r.add_argument("--llama-bench", metavar="PATH",
                   help="path to the llama-bench binary (default: $LLAMA_BENCH, PATH, "
                        "then common build locations)")
    r.add_argument("--threads", "-t", metavar="LIST",
                   help="comma-separated thread counts (default: 1, half, physical, logical)")
    r.add_argument("--n-prompt", type=int, default=128, metavar="N",
                   help="prompt tokens for the prefill test (default: 128; must be > 0)")
    r.add_argument("--n-gen", type=int, default=128, metavar="N",
                   help="tokens to generate for the decode test (default: 128)")
    r.add_argument("--reps", type=int, default=3, metavar="N",
                   help="llama-bench repetitions per test (default: 3)")
    r.add_argument("--depth", type=int, default=None, metavar="N",
                   help="generate with N tokens already in the KV cache (llama-bench -d). "
                        "Use this to see how far your decode falls off the weights-only "
                        "roofline at a realistic context length. Note llama-bench has no "
                        "context-size flag; it sizes the context from these values itself.")
    r.add_argument("--gpu-layers", type=int, default=0, metavar="N",
                   help="layers to offload (default: 0 -- a CPU/system-RAM roofline. Use -1 "
                        "for llama.cpp's automatic offload, but note the ceiling measured here "
                        "is system RAM, not VRAM)")
    r.add_argument("--bench-arg", action="append", default=None, metavar="ARG",
                   help="extra argument passed through to llama-bench (repeatable)")
    r.add_argument("--timeout", type=float, default=3600.0, metavar="SEC",
                   help="per-invocation timeout (default: 3600)")
    r.add_argument("--peak-bw", type=float, default=None, metavar="GBS",
                   help="skip the microbenchmark and use this ceiling in GB/s")
    r.add_argument("--skip-membw", action="store_true",
                   help="do not measure the ceiling (utilisation will be unavailable)")
    r.add_argument("--membw-mb", type=int, default=None, metavar="MB",
                   help="microbenchmark working set (default: 512 MB, capped at RAM/8)")
    r.add_argument("--membw-reps", type=int, default=5, metavar="N",
                   help="microbenchmark repetitions (default: 5)")
    r.add_argument("--out", "-o", default="llama-roofline-out", metavar="DIR",
                   help="output directory (default: ./llama-roofline-out)")
    r.add_argument("--json", metavar="PATH", help="results JSON path override")
    r.add_argument("--markdown", metavar="PATH", help="Markdown report path override")
    r.add_argument("--no-plot", action="store_true", help="skip the PNG figure")
    r.add_argument("--quiet", "-q", action="store_true", help="only print the final report")
    r.set_defaults(func=cmd_run)

    m = sub.add_parser("membw", help="measure this machine's memory-bandwidth ceiling only")
    m.add_argument("--membw-mb", type=int, default=None, metavar="MB")
    m.add_argument("--membw-reps", type=int, default=5, metavar="N")
    m.add_argument("--json", metavar="PATH", help="write the measurement to this file")
    m.set_defaults(func=cmd_membw)

    rep = sub.add_parser("report", help="re-render a report from a saved results JSON")
    rep.add_argument("results", metavar="RESULTS_JSON")
    rep.add_argument("--markdown", metavar="PATH")
    rep.add_argument("--plot", metavar="PNG")
    rep.add_argument("--reanalyze", action="store_true",
                     help="re-run the analysis over the stored per-model measurements with "
                          "this version of the tool, and rewrite the results file. Use after "
                          "upgrading; the raw llama-bench numbers are never changed.")
    rep.set_defaults(func=cmd_report)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        _eprint("\ninterrupted")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
