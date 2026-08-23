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
import struct
import sys
import time
from typing import Any, Dict, List, Optional

from . import (SCHEMA_VERSION, __version__, advisor, bench, gguf, membw, report,
               roofline, silicon, sysinfo)

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


# --------------------------------------------------------------------------- entries

def new_model_entry(path: str) -> Dict[str, Any]:
    """Describe one GGUF before any benchmarking, from its own header."""
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
    # Bytes actually read per token, when the tensor table can be parsed. Multi-part
    # models are skipped: only the first shard is opened, so the total would be wrong.
    if not SHARD_RE.match(os.path.basename(path)):
        try:
            tensors = gguf.read_tensors(path)
            stream = gguf.streamed_bytes(tensors)
        except (gguf.GGUFError, OSError, struct.error):
            stream = None
        if stream:
            entry["streamed"] = stream
            entry["type_map"] = gguf.type_map(tensors)
    return entry


def absorb_run(entry: Dict[str, Any], r: Dict[str, Any]) -> None:
    """Fold one llama-bench result into the model's entry."""
    entry["runs"].append(r)
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
        entry = new_model_entry(path)
        desc_quant = entry.get("quant")
        say(f"\n  {entry['name']}  ({(entry.get('file_bytes') or 0) / 1e6:.0f} MB"
            + (f", {desc_quant}" if desc_quant else "")
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
            absorb_run(entry, r)
            if not build_info and r.get("build_number"):
                build_info = {k: r.get(k) for k in
                              ("build_number", "build_commit", "backends", "gpu_info", "cpu_info")}

            dec, pre = r.get("decode_ts"), r.get("prefill_ts")
            bw = (dec * (entry.get("model_size") or entry.get("file_bytes") or 0) / 1e9) if dec else None
            line = (f"    t={t:<3} decode {dec:7.2f} tok/s   prefill {pre or 0:8.1f} tok/s")
            if bw:
                line += f"   {bw:6.2f} GB/s"
                if peak:
                    line += f"  ({100 * bw / peak:.0f}% of ceiling)"
            say(line)

        entry["bytes_per_token"] = entry.get("model_size") or entry.get("file_bytes")
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


# --------------------------------------------------------------------------- discovery

# Where GGUF files usually end up. Searched in order, two levels deep, first hit wins.
MODEL_DIRS = [
    "models",
    "~/models",
    "~/.cache/llama.cpp",
    "~/.cache/lm-studio/models",
    "~/.lmstudio/models",
    "~/Documents/models",
    "/usr/share/llama.cpp/models",
    "C:/llmpc/models",
]
DISCOVERY_DEPTH = 2


def find_models(extra_dirs: Optional[List[str]] = None, limit: int = 200,
                defaults: bool = True) -> List[str]:
    """Look for GGUF files in the usual places, so `diagnose` needs no arguments.

    Shallow on purpose. Walking a home directory to its leaves is slow and finds things
    the user did not mean, and $LLAMA_MODELS covers anywhere unusual. Pass
    ``defaults=False`` to search only the directories given.
    """
    roots: List[str] = list(extra_dirs or [])
    if defaults:
        env = os.environ.get("LLAMA_MODELS")
        if env:
            roots.extend(env.split(os.pathsep))
        roots.append(os.getcwd())
        roots.extend(MODEL_DIRS)

    found: List[str] = []
    seen = set()
    for root in roots:
        root = os.path.abspath(os.path.expanduser(root))
        if not os.path.isdir(root) or root in seen:
            continue
        seen.add(root)
        base_depth = root.rstrip(os.sep).count(os.sep)
        for dirpath, dirnames, filenames in os.walk(root):
            if dirpath.count(os.sep) - base_depth >= DISCOVERY_DEPTH:
                dirnames[:] = []
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            for fn in sorted(filenames):
                if fn.lower().endswith(".gguf"):
                    m = SHARD_RE.match(fn)
                    if m and m.group("idx") != "00001":
                        continue
                    found.append(os.path.join(dirpath, fn))
            if len(found) >= limit:
                return found
    return found


def spread_selection(paths: List[str], count: int) -> List[str]:
    """Pick models spanning the size range, so the roofline slope can be fitted.

    Three files of nearly the same size say nothing about how throughput scales with
    bytes. Smallest, middle and largest do.
    """
    if len(paths) <= count:
        return paths
    sized = sorted(((total_model_bytes(p) or 0, p) for p in paths))
    if count == 1:
        return [sized[0][1]]
    idx = [round(i * (len(sized) - 1) / (count - 1)) for i in range(count)]
    out, seen = [], set()
    for i in idx:
        if i not in seen:
            seen.add(i)
            out.append(sized[i][1])
    return out


# --------------------------------------------------------------------------- diagnose

DIAGNOSE_MAX_MODELS = 3


def cmd_diagnose(args: argparse.Namespace) -> int:
    """Measure this machine and say, in words, what is limiting token generation."""
    quiet = args.quiet

    def say(*a):
        if not quiet:
            print(*a, flush=True)

    if args.models:
        paths = [p for p in expand_model_args(args.models) if os.path.isfile(p)]
        if not paths:
            _eprint("error: none of those paths is a GGUF file.")
            return 2
    else:
        say("looking for GGUF models ...")
        candidates = find_models()
        if not candidates:
            _eprint(
                "error: no GGUF models found.\n"
                "  Pass one:            llama-roofline diagnose path/to/model.gguf\n"
                "  Or point at a set:   llama-roofline diagnose ~/models\n"
                "  Or set LLAMA_MODELS to the directory that holds them.\n"
                "  Searched: " + ", ".join(MODEL_DIRS))
            return 2
        paths = spread_selection(candidates, DIAGNOSE_MAX_MODELS)
        say(f"found {len(candidates)} model(s), using {len(paths)}: "
            + ", ".join(os.path.basename(p) for p in paths))

    try:
        binary = bench.find_llama_bench(args.llama_bench)
    except bench.BenchNotFound as exc:
        _eprint(f"error: {exc}")
        return 3

    system = sysinfo.collect()
    if args.threads:
        threads = sorted({int(t) for t in args.threads.split(",") if t.strip()})
    else:
        threads = sysinfo.default_thread_sweep()
        if args.quick:
            threads = threads[:2] if len(threads) > 2 else threads
    reps = args.reps if args.reps else (1 if args.quick else 3)

    say(f"machine     : {system.get('cpu')} ({system.get('logical_cores')} logical cores)")
    say(f"llama-bench : {binary}")
    say(f"plan        : {len(paths)} model(s) x {len(threads)} thread setting(s), "
        f"{reps} repetition(s)")

    mem: Dict[str, Any] = {}
    peak: Optional[float] = args.peak_bw
    if args.peak_bw:
        mem = {"peak_read_GBs": args.peak_bw, "source": "user-supplied (--peak-bw)"}
    elif args.skip_membw:
        say("\nceiling     : skipped, so there will be no verdict")
    else:
        say("\n[1/2] measuring the memory ceiling ...")
        try:
            mem = membw.measure(reps=(3 if args.quick else 5),
                                ram_bytes=system.get("ram_bytes"))
            peak = mem["peak_read_GBs"]
            say(f"      -> {peak:.1f} GB/s ({mem['best_kernel']} kernel, "
                f"{mem['best_threads']} threads)")
            if mem.get("warning"):
                _eprint(f"warning: {mem['warning']}")
        except membw.NumpyMissing as exc:
            _eprint(f"warning: {exc}")

    say("\n[2/2] benchmarking ...")
    models: List[Dict[str, Any]] = []
    build_info: Dict[str, Any] = {}
    t_start = time.time()
    for path in paths:
        entry = new_model_entry(path)
        say(f"\n  {entry['name']}  ({(entry.get('file_bytes') or 0) / 1e6:.0f} MB"
            + (f", {entry['quant']}" if entry.get("quant") else "")
            + (", MoE" if entry["is_moe"] else "") + ")")
        for t in threads:
            try:
                r = bench.bench_model(binary, path, threads=t, n_prompt=128,
                                      n_gen=(64 if args.quick else 128), reps=reps,
                                      gpu_layers=args.gpu_layers, timeout=args.timeout)
            except bench.BenchError as exc:
                _eprint(f"    t={t}: FAILED -- {exc}")
                entry.setdefault("error", str(exc))
                continue
            absorb_run(entry, r)
            if not build_info and r.get("build_number"):
                build_info = {k: r.get(k) for k in ("build_number", "build_commit",
                                                    "backends", "gpu_info", "cpu_info")}
            say(f"    t={t:<3} decode {r.get('decode_ts') or 0:7.2f} tok/s   "
                f"prefill {r.get('prefill_ts') or 0:8.1f} tok/s")

        # Decode reads the repeating layers and the output head every token and looks up
        # one row of the embedding table, so streamed bytes are the honest denominator
        # when the tensor map parses. `run` keeps the resident-size convention.
        stream = entry.get("streamed")
        if stream and stream.get("streamed_bytes") and not args.file_bytes:
            entry["bytes_per_token"] = stream["streamed_bytes"]
            entry["bytes_source"] = "streamed tensors (GGUF tensor map)"
        else:
            entry["bytes_per_token"] = entry.get("model_size") or entry.get("file_bytes")
            entry["bytes_source"] = ("llama-bench model_size" if entry.get("model_size")
                                     else "file size")
        models.append(entry)

    analysis = roofline.analyze(models, peak)
    if mem.get("warning"):
        analysis["warnings"].insert(0, mem["warning"])
    results: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "tool_version": __version__,
        "command": "diagnose",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_s": round(time.time() - t_start, 1),
        "system": system,
        "llama_cpp": {**bench.bench_version(binary), **build_info},
        "settings": {"threads": threads, "n_prompt": 128,
                     "n_gen": 64 if args.quick else 128, "reps": reps,
                     "gpu_layers": args.gpu_layers, "quick": bool(args.quick)},
        "membw": mem,
        "analysis": analysis,
    }

    print("\n" + report.render_diagnosis(results))

    outdir = args.out
    os.makedirs(outdir, exist_ok=True)
    written = []
    json_path = os.path.join(outdir, "diagnosis.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    written.append(json_path)
    md_path = args.markdown or os.path.join(outdir, "diagnosis.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(report.render_diagnosis_markdown(results) + "\n")
    written.append(md_path)
    print("  wrote:")
    for w in written:
        print(f"    {w}")
    print(f"  {os.path.basename(md_path)} is the shareable one: paste it into an issue "
          f"or a forum thread as is.")

    if analysis.get("n_models_ok", 0) == 0:
        _eprint("\nerror: no model produced a decode measurement. Check that these GGUF "
                "files load with your llama-bench build.")
        return 1
    return 0


# --------------------------------------------------------------------------- inspect

def cmd_inspect(args: argparse.Namespace) -> int:
    """Show what is actually inside a GGUF file, since the filename does not say."""
    rc = 0
    for path in expand_model_args(args.models):
        if not os.path.isfile(path):
            _eprint(f"error: not a file: {path}")
            rc = 2
            continue
        try:
            tensors = gguf.read_tensors(path)
        except (gguf.GGUFError, OSError, struct.error) as exc:
            _eprint(f"error: could not read the tensor table of {path}: {exc}")
            rc = 2
            continue
        desc = gguf.describe(path)
        stream = gguf.streamed_bytes(tensors)
        size = os.path.getsize(path)

        print(report.RULE)
        print(f"  {os.path.basename(path)}")
        print(report.RULE)
        print(f"  label      : {desc.get('quant') or 'unknown'}"
              + (f"   architecture: {desc['arch']}" if desc.get("arch") else ""))
        print(f"  file       : {size / 1e6:.0f} MB, {len(tensors)} tensors")
        if stream:
            print(f"  per token  : {stream['streamed_bytes'] / 1e6:.0f} MB read "
                  f"({100 * stream['streamed_bytes'] / stream['total_bytes']:.0f}% of the "
                  f"file)")
            print(f"  output head: {stream['head_bytes'] / 1e6:.0f} MB, "
                  f"{stream['head_share_pct']:.0f}% of what is read per token"
                  + ("   (tied to the embedding)" if stream["tied_embedding"] else ""))
        else:
            print("  per token  : not computable, at least one tensor uses a ggml type "
                  "this build does not know")
        print()
        print("  TENSOR TYPES")
        print(report.THIN)
        by_type: Dict[str, Dict[str, Any]] = {}
        for t in tensors:
            slot = by_type.setdefault(t["type"], {"n": 0, "bytes": 0})
            slot["n"] += 1
            slot["bytes"] += t["bytes"] or 0
        rows = sorted(by_type.items(), key=lambda kv: -kv[1]["bytes"])
        print(f"    {'type':<10}{'tensors':>9}{'MB':>10}{'share':>8}")
        for name, slot in rows:
            share = 100.0 * slot["bytes"] / size if size else 0
            print(f"    {name:<10}{slot['n']:>9}{slot['bytes'] / 1e6:>10.0f}{share:>7.0f}%")
        print()

        # The type a label implies: Q4_K_M means Q4_K tensors, Q4_0 means Q4_0, IQ4_XS
        # means IQ4_XS. The repeating layers are where that matters, because the output
        # head is chosen by the recipe and is often a different type on purpose.
        label = (desc.get("quant") or "").upper()
        expected = label if label in by_type else label.rsplit("_", 1)[0]
        repeating = [t for t in tensors if t["name"].startswith("blk.")
                     and t["type"] not in ("F32", "F16", "BF16")]
        rep_bytes = sum(t["bytes"] or 0 for t in repeating)
        on_label = sum(t["bytes"] or 0 for t in repeating if t["type"] == expected)
        share = 100.0 * on_label / rep_bytes if rep_bytes else None
        others: Dict[str, int] = {}
        for t in repeating:
            if t["type"] != expected:
                others[t["type"]] = others.get(t["type"], 0) + (t["bytes"] or 0)
        ranked = sorted(others, key=lambda k: -others[k])

        if label and share is not None:
            print("  WHAT THE LABEL MEANS HERE")
            print(report.THIN)
            head_type = None
            for t in tensors:
                if t["name"].startswith("output.weight") or (
                        stream and stream["tied_embedding"]
                        and t["name"] == "token_embd.weight"):
                    head_type = t["type"]
                    break
            text = (f"A GGUF label names a recipe, not a type. In this file "
                    f"{share:.0f}% of the repeating-layer bytes are {expected}"
                    + (f", and the rest are {', '.join(ranked[:4])}" if ranked else "")
                    + (f". The output head is {head_type}" if head_type else "")
                    + ". llama-quantize picks a type per tensor from the recipe, from "
                      "whether the shape divides evenly, and from what the file was "
                      "converted from. Two files with this name and this nominal bit "
                      "width can hold different maps and decode at very different "
                      "speeds, so a benchmark that does not pin the artifact is "
                      "measuring something unnamed.")
            for line in report._wrap(text, 68):
                print(f"  {line}")
            print()
        if args.tensors:
            print("  EVERY TENSOR")
            print(report.THIN)
            for t in tensors:
                dims = "x".join(str(d) for d in t["dims"])
                print(f"    {t['name']:<40}{t['type']:<10}{dims:<20}"
                      f"{(t['bytes'] or 0) / 1e6:>8.1f} MB")
            print()
    return rc


# --------------------------------------------------------------------------- advise

def cmd_advise(args: argparse.Namespace) -> int:
    """Rank quantization formats for the core this machine actually has."""
    try:
        matrix = advisor.load_matrix()
    except advisor.MatrixMissing as exc:
        _eprint(f"error: {exc}")
        return 4

    if args.list_cores:
        print("cores with measurements in this build:")
        for c in advisor.cores(matrix):
            rows = [r for r in matrix if r["core"] == c]
            scales = sorted({r["params_b"] for r in rows})
            threads = sorted({r["threads"] for r in rows})
            print(f"  {c:<14}{rows[0]['core_label']:<30}"
                  f"scales {scales}, threads {threads}, {len(rows)} rows")
        return 0

    if args.measure:
        return _advise_by_measuring(args)

    detection = silicon.detect()
    if args.core:
        known = advisor.cores(matrix)
        if args.core not in known:
            _eprint(f"error: no measurements for core '{args.core}'. "
                    f"Known: {', '.join(known)}")
            return 2
        core_infos = [{"core": args.core, "confidence": "forced",
                       "why": "nothing was detected or checked against the local CPU"}]
    else:
        core_infos = detection["cores"]

    advices = [(ci, advisor.advise_core(matrix, ci["core"], params_b=args.scale,
                                        threads=args.threads))
               for ci in core_infos]
    advices = [(ci, a) for ci, a in advices if not a.get("error")]

    text = advisor.render(detection, advices, matrix, __version__)
    print(text)
    if args.markdown:
        with open(args.markdown, "w", encoding="utf-8") as f:
            f.write(advisor.render_markdown(detection, advices, matrix, __version__) + "\n")
        print(f"  wrote {args.markdown}")
    if args.json:
        payload = {"tool_version": __version__, "detection": detection,
                   "advice": [{"core_info": ci, **{k: v for k, v in a.items()
                                                   if k != "pareto"}}
                              for ci, a in advices]}
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"  wrote {args.json}")
    return 0


def _advise_by_measuring(args: argparse.Namespace) -> int:
    """Build the ranking from the user's own files, which is the only way to be sure."""
    if not args.models:
        _eprint("error: --measure needs models. Point it at several quantizations of the "
                "same model:\n"
                "  llama-roofline advise --measure --models ~/models/llama3-8b-*.gguf")
        return 2
    paths = [p for p in expand_model_args(args.models) if os.path.isfile(p)]
    if len(paths) < 2:
        _eprint("error: --measure needs at least two files to compare. One file has no "
                "ranking.")
        return 2
    try:
        binary = bench.find_llama_bench(args.llama_bench)
    except bench.BenchNotFound as exc:
        _eprint(f"error: {exc}")
        return 3

    threads = ([int(t) for t in args.threads_sweep.split(",") if t.strip()]
               if args.threads_sweep else sysinfo.default_thread_sweep()[:3])
    system = sysinfo.collect()
    print(f"machine : {system.get('cpu')} ({system.get('logical_cores')} logical cores)")
    print(f"plan    : {len(paths)} file(s) x {len(threads)} thread setting(s)")
    print()

    measured: List[Dict[str, Any]] = []
    for path in paths:
        entry = new_model_entry(path)
        stream = entry.get("streamed")
        print(f"  {entry['name']}")
        for t in threads:
            try:
                r = bench.bench_model(binary, path, threads=t, n_prompt=128, n_gen=128,
                                      reps=args.reps or 3, timeout=args.timeout)
            except bench.BenchError as exc:
                _eprint(f"    t={t}: FAILED -- {exc}")
                continue
            dec = r.get("decode_ts")
            if not dec:
                continue
            sb = stream["streamed_bytes"] if stream else None
            measured.append({
                "format": entry.get("quant") or entry["name"],
                "name": entry["name"],
                "threads": t,
                "tok_s": dec,
                "streamed_MiB": round(sb / (1024 * 1024)) if sb else None,
                "streamed_GBs": (dec * sb / 1e9) if sb else None,
                "file_MiB": (entry.get("file_bytes") or 0) / (1024 * 1024),
            })
            print(f"    t={t:<3} decode {dec:7.2f} tok/s")
    if not measured:
        _eprint("\nerror: nothing decoded. Check these files load with your llama-bench.")
        return 1

    print()
    print(advisor.render_measured(system, measured, threads))
    if args.markdown:
        with open(args.markdown, "w", encoding="utf-8") as f:
            f.write(advisor.render_measured_markdown(system, measured, threads) + "\n")
        print(f"  wrote {args.markdown}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"tool_version": __version__, "system": system,
                       "measured": measured}, f, indent=2)
        print(f"  wrote {args.json}")
    return 0


# --------------------------------------------------------------------------- parser

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="llama-roofline",
        description="What is limiting your llama.cpp token generation? Measure it.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "start here:\n"
            "  llama-roofline diagnose                 find your models and answer\n"
            "  llama-roofline diagnose model.gguf      answer for one model\n"
            "  llama-roofline advise                   which quant to run on this core\n"
            "  llama-roofline inspect model.gguf       what is actually inside the file\n"
            "\n"
            "the longer path:\n"
            "  llama-roofline run --models ~/models/*.gguf\n"
            "  llama-roofline run --models a.gguf b.gguf --threads 1,2,4,8 --reps 5\n"
            "  llama-roofline membw\n"
            "  llama-roofline report out/roofline.json --markdown report.md\n"
        ),
    )
    p.add_argument("--version", action="version", version=f"llama-roofline {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser(
        "diagnose",
        help="one command, no flags: what is limiting token generation here",
        description="Measures this machine's memory ceiling, benchmarks your GGUF "
                    "models, and says in plain words what is holding your tokens per "
                    "second down. With no arguments it looks for models itself.",
    )
    d.add_argument("models", nargs="*", metavar="GGUF",
                   help="GGUF files, directories or globs. Omit and it searches the "
                        "usual places and $LLAMA_MODELS.")
    d.add_argument("--llama-bench", metavar="PATH",
                   help="path to llama-bench (default: $LLAMA_BENCH, PATH, then the "
                        "usual build locations)")
    d.add_argument("--threads", "-t", metavar="LIST",
                   help="comma-separated thread counts (default: 1, half, physical, "
                        "logical)")
    d.add_argument("--reps", type=int, default=None, metavar="N",
                   help="llama-bench repetitions per test (default: 3, or 1 with --quick)")
    d.add_argument("--quick", action="store_true",
                   help="fewer thread settings, shorter generations, one repetition. "
                        "Faster, noisier.")
    d.add_argument("--peak-bw", type=float, default=None, metavar="GBS",
                   help="skip the microbenchmark and use this ceiling in GB/s")
    d.add_argument("--skip-membw", action="store_true",
                   help="do not measure the ceiling; there will be no verdict")
    d.add_argument("--file-bytes", action="store_true",
                   help="count bytes per token as the whole model rather than the "
                        "tensors decode actually reads. Matches what `run` reports.")
    d.add_argument("--gpu-layers", type=int, default=0, metavar="N",
                   help="layers to offload (default: 0, a CPU and system-RAM answer)")
    d.add_argument("--timeout", type=float, default=3600.0, metavar="SEC")
    d.add_argument("--out", "-o", default="llama-roofline-out", metavar="DIR")
    d.add_argument("--markdown", metavar="PATH", help="shareable report path override")
    d.add_argument("--quiet", "-q", action="store_true")
    d.set_defaults(func=cmd_diagnose)

    a = sub.add_parser(
        "advise",
        help="which quantization format to run on this machine's cores",
        description="Ranks GGUF quantization formats for the CPU core this machine "
                    "has, from measurements on an Arm Cortex-A76, an Intel Golden Cove "
                    "P-core and an Intel Gracemont E-core. On anything else it says so "
                    "instead of guessing, and --measure builds the ranking from your "
                    "own files.",
    )
    a.add_argument("--core", metavar="NAME",
                   help="force a core instead of detecting one (see --list-cores)")
    a.add_argument("--list-cores", action="store_true",
                   help="show which cores this build has measurements for")
    a.add_argument("--threads", type=int, default=None, metavar="N",
                   help="advise for this thread count (default: the fastest measured)")
    a.add_argument("--scale", type=float, default=None, metavar="B",
                   help="model size in billions of parameters (default: 0.5)")
    a.add_argument("--measure", action="store_true",
                   help="ignore the shipped matrix and benchmark your own files")
    a.add_argument("--models", "-m", nargs="+", metavar="GGUF",
                   help="files to benchmark with --measure; several quantizations of "
                        "one model is the useful case")
    a.add_argument("--threads-sweep", metavar="LIST",
                   help="thread counts for --measure (default: 1, half, physical)")
    a.add_argument("--reps", type=int, default=None, metavar="N")
    a.add_argument("--llama-bench", metavar="PATH")
    a.add_argument("--timeout", type=float, default=3600.0, metavar="SEC")
    a.add_argument("--markdown", metavar="PATH", help="also write a shareable Markdown copy")
    a.add_argument("--json", metavar="PATH", help="also write the machine-readable form")
    a.set_defaults(func=cmd_advise)

    i = sub.add_parser(
        "inspect",
        help="what a GGUF file actually contains, which its name does not tell you",
        description="Prints the per-tensor type map, how many bytes decode reads per "
                    "token, and how much of that is the output head. Two files with the "
                    "same format label can hold different types and decode at very "
                    "different speeds; this is how you tell them apart.",
    )
    i.add_argument("models", nargs="+", metavar="GGUF")
    i.add_argument("--tensors", action="store_true", help="list every tensor")
    i.set_defaults(func=cmd_inspect)

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
