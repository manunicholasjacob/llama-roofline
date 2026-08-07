"""Convert the published Raspberry Pi 5 llama-bench data into a llama-roofline results file.

Provenance matters here, so read this before trusting the numbers in
``roofline.json`` next to this script.

The Pi 5 figures were NOT produced by running ``llama-roofline`` on the Pi. They are the
raw ``llama-bench`` measurements from the study this tool grew out of, re-expressed in
the tool's schema and re-analysed by the tool's own ``roofline.analyze``. That is a
faithful reuse -- the underlying measurement command is the same one ``llama-roofline
run`` issues (``llama-bench -m MODEL -t T -p P -n N -r 3 -o json``) -- but two details
differ from a native run and are recorded in the output:

  * bytes-per-token comes from the GGUF *file size*, not llama-bench's ``model_size``
    field, so it is ~1-2% larger. Utilisation figures are correspondingly ~1-2% high.
  * the bandwidth ceiling (13.98 GB/s) is a separately measured STREAM-style read peak
    for this board, supplied here the way ``--peak-bw`` supplies one, rather than from
    this tool's numpy microbenchmark.

Source data: https://github.com/manunicholasjacob/edge-llm-memory-wall
Usage: python convert_paper_data.py <path-to-edge-llm/results> [-o roofline.json]
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))

from llama_roofline import SCHEMA_VERSION, __version__, roofline  # noqa: E402

PEAK_BW_GBS = 13.98
PI_THREADS_QUANT = 4  # the quant sweep in the source data was run at 4 threads

DISPLAY = {
    "qwen0.5b-q4km": ("Qwen2.5-0.5B", "Q4_K_M"),
    "llama1b-q4km": ("Llama-3.2-1B", "Q4_K_M"),
    "qwen1.5b-q4km": ("Qwen2.5-1.5B", "Q4_K_M"),
    "qwen0.5b-q2k": ("Qwen2.5-0.5B", "Q2_K"),
    "qwen0.5b-q3km": ("Qwen2.5-0.5B", "Q3_K_M"),
    "qwen0.5b-q5km": ("Qwen2.5-0.5B", "Q5_K_M"),
    "qwen0.5b-q8": ("Qwen2.5-0.5B", "Q8_0"),
}


def build_models(l1, l2):
    models = []
    for key, entry in l1["models"].items():
        disp, quant = DISPLAY.get(key, (key, None))
        runs = []
        for t, r in sorted(entry["threads"].items(), key=lambda kv: int(kv[0])):
            if r.get("tg_ts"):
                runs.append({"threads": int(t), "decode_ts": r["tg_ts"],
                             "prefill_ts": r.get("pp_ts")})
        models.append({
            "name": f"{disp} {quant}" if quant else disp,
            "quant": quant, "arch": None, "is_moe": False,
            "file_bytes": int(entry["file_mb"] * 1e6),
            "bytes_per_token": int(entry["file_mb"] * 1e6),
            "bytes_source": "GGUF file size (source data predates model_size capture)",
            "runs": runs,
        })

    seen = {m["name"] for m in models}
    for key, entry in (l2 or {}).get("models", {}).items():
        disp, quant = DISPLAY.get(key, (key, entry.get("quant")))
        name = f"{disp} {quant}" if quant else disp
        if name in seen or not entry.get("tg_ts"):
            continue
        models.append({
            "name": name, "quant": entry.get("quant"), "arch": None, "is_moe": False,
            "file_bytes": int(entry["file_mb"] * 1e6),
            "bytes_per_token": int(entry["file_mb"] * 1e6),
            "bytes_source": "GGUF file size (source data predates model_size capture)",
            "runs": [{"threads": PI_THREADS_QUANT, "decode_ts": entry["tg_ts"],
                      "prefill_ts": entry.get("pp_ts")}],
        })
    return models


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_dir", help="edge-llm/results directory holding L1_roofline.json")
    ap.add_argument("-o", "--out", default=os.path.join(os.path.dirname(__file__),
                                                        "roofline.json"))
    args = ap.parse_args()

    with open(os.path.join(args.results_dir, "L1_roofline.json"), encoding="utf-8") as f:
        l1 = json.load(f)
    l2_path = os.path.join(args.results_dir, "L2_quant.json")
    l2 = None
    if os.path.exists(l2_path):
        with open(l2_path, encoding="utf-8") as f:
            l2 = json.load(f)

    models = build_models(l1, l2)
    analysis = roofline.analyze(models, PEAK_BW_GBS)
    analysis["warnings"].insert(
        0,
        "converted data: measurements come from the published Raspberry Pi 5 study, not "
        "from a native llama-roofline run. Bytes-per-token uses the GGUF file size rather "
        "than llama-bench's model_size, so utilisation reads ~1-2% high. See "
        "convert_paper_data.py for the full provenance note.",
    )

    results = {
        "schema_version": SCHEMA_VERSION,
        "tool_version": __version__,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "provenance": "converted from published llama-bench data; see convert_paper_data.py",
        "system": {
            "os": "Linux", "os_release": "6.x (Raspberry Pi OS, 64-bit)",
            "machine": "aarch64", "cpu": "Raspberry Pi 5 (Broadcom BCM2712, Cortex-A76 @ 2.4 GHz)",
            "logical_cores": 4, "physical_cores": 4, "ram_bytes": 2_000_000_000,
        },
        "llama_cpp": {"backends": "CPU", "notes": "native aarch64 build, ARM dotprod kernels"},
        "settings": {"threads": [1, 2, 3, 4], "n_prompt": 512, "n_gen": 128, "reps": 3,
                     "gpu_layers": 0,
                     "notes": "quant-sweep points were measured at 4 threads with -p 128"},
        "membw": {"peak_read_GBs": PEAK_BW_GBS,
                  "source": "external STREAM-style read benchmark for this board"},
        "analysis": analysis,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {args.out}")
    fit = analysis.get("fit")
    if fit:
        print(f"  fit: {fit['bw_eff_GBs']:.2f} GB/s / bytes, R^2 = {fit['r2']:.4f}, "
              f"n = {fit['n_points']}")
    print(f"  verdict: {analysis.get('verdict')}")


if __name__ == "__main__":
    main()
