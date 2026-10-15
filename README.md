# llama-roofline

**Why is your llama.cpp generation that fast and not faster? One command answers it.**

[![tests](https://github.com/manunicholasjacob/llama-roofline/actions/workflows/ci.yml/badge.svg)](https://github.com/manunicholasjacob/llama-roofline/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/llama-roofline.svg)](https://pypi.org/project/llama-roofline/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21842493.svg)](https://doi.org/10.5281/zenodo.21842493)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/pypi/pyversions/llama-roofline.svg)](https://pypi.org/project/llama-roofline/)
[![Results gallery](https://img.shields.io/badge/results-add%20your%20machine-blue.svg)](https://github.com/manunicholasjacob/llama-roofline/issues/new?template=results-gallery.md)
[![Docs](https://img.shields.io/badge/docs-site-orange.svg)](https://manunicholasjacob.com/llama-roofline/)
[![GGUF Inspector](https://img.shields.io/badge/GGUF%20Inspector-in%20your%20browser-orange.svg)](https://manunicholasjacob.com/llama-roofline/inspector.html)

When a local LLM generates text one token at a time, it has to read **every weight in the
model, from memory, for every single token**. Nothing is reused. That makes token
generation a memory-bandwidth problem, not a compute problem, and it means your tokens per
second are governed by a single equation:

```
decode tok/s  =  BW_eff / bytes_per_token
```

`llama-roofline` measures both sides of that equation on your machine and tells you where
you sit, in words rather than in a table of numbers.

```bash
pip install llama-roofline
llama-roofline diagnose
```

No flags, no config. It looks for your GGUF files, finds `llama-bench`, measures what your
memory can actually sustain, benchmarks across a thread sweep, and prints this:

```
========================================================================
  llama-roofline v0.2.3  --  diagnosis
========================================================================
  Machine   : 12th Gen Intel(R) Core(TM) i7-12700H
              20 logical / 14 physical cores, 34.0 GB RAM, Windows AMD64
  Memory    : 54.3 GB/s sustained read (measured: max kernel, 20 threads)
  llama.cpp : build 10154 (0e4a03622), backends: CPU

  THE ANSWER
------------------------------------------------------------------------
  Yes. Decode is memory-bandwidth-bound, at 78% of what this machine
  sustains.

  Every token reads the whole model out of RAM, and at this utilisation
  the cores are mostly waiting for it. A faster CPU changes nothing
  here. Fewer bytes does.

  For qwen3b-q4km the ceiling on this machine is about 28 tok/s. You
  measured 21.9 at 8 threads.

  WHAT TO CHANGE
------------------------------------------------------------------------
  * Threads: use 8. Running 20 cost 25% of decode throughput, because
    decode is waiting on memory and more waiters do not make memory
    faster.
  * Size is the only dial that moves this. Halving the bytes read per
    token roughly doubles tok/s, which is why a smaller model or a lower
    quantization buys speed almost exactly in proportion to the bytes it
    removes.
  * Not every format of the same size costs the same, though. Run
    `llama-roofline advise` to see what has been measured.
  * Measured here: qwen0.5b-q4km is 4.91x smaller than qwen3b-q4km and
    decodes 4.28x faster.
  * Prompt processing is the opposite case: it scaled 4.9x with threads.
    If your workload is long prompts and short answers, you do want the
    cores.

  WHAT WAS MEASURED
------------------------------------------------------------------------
  model                     quant         size    decode   prefill  thr    GB/s  %ceil
  ------------------------------------------------------------------------------------
  qwen0.5b-q4km             Q4_K_M      392 MB     93.7t      277t    8    36.7    68%
  qwen1.5b-q4km             Q4_K_M      980 MB     39.7t      133t    8    38.9    72%
  qwen3b-q4km               Q4_K_M     1.92 GB     21.9t       76t    8    42.1    78%

  THE ROOFLINE
------------------------------------------------------------------------
    decode tok/s  =  37.18 GB/s  /  bytes per token
    fitted across 3 models, R^2 = 0.9960,
    taking each model at its own best thread count
```

That is a real run on a real laptop, trimmed of its caveats section, which is printed and
is not optional.

The line worth staring at is the thread one. Setting `-t 20` on a 20-thread machine looks
free. It cost a quarter of the decode throughput.

`diagnose` also writes `diagnosis.md`, which is the same thing shaped for pasting into an
issue or a forum thread when you want someone to look at your numbers.

### Use Ollama or LM Studio and have no llama.cpp build?

You can still look inside the models you already have. `inspect` reads only the file
header, needs no `llama-bench`, and takes about a second:

```bash
pip install llama-roofline
llama-roofline inspect ollama:qwen2.5:7b          # any model you have pulled
llama-roofline inspect ~/path/to/model.gguf         # LM Studio and every other GGUF
```

It prints what the file actually contains under its format label, how many bytes decode
reads per token, and how much of that is the output head. Divide your memory bandwidth by
that per-token figure and you have the ceiling on your generation speed before you run
anything.

### Check a model before you download it

`inspect` also reads a file on the Hugging Face Hub, or on any server that honours HTTP
range requests, without downloading the weights. Only the header is fetched, usually a few
MB of a file that may be tens of GB:

```bash
llama-roofline inspect hf:Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_k_m.gguf
llama-roofline inspect hf:OWNER/REPO@REVISION/FILE.gguf      # a branch, tag or commit
llama-roofline inspect https://huggingface.co/OWNER/REPO/blob/main/FILE.gguf
llama-roofline inspect hf:OWNER/REPO/FILE.gguf --peak-bw 50  # and the speed ceiling at 50 GB/s
```

A split model (`...-00001-of-00003.gguf`) is read part by part, locally or remotely, so
the figures cover the whole model. For a gated or private repository set `HF_TOKEN`; it is
sent to huggingface.co only. `--json` prints the same summary as data.

The same report runs in a web page, with nothing to install:
[GGUF Inspector](https://manunicholasjacob.com/llama-roofline/inspector.html). Pick a local
file (only its header is read, inside the browser) or name a Hub file. It runs this
package's own `gguf.py` and `inspection.py` in Pyodide, so it prints what the CLI prints.

## Which quantization should you run?

`advise` answers "which quantization format should I run on this CPU?" from measurements,
not from file names. On silicon it has data for, it prints a per-format table with the
measurement each row came from. On anything else it says so and measures your own files
instead of guessing:

```bash
llama-roofline advise
```

Read any 0.5B comparison knowing what is in the files. At that size most formats are
mostly not what they are named after, so part of what separates them is which types
`llama-quantize` substituted rather than which format was asked for. The report says so
where it prints such a table, and [the next section](#what-is-actually-in-that-gguf-file)
is how to check any file you have.

The data file ships with the package and records, for every row, the run it came from in
the `source` column. On silicon nobody measured, the tool says so and shows you how to measure
it rather than guessing:

```bash
llama-roofline advise --measure --models ~/models/qwen0.5b-*.gguf
```

That benchmarks the files you already have and prints the same table from your own numbers.

Most people have one file per model rather than several formats of one model, which
compares sizes instead of formats. If that is you:

```bash
llama-roofline advise --plan ~/models/qwen0.5b-f16.gguf
```

prints the `llama-quantize` commands that build a comparable set, and refuses if you point
it at a file that is already quantized, so every file in the set comes from the same
source.

## What is actually in that GGUF file?

A format label names a recipe, not a type, and at small model sizes the recipe substitutes
so heavily that the name stops describing the file.

```bash
llama-roofline inspect ~/models/qwen0.5b-Q4_K_M.gguf
```

```
  label      : Q4_K_M   architecture: qwen2
  actually   : 12% Q4_K in the repeating layers, 5.52 bits per weight
  file       : 491 MB, 291 tensors
  per token  : 392 MB read (81% of the file)
  output head: 145 MB, 37% of what is read per token

  TENSOR TYPES
------------------------------------------------------------------------
    type        tensors        MB   share
    Q5_0            133       267     54%
    Q8_0             13       146     30%
    Q6_K             12        43      9%
    Q4_K             12        29      6%
    F32             121         0      0%

  WHAT THE LABEL MEANS HERE
------------------------------------------------------------------------
  Only 12% of this file's repeating-layer bytes are Q4_K. Most of it
  is Q5_0 70%, Q6_K 17%, Q8_0 1%. The output head is Q8_0. It stores
  5.52 bits per weight. A GGUF label names a recipe rather than a
  type. llama-quantize substitutes per tensor when a shape does not
  divide by the block size. None of that is visible from the filename,
  and at small model sizes the substitution can take over the file.
```

Eight formats of Qwen2.5-0.5B built from one FP16 source, measured over their repeating
layers:

| format | on-label | bits/weight | what most of it is |
|---|---:|---:|---|
| Q4_0 | 100% | 4.50 | |
| Q8_0 | 100% | 8.50 | |
| IQ4_NL | 95% | 4.55 | |
| IQ4_XS | 24% | 4.48 | IQ4_NL 70% |
| Q6_K | 24% | 7.93 | Q8_0 76% |
| Q4_K_M | 12% | 5.52 | Q5_0 70% |
| Q3_K_M | 0% | 4.57 | Q4_0 64%, Q4_K 28% |
| Q2_K | 0% | 4.20 | Q4_0 75%, Q3_K 24% |

Three of eight are what they say. K-quants want blocks of 256 and this model's embedding
dimension is 896, so `llama-quantize` substitutes per tensor, warns per tensor, and prints
no summary. At 1.5B the same recipes land at 58 to 100% on-label, so this is a small-model
effect, and small models are where quantization comparisons usually get run.

Which type fills in depends on the llama.cpp build that did the quantizing. Qwen's own
Q2_K and Q3_K_M files on the Hub carry IQ4_NL where the build behind this table wrote
Q4_0 (75% and 64% of the repeating layers, at the same 4.20 and 4.57 bits per weight,
since both types store 4.5 bits). Run `inspect` on the file you actually have.

`inspect` is also where the bytes-per-token figure comes from. The token embedding is a row
lookup rather than a stream, so file size overstates what decode actually reads, by 18 to
22% on these files.

## The longer path

If you want the full sweep, the figure and the fitted roofline across every model you
own, `run` is still there and unchanged.

```bash
llama-roofline run --models ~/models/*.gguf
```

## What it tells you that a benchmark does not

`llama-bench` already tells you your tokens per second. This tells you **why that number
is what it is, and which knob actually moves it**:

- **Are you at the wall?** If decode is at 85% of your memory ceiling, a faster CPU will
  do nothing for you. If it is at 30%, something else is wrong and the report says what to
  check.
- **How much will a smaller quant buy?** Not a guess. The fit predicts it, and the report
  shows the ratio measured on your own models.
- **How many threads should you actually use?** Decode saturates early and then *goes
  backwards*. Prefill keeps scaling. The report gives you the knee for both.
- **Where does the model stop applying?** MoE models are detected and excluded from the
  fit. Measurements that exceed the ceiling are reported as lower bounds rather than as an
  impossible ">100% of peak".

## Commands

```bash
# the zero-configuration answer: find models, measure, explain
llama-roofline diagnose

# one model, or a directory, or a glob
llama-roofline diagnose ~/models/llama3-8b-q4km.gguf
llama-roofline diagnose ~/models --quick        # fewer settings, one repetition

# which quantization format to run on this machine's cores
llama-roofline advise
llama-roofline advise --core cortex-a76 --threads 4
llama-roofline advise --measure --models ~/models/qwen-*.gguf   # from your own files
llama-roofline advise --plan ~/models/qwen-f16.gguf             # build a set worth comparing

# what a GGUF file actually contains
llama-roofline inspect ~/models/model.gguf
llama-roofline inspect ~/models/model.gguf --tensors
llama-roofline inspect hf:OWNER/REPO/FILE.gguf --json     # remote: header only

# the full sweep: every model, the fit, the figure
llama-roofline run --models ~/models/*.gguf
llama-roofline run --models a.gguf b.gguf --threads 1,2,4,8,16 --reps 5

# just this machine's memory-bandwidth ceiling
llama-roofline membw

# already have a STREAM number? skip the microbenchmark and tighten the percentages
llama-roofline run --models ~/models --peak-bw 204.8

# re-render a report or figure from saved results
llama-roofline report out/roofline.json --markdown report.md
```

Useful flags: `--llama-bench PATH` if it is not found automatically (or set `$LLAMA_BENCH`),
`--n-gen` / `--n-prompt` to change the generation and prompt lengths, `--depth N` to
generate with N tokens already in the KV cache, `--gpu-layers` (default `0`, see
limitations), `--out DIR`, `--no-plot`, `--quiet`. `--help` on any subcommand lists
everything.

`--depth` is the one to reach for if you run long contexts, because the weights-only
roofline is a short-context result. See [Long context](#long-context) below.

### diagnose and run count bytes differently, on purpose

`diagnose` parses the GGUF tensor table and counts only what decode reads per token: the
repeating layers and the output head, with the token embedding excluded because generation
looks up one row of it rather than streaming it. On a 0.5B model that is 18 to 22% fewer
bytes than the file, and the difference is format-dependent, so it changes which format
looks efficient.

`run` keeps the older convention, the model's resident size as `llama-bench` reports it,
because that is what the published fits this tool reproduces were computed with, and a
tool that quietly stops reproducing its own reference result is not worth much. Pass
`--file-bytes` to `diagnose` if you want the two to agree.

## Output

`diagnose` writes two files to `--out` (default `./llama-roofline-out`):

| file | what it is |
|---|---|
| `diagnosis.md` | the shareable one: paste it into an issue or a forum thread as is |
| `diagnosis.json` | everything, versioned schema, for your own analysis |

`run` writes four: `report.txt`, `report.md`, `roofline.png` and `roofline.json`.
`advise` prints to the terminal and takes `--markdown` and `--json` if you want files.

## Install

```bash
pip install llama-roofline
```

Or as an isolated command-line tool, without touching your environment:

```bash
pipx install llama-roofline        # or: uv tool install llama-roofline
uvx llama-roofline inspect model.gguf   # one-off run, nothing installed
```

Python 3.9 or newer, on Linux, macOS or Windows. That pulls in `numpy` and `matplotlib` so
the tool works end to end on first run.

The **analysis core is pure standard library**. `numpy` is used only to measure the
bandwidth ceiling (skip it with `--peak-bw`) and `matplotlib` only to draw the figure
(skip it with `--no-plot`). `advise` and `inspect` need neither, and CI has a job that
proves all of it still runs with both absent. So on a constrained box,
`pip install llama-roofline --no-deps` gets you a working tool as long as you supply the
ceiling yourself with `--peak-bw`.

Do not have llama.cpp yet? You almost certainly do not need to build it. `llama-bench`
ships with every packaged copy:

```bash
brew install llama.cpp                          # macOS, Linux
winget install llama.cpp                        # Windows
conda install -c conda-forge llama.cpp          # anywhere
```

Prebuilt archives for macOS, Linux, Windows and Android are attached to every build at
[github.com/ggml-org/llama.cpp/releases](https://github.com/ggml-org/llama.cpp/releases).
From source, if you would rather:

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build && cmake --build build --target llama-bench -j
```

Ollama and LM Studio both bundle llama.cpp and neither exposes `llama-bench`, so having
one of them installed does not count.

## Results gallery

See [`examples/`](examples/) for full output from an Intel i7-12700H (DDR5) and a
Raspberry Pi 5 (LPDDR4X). Those two machines differ by 3.5x in fitted bandwidth and by
roughly 20x in price, and both land in the same place: decode between 65% and 97% of the
memory ceiling, throughput tracking `1/model_bytes` with an R^2 above 0.98. Both fits take
each model at its own best thread count, which is the upper envelope of the sweep rather
than any one setting, and every report the tool prints now says so next to the number.

**Please add yours.** Open an issue with the "Results gallery" template and paste your
`report.md`. Hardware I do not own is the most useful contribution anyone can make, and a
result that *contradicts* the model is more interesting than one that confirms it.
No llama.cpp build? [`examples/colab_cpu.ipynb`](examples/colab_cpu.ipynb) runs the whole
thing on a Colab CPU runtime or any Linux x86-64 machine and prints the issue body for you.

## How it works

[`docs/METHOD.md`](docs/METHOD.md) has the full methodology: why decode is bandwidth-bound
and prefill is not, which bandwidth kernels are used and why `copy` is excluded from the
ceiling, how bytes-per-token is determined, and how the fit is computed.

The short version: three read-only numpy kernels over a thread pool establish the ceiling
as a **measured lower bound**; `llama-bench` supplies throughput; `tok/s = BW * (1/bytes)`
is fitted through the origin by least squares; R^2 against the mean of y tells you whether
the model actually holds on your machine.

## Long context

The fitted roofline assumes bytes-per-token is the weights alone, which is true with an
almost-empty KV cache and progressively less true as context grows. That is not a
hand-wave, it is measured. Decode on this laptop, with the cache pre-filled to each depth:

Decode throughput as a percentage of the same model with an empty cache:

| model | 0 ctx | 512 ctx | 2,048 ctx | 8,192 ctx |
|---|---:|---:|---:|---:|
| Qwen2.5-0.5B Q4_K_M | **100%** | **96%** | **74%** | **39%** |
| Qwen2.5-1.5B Q4_K_M | **100%** | **97%** | **80%** | **45%** |

The falloff is far steeper than the extra KV bytes alone would cause, so it is not just
more streaming; it is attention work over the cache, which grows with context length.

So: treat the headline fit and the utilisation percentages as **short-context** numbers,
and if you run long contexts, measure at your own context length:

```bash
llama-roofline run --models ~/models/*.gguf --depth 8192
```

Full analysis in [docs/METHOD.md](docs/METHOD.md#7-long-context-where-this-model-stops-working).

## Limitations

Read these before quoting a number.

- **CPU inference is the target.** GPU offload is detected and warned about, but the
  ceiling measured is *system RAM* bandwidth, not VRAM, so the percentages will not apply.
  `--gpu-layers 0` is the default for that reason.
- **Bytes-per-token is a short-context figure.** `diagnose` counts the tensors decode
  actually reads and `run` counts the model's resident size, and both are exact for a
  dense transformer with a nearly empty KV cache and an overestimate once the cache grows.
  Treat every percentage here as a short-context result.
- **Mixture-of-experts models are flagged, not solved.** Only the routed experts are read
  per token, so they sit off the dense roofline and are excluded from the fit.
- **The ceiling is a lower bound.** llama.cpp's hand-written SIMD kernels can stream faster
  than numpy. If your decode exceeds the measured ceiling the report tells you so instead
  of printing nonsense. Pass `--peak-bw` with a real STREAM number to tighten it.
- **Run it on an idle machine.** Background load depresses the ceiling and inflates every
  percentage derived from it. The tool checks its own repetitions for disagreement and
  flags the measurement as unstable when it finds it, but the cheapest fix is to close
  things first.
- **Throughput only, except where the advisor says otherwise.** Nothing this tool measures
  on your machine touches output quality. The perplexity column in `advise` comes from the
  shipped study, on one model at one scale, and is not a measurement of your files.
- **The advisor knows one core.** The Raspberry Pi 5's Cortex-A76, at 0.5B and 1.5B, on
  one model family. Close relatives are labelled as extrapolation and everything
  else gets told to measure for itself. That is a narrow base for a broad question, and
  widening it is what the results gallery is for.
- **Single-stream decode only.** Batching amortises the weight read across sequences, which
  is precisely the escape hatch from this roofline. This measures the worst case, which is
  the case most local users are actually in.

## Related tools

Other tools by the same author for people measuring local LLM inference:

- [ml-systems-lab](https://github.com/manunicholasjacob/ml-systems-lab): one YAML config drives llama.cpp and ONNX Runtime sweeps across laptops, Raspberry Pi and GPUs, with power and thermal telemetry.
- [qgemv-roofline](https://github.com/manunicholasjacob/qgemv-roofline): quantized GEMV kernels for batch-1 LLM decode, written from scratch and measured against a bandwidth roof the repository measures for itself.
- [gguf-faultscope](https://github.com/manunicholasjacob/gguf-faultscope): what one flipped bit does to a quantized GGUF model, and where it has to land to matter.

## Citing

If this tool is useful in something you publish or post, please cite it. See
[`CITATION.cff`](CITATION.cff), or:

> M. N. Jacob, *llama-roofline: a portable memory-bandwidth roofline for llama.cpp*,
> v0.2.3, 2026. doi:[10.5281/zenodo.21842493](https://doi.org/10.5281/zenodo.21842493)

```bibtex
@software{jacob_llama_roofline,
  author    = {Jacob, Manu Nicholas},
  title     = {llama-roofline: a portable memory-bandwidth roofline for llama.cpp},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.21842493},
  url       = {https://github.com/manunicholasjacob/llama-roofline}
}
```

`10.5281/zenodo.21842493` is the concept DOI and always resolves to the newest version.
To cite this exact release, use `10.5281/zenodo.23341323`.

The methodology comes from a study of LLM inference on a 2 GB Raspberry Pi 5 and an x86
laptop, currently under review; the reproducibility artifact for that work is at
[edge-llm-memory-wall](https://github.com/manunicholasjacob/edge-llm-memory-wall). The
underlying DRAM-ceiling method follows the earlier memory-wall characterisation in the
same line of work.

## Credits

All throughput numbers come from [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT),
without which none of this exists. This tool drives it and does arithmetic on the output;
the hard part was already done by its contributors.

MIT licensed. Contributions welcome, see [CONTRIBUTING.md](CONTRIBUTING.md).
