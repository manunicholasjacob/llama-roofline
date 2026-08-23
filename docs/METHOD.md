# Method

What the tool measures, how, and where it stops being true.

## 1. The model

A dense transformer generating tokens one at a time has to read every weight from memory
for every token it produces. There is no reuse to exploit: a single token's forward pass
touches each weight exactly once, so the arithmetic intensity of decode is close to one
multiply-accumulate per weight loaded. That is deep in the memory-bound region of a
roofline diagram, and it means throughput is set by how fast bytes arrive:

```
decode tok/s  =  BW_eff / bytes_per_token
```

where `BW_eff` is the memory bandwidth the inference engine actually achieves.

Prefill is the opposite. Processing a prompt of N tokens is a matrix-matrix product: the
same weights are reused across all N positions, so arithmetic intensity scales with N and
prefill lands in the compute-bound region. This is why the two phases respond so
differently to adding cores, and why the report treats them separately.

The consequence people care about: on a memory-bound machine, **halving the bytes halves
the time per token**. Quantization is not a tradeoff between speed and memory, it is the
same lever pulled once.

## 2. Measuring the ceiling

`llama-roofline membw` measures the achievable streaming-read rate. Portability rules out
shipping a compiled STREAM binary, so the tool uses numpy ufuncs, which run in C with the
GIL released, over disjoint slices in a thread pool.

Three read-only kernels run, because no single one saturates every machine:

| kernel | operation | bytes counted | why it is here |
|---|---|---|---|
| `sum` | `a[s:e].sum()` | `n` | one stream, but the reduction can bottleneck on the accumulator |
| `max` | `a[s:e].max()` | `n` | one stream, no accumulation dependency |
| `dot` | `np.dot(a[s:e], b[s:e])` | `2n` | two streams through BLAS, usually the fastest |

`copy` (`b[s:e] = a[s:e]`) is also run and reported, but it is **excluded from the
ceiling**: a write-allocating store pulls the destination line into cache before
overwriting it, so copy moves more traffic than its nominal byte count and would inflate
the number relative to what decode actually does, which is pure reading.

The ceiling is the best read-only result across all kernels and thread counts. The
working set defaults to 512 MB, capped at one eighth of system RAM so a 2 GB board does
not get pushed into swap, and each measurement is a best-of-N to reject scheduler noise.

**Run it on an idle machine.** Best-of-N rejects a single unlucky repetition, but it cannot
save you from *constant* contention: if something else is using the machine throughout,
every repetition is depressed and the "best" is simply wrong, which makes the ceiling too
low and every utilisation percentage too high. Measured on one laptop, the same
microbenchmark returned 50.7 GB/s on an idle machine and 14.1 GB/s while a package install
was finishing. To catch this the tool records every repetition and compares the median to
the best; when they disagree by more than 25% it reports the measurement as unstable and
tells you to re-run or supply `--peak-bw`.

**This number is a lower bound and the tool says so.** Hand-written SIMD kernels in
llama.cpp can and do stream faster than numpy. When measured decode bandwidth comes out
above the ceiling, the report prints a warning explaining that the percentages are lower
bounds rather than reporting an impossible ">100% of peak". If you have a real STREAM
number for your machine, pass it with `--peak-bw` and every percentage tightens.

## 3. Measuring throughput

Everything comes from the user's own `llama-bench`, invoked as:

```
llama-bench -m MODEL -t THREADS -p N_PROMPT -n N_GEN -r REPS -o json
```

`llama-bench` is the only measurement path in llama.cpp that is non-interactive, reports
prefill and generation separately, and already handles warmup and repetitions. A record
with `n_prompt > 0, n_gen == 0` is prefill; `n_gen > 0, n_prompt == 0` is decode.

Two details that are easy to get wrong:

- **`-p 0` is not used.** Some llama.cpp builds report a spuriously low generation rate
  when the prompt length is zero. The tool always runs a real prompt (128 tokens by
  default) alongside generation, and refuses `--n-prompt 0`.
- **`llama-cli` is never used.** It can block on interactive input even with generation
  flags set, which turns a scripted benchmark into a hang.

Each model is measured across a thread sweep (1, half the physical cores, all physical
cores, all logical cores, by default). The best decode result across the sweep is the
model's operating point; the thread knee is the smallest count within 2% of that best.

## 4. Bytes per token

There are two conventions in this tool and the difference between them is not cosmetic.

**`run` uses the model's resident size.** It prefers `model_size` from llama-bench's own
JSON and falls back to the file size on disk, summed across shards for a multi-part GGUF.
The two differ by one or two percent, mostly GGUF metadata. This is the convention the
published fits that this tool reproduces were computed with, and `run` keeps it so that
the reproduction keeps working. A tool that quietly stops reproducing its own reference
result has lost the only correctness check it has.

**`diagnose` counts the tensors decode actually reads.** It parses the GGUF tensor table
and sums the repeating layers plus the output head. The token embedding is excluded,
because generating a token looks up one row of it rather than streaming the whole matrix,
unless the model ties the embedding to the output head, in which case that one tensor is
the output projection, is streamed in full, and is counted.
On a 0.5B model that removes 18 to 22% of the file, and the amount removed is
format-dependent, so it changes which format looks efficient and not just by how much.

The second convention is the more accurate account of decode traffic, which is why the
newer command uses it, and `diagnose --file-bytes` switches back when you want the two
commands to agree.

The tensor-table parse is checked against an external result rather than against itself.
A published study parsed the per-tensor type maps of eight canonical Qwen2.5-0.5B
artifacts and reported their streamed sizes as 317, 330, 330, 332, 334, 374, 477 and
501 MiB. This tool's parser, given the same eight files, returns every one of them to
within 0.2%.

Both conventions are exact for a dense transformer at short context, and both are wrong
in the same two known ways:

- **Mixture of experts.** Only the routed experts are read per token, so bytes-per-token
  is far below the model size. The tool parses the GGUF header, detects `expert_count`,
  flags such models, plots them with a distinct marker, and **excludes them from the
  fit** rather than quietly bending the line.
- **Long context.** The KV cache is read per token too, and it grows with context length.
  At the default 128-token generation it is negligible next to the weights. At long
  context it is not, and the roofline stops predicting throughput. This is measured, not
  asserted: see [Long context](#7-long-context-where-this-model-stops-working) below.

### What the tensor table also tells you

The same parse is what `inspect` prints, and it answers a question the filename cannot.
A GGUF format label names a quantization recipe, not a tensor type. `llama-quantize`
substitutes a different type per tensor when a shape does not divide evenly by the block
size, and it substitutes different ones depending on whether the file was built from FP16
or requantized from an intermediate. In a canonical Qwen2.5-0.5B Q4_K_M file, 12% of the
repeating-layer bytes are Q4_K and 54% are Q5_0, because the embedding dimension of 896
is not divisible by 256. Two files with the same name can therefore hold different maps
and decode at very different rates, which is what makes `inspect` worth running before
trusting any benchmark that did not pin its artifact.

## 5. The fit

`tok/s = BW * (1/bytes)` is a one-parameter model through the origin, fitted by ordinary
least squares over `x = 1/bytes`:

```
BW = sum(x*y) / sum(x*x)
```

R-squared is computed the standard way, against the mean of y, so it is comparable with a
regression that had an intercept. A tight fit across models of different sizes is the
actual evidence that decode is bandwidth-bound: it says one number, bytes, predicts
throughput. When the fit is loose (R-squared below 0.90 with three or more points) the
report says the setup is not cleanly bandwidth-bound and lists the usual causes.

At least two models are needed. With one model there is no fit, only a single point, and
the report says so instead of drawing a line through it.

## 6. What this does not measure

- **Output quality, on your machine.** Nothing this tool runs evaluates perplexity or task
  accuracy. A Q2 model is faster than a Q8 model and that says nothing about whether it is
  still useful. `advise` does show perplexity, but it comes from the shipped study, on one
  model family at one scale, and it is a property of those artifacts rather than of yours.
- **Energy, on your machine.** Measuring it needs a power rail this tool cannot assume.
  `advise` reports energy per token where the shipped study had one, which is the
  Raspberry Pi's PMIC and nowhere else, and it says the column is absent rather than
  printing a zero anywhere else.
- **GPU inference.** Offload is detected and warned about. The ceiling measured is system
  RAM bandwidth; offloaded layers stream from VRAM instead, so the percentages do not
  apply. `--gpu-layers 0` is the default for this reason.
- **Batched or concurrent serving.** Batching amortises the weight read across several
  sequences, which is exactly the escape hatch from this roofline. Single-stream decode
  is the worst case and the case most local users are in.

## 7. Long context: where this model stops working

The weights-only roofline is a short-context result. To find out how short, decode was
measured with the KV cache pre-filled to various depths (`llama-bench -d`, which
`llama-roofline run --depth N` exposes), on an Intel i7-12700H at 8 threads.

**Qwen2.5-0.5B Q4_K_M** (392 MB of weights, 12 KB of KV cache per token)

| context depth | decode tok/s | vs. empty cache | KV cache | predicted, if extra bytes were the whole story |
|---:|---:|---:|---:|---:|
| 0 | 83.6 | **100%** | 0 MB | 100% |
| 512 | 80.5 | **96%** | 6 MB | 98% |
| 2,048 | 62.2 | **74%** | 25 MB | 94% |
| 8,192 | 32.8 | **39%** | 101 MB | 80% |

**Qwen2.5-1.5B Q4_K_M** (980 MB of weights, 28 KB of KV cache per token)

| context depth | decode tok/s | vs. empty cache | KV cache | predicted, if extra bytes were the whole story |
|---:|---:|---:|---:|---:|
| 0 | 38.7 | **100%** | 0 MB | 100% |
| 512 | 37.6 | **97%** | 15 MB | 99% |
| 2,048 | 30.9 | **80%** | 59 MB | 94% |
| 8,192 | 17.5 | **45%** | 235 MB | 81% |

The 0.5B model was measured further still: **23% at 16k and 13% at 32k**, a 7.6x
slowdown from an empty cache.

The KV cache sizes are computed from the GGUF metadata:
`2 (K and V) x n_layer x n_head_kv x head_dim x 2 bytes` at f16.

**The measured falloff is much steeper than KV traffic alone can explain.** If decode were
still purely bandwidth-bound and the only change were extra bytes, throughput would scale
as `weights / (weights + KV)`. That predicts the "if bytes only" column, which is nowhere
near what actually happens. The rest is attention work over the cache, which grows with
context length and is not a streaming-bandwidth cost at all.

Two consequences, both of which the tool states rather than hides:

1. Quote the fitted `BW_eff` and the utilisation percentages as **short-context** numbers.
   They describe generation with an almost-empty cache.
2. If you actually run long contexts, measure at your context length with `--depth`. The
   headline fit will tell you what your setup can do at best, not what it does at 16k.

Making bytes-per-token KV-aware would fix the first-order part of this, but not the
attention term, which is why the fit is restricted to short context rather than patched.

## 8. Where the quantization rankings come from

`advise` does not measure anything on your machine unless you ask it to with `--measure`.
It reads `src/llama_roofline/data/format_matrix.csv`, which holds one row per measurement
with the file that measurement came from in its `source` column, and nothing in it is
averaged across cores.

The measurements are a controlled format study: Qwen2.5-0.5B-Instruct and
Qwen2.5-1.5B-Instruct, quantized from the official FP16 GGUF into eight formats, run
under `llama-bench -p 128 -n 128` with five repetitions on three microarchitectures. The
Cortex-A76 is a Raspberry Pi 5. The Golden Cove and Gracemont rows are one i7-12700H with
`llama-bench` pinned by affinity mask to one core type at a time, so the core varies while
the memory system stays fixed. Energy is the Pi's PMIC rail sum, collected in a separate
identical run because sampling power perturbs decode, and it is uncalibrated, so it
supports ratios between formats and not absolute joules. Perplexity is 100 chunks of 512
tokens from a public-domain corpus, measured at 0.5B on the same artifacts, so it is a
property of the file rather than of the core. The dataset is archived at
[10.5281/zenodo.21938812](https://doi.org/10.5281/zenodo.21938812).

Four rules keep the advice honest:

- **No global ranking.** Every table is one core at one thread count at one model scale,
  because the orderings genuinely differ. Q4_0 leads Golden Cove at two threads; IQ4_NL
  leads Gracemont at four; the A76 converges the whole 4-bit class to within 4%.
- **The evidence level is stated per core.** Detection distinguishes the exact machine
  measured, a different chip with the same core design, a close relative, and nothing.
  Only the first three produce a table, the third is labelled as extrapolation in the
  output, and the fourth produces the recipe for measuring it yourself instead.
- **Speed is compared at matched bytes.** Ranking Q8_0 against Q4_0 measures size, which
  everyone already knows about. The advisor also reports the spread within the largest
  group of formats that stream within 3% of the same bytes, which is the only comparison
  that isolates what the format itself costs.
- **Absent is not zero.** There is no energy column on x86, because Windows reports no
  package power without a driver and the study did not measure it.

## 9. Provenance

The analysis reimplements, in portable form, the decode-roofline methodology from a study
of LLM inference on a 2 GB Raspberry Pi 5 and an x86 laptop. The tool reproduces that
study's published fit exactly when handed its raw data: see
`examples/rpi5-cortex-a76/convert_paper_data.py`, which converts the published
measurements and re-analyses them through this tool's own code path.

All throughput numbers come from [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT),
without which none of this exists.
