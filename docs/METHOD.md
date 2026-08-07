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

The tool prefers `model_size` from llama-bench's own JSON, which is the resident tensor
size, and falls back to the file size on disk (summed across shards for a multi-part
GGUF). The two differ by one or two percent, mostly GGUF metadata.

This is exact for a dense transformer at short context. It is wrong in two known ways:

- **Mixture of experts.** Only the routed experts are read per token, so bytes-per-token
  is far below the model size. The tool parses the GGUF header, detects `expert_count`,
  flags such models, plots them with a distinct marker, and **excludes them from the
  fit** rather than quietly bending the line.
- **Long context.** The KV cache is read per token too, and it grows with context length.
  At the default 128-token generation it is negligible next to the weights. At long
  context it is not, and the roofline stops predicting throughput. This is measured, not
  asserted: see [Long context](#7-long-context-where-this-model-stops-working) below.

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

- **Output quality.** Nothing here evaluates perplexity or task accuracy. A Q2 model is
  faster than a Q8 model and that says nothing about whether it is still useful.
- **Energy.** Not in this release. Energy per token is not simply proportional to time
  per token, and measuring it properly needs a power rail this tool cannot assume.
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

## 8. Provenance

The analysis reimplements, in portable form, the decode-roofline methodology from a study
of LLM inference on a 2 GB Raspberry Pi 5 and an x86 laptop. The tool reproduces that
study's published fit exactly when handed its raw data: see
`examples/rpi5-cortex-a76/convert_paper_data.py`, which converts the published
measurements and re-analyses them through this tool's own code path.

All throughput numbers come from [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT),
without which none of this exists.
