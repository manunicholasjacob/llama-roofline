# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-08-23

### Added, second pass
- `llama-roofline advise --plan <model-f16.gguf>` prints the `llama-quantize` commands
  that build a set worth comparing, and refuses when pointed at a file that is already
  quantized. Most people have one file per model rather than several formats of one, which
  makes `--measure` a size comparison; it now says so instead of presenting one as the
  other. Requantizing from an intermediate produced files up to 38% slower on the A76 under
  an identical label, which is why the source has to be unquantized.
- `tools/verify_wheel.py` installs the built wheel and sdist into throwaway environments,
  runs the commands that need packaged data from a directory that is not the checkout, and
  asserts the matrix was loaded from site-packages. A new CI job runs it on Linux and
  Windows. A checkout has that CSV on disk whether or not the build carries it, so the
  whole suite can pass while every installed copy fails.
- Both the measured report and the unmeasured-silicon path now end with where to send the
  result.

### Changed, second pass
- The `llama-bench` not-found error leads with `brew`, `winget` and `conda-forge`, which
  llama.cpp documents, and with the prebuilt release archives, rather than with a cmake
  line. It also says that Ollama and LM Studio bundle llama.cpp without exposing
  `llama-bench`, which is the wrong assumption to leave someone holding.
- `inspect` on a directory or glob that matches nothing is an error rather than silence.


Three new commands, and the tool now answers the question people actually arrive with
rather than the one the roofline literature asks.

### Added
- `llama-roofline diagnose`. No flags. It looks for GGUF files in the usual places and in
  `$LLAMA_MODELS`, picks up to three that span the size range so the slope can be fitted,
  finds `llama-bench`, measures the memory ceiling, sweeps threads, and prints the answer
  in words: whether decode is bandwidth-bound, what this machine's ceiling is for that
  model in tokens per second, what thread count to use, and what running every thread
  costs. It also writes `diagnosis.md` for pasting into an issue.
- `llama-roofline advise`. Ranks GGUF quantization formats for the CPU core this machine
  has, from a shipped matrix of 98 measurements across an Arm Cortex-A76, an Intel Golden
  Cove P-core and an Intel Gracemont E-core, at two model scales, with Raspberry Pi PMIC
  energy and perplexity on the same artifacts. There is no single ranking, because the
  measurements do not support one: the fastest 4-bit format differs by core and by thread
  count. The report states its evidence level per core (measured here, same core design,
  extrapolated from a relative, or nothing), and on unmeasured silicon it refuses to guess
  and shows how to measure instead. `--measure` builds the ranking from your own files.
- `llama-roofline inspect`. Prints a GGUF file's per-tensor type map, the bytes decode
  reads per token, and how much of that is the output head. A format label names a recipe,
  not a type: in a canonical Qwen2.5-0.5B Q4_K_M file only 12% of the repeating-layer
  bytes are Q4_K, because the embedding dimension does not divide by 256.
- The GGUF reader now parses the tensor table, not just the header key/value block, and
  computes bytes streamed per token. Checked against an external result: given the eight
  canonical artifacts from a published format study, it returns every streamed size the
  study reported to within 0.2%.

### Changed
- `diagnose` counts bytes per token as the tensors decode actually reads, excluding the
  token embedding because generation looks up one row of it rather than streaming it. That
  is 18 to 22% below the file size on a 0.5B model and the gap is format-dependent.
  `run` keeps the resident-size convention so that the published fits this tool reproduces
  keep reproducing; `diagnose --file-bytes` makes the two agree.
- The package is on PyPI, so the README installs from there instead of from git.

### Fixed
- `--ctx` passed `-c` to llama-bench, which current builds reject outright, so any run
  using the flag failed and measured nothing. llama-bench has no context-size option; it
  sizes the context from `-p`, `-n` and `-d`. Replaced by `--depth`, which maps to `-d`
  (generate with N tokens already in the KV cache). Regression tests assert the built
  argv never contains `-c`, `--ctx` or `--n-ctx`.
- A run in which every model failed to load exited 0, so a script checking the exit code
  would have read it as a success. It now returns 1, while still writing the output.
- llama.cpp's backend-loading banners buried the actual error in a multi-line stderr dump
  per failure. `summarize_stderr()` drops the banners and collapses the real error to one
  line.

### Documented
- Measured where the weights-only roofline stops holding, instead of asserting it. With
  the KV cache pre-filled, decode falls to 39% (0.5B) and 45% (1.5B) of its empty-cache
  throughput at 8k context, and the 0.5B model reaches 13% at 32k. The falloff is far
  steeper than the extra KV bytes alone predict, so it is attention work over the cache,
  not just more streaming. See METHOD.md section 7 and the README's Long context section.
- METHOD.md section 4 now explains the two byte conventions and why they differ, and a new
  section 8 documents where the quantization rankings come from and the four rules that
  keep them honest.

### Tests
- 149 tests, up from 77. The new ones cover the tensor-table parser against synthetic GGUF
  files, silicon detection across nine CPU strings including the two that must NOT match
  (Core Ultra and pre-hybrid Intel), the advisor's logic, and the shipped matrix against
  the published table it came from. Reports are asserted to be plain ASCII.
- CI's no-dependency job now also proves `advise` and the GGUF reader work without numpy.

## [0.1.0] - 2026-08-07

First release.

### Added
- `llama-roofline run`: benchmarks GGUF models through `llama-bench` across a thread
  sweep, fits the decode roofline `tok/s = BW_eff / model_bytes`, and writes a report
  card, a Markdown summary, a roofline figure, and machine-readable JSON.
- `llama-roofline membw`: portable memory-bandwidth ceiling microbenchmark
  (`sum`, `max` and `dot` read kernels plus `copy` for context), reporting the best
  read-only result as a measured lower bound on peak.
- `llama-roofline report`: re-render a report or figure from a saved results JSON.
- Minimal GGUF metadata reader: architecture, quantization type, context length, and
  mixture-of-experts detection. MoE models are flagged and excluded from the fit,
  because only the active experts are read per token.
- Multi-part (sharded) GGUF handling: shards collapse to one model and the size sums
  across every part.
- Thread-knee detection, prefill-versus-decode scaling analysis (reported as the median
  across models, so one thrashing single-thread run cannot become the headline), and a
  size and quantization frontier.
- Stability check on the bandwidth measurement: every repetition is recorded and the
  median compared against the best. Best-of-N cannot detect constant contention, so a busy
  machine would otherwise report a silently-too-low ceiling and silently-too-high
  utilisation. Disagreement above 25% is reported as an unstable measurement.
- `llama-roofline report --reanalyze` re-runs the analysis over stored measurements with
  the current version of the tool, leaving the raw llama-bench numbers untouched.
- Honest reporting when measured decode bandwidth exceeds the microbenchmark ceiling:
  the ceiling is a lower bound, so the utilisation figures are reported as lower bounds
  rather than as an impossible ">100% of peak".
- Sample results from an Intel i7-12700H (DDR5) and a Raspberry Pi 5 (LPDDR4X).

### Known limitations
- CPU inference is the primary target. GPU offload is detected and warned about, but the
  ceiling measured is system RAM, not VRAM.
- Bytes-per-token is approximated by the model's resident size. This is exact for a dense
  transformer at short context and an overestimate once the KV cache grows large.
- Energy and DVFS analysis is not included in this release.
