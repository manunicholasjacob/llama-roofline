# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `SECURITY.md`: what the tool reads and runs, and how to report a problem privately.

### Changed
- Remote `inspect` retries a dropped connection four times, waiting 0.5, 1, 2 and 4 s,
  instead of twice; a reset partway through a response body is now retried as well,
  rather than ending the run with a traceback.
- README: the substitution table notes that the fallback type depends on the llama.cpp
  build (Qwen's own Q2_K and Q3_K_M files carry IQ4_NL where the table's build wrote
  Q4_0, at the same bits per weight).
- Package metadata gives the licence as the SPDX expression `MIT` with `license-files`
  (PEP 639) instead of the deprecated table and classifier, which setuptools warned about
  on every build; the build now needs setuptools 77 or later.

### Fixed
- A GGUF header whose array claims an impossible element count (for example 2**62)
  now ends in a `GGUFError` ("implausible array length") instead of an uncaught
  `ValueError` from `describe()` and `read_tensors()`, or an `OverflowError` in the
  browser inspector. An array skip that runs past the end of the file is now reported
  as truncation even when it is the last thing in the header; before, the parse
  succeeded with a data section beyond the end of the file. Local files, `BytesIO`
  and remote `inspect` all take this path.

## [0.2.3] - 2026-10-13

### Added
- `inspect` reads models it does not have locally: `hf:OWNER/REPO[@REV]/FILE.gguf`, a
  Hugging Face `blob/` or `resolve/` link, or any https URL whose server honours HTTP
  range requests. Only the header is fetched (8.4 MB of the 491 MB Qwen2.5-0.5B Q4_K_M
  file), in 4 MB blocks with two retries. `HF_TOKEN` is used for gated repositories and is
  sent to huggingface.co only. A missing file prints the repository's file list URL; a
  401 or 403 says the repository is gated; a server that ignores ranges is refused rather
  than downloaded.
- `inspect --json` prints the summary as data, and `inspect --peak-bw GBS` adds the decode
  ceiling the file implies at that bandwidth.
- GGUF types: NVFP4, Q1_0 and Q2_0 (current llama.cpp), and the Arm-repacked
  Q4_0_4_4, Q4_0_4_8 and Q4_0_8_8 that late-2024 builds wrote and that are still on the
  Hub. Each layout except Q2_0 was checked against the tensor offsets of a real file.
- A tensor of a type newer than this release is sized from the tensor offsets, exact to
  within the alignment padding, and the report says so, instead of the per-token figure
  going missing.
- `llama_roofline.inspection`: the summary and text report of `inspect`, standard
  library only, so the browser inspector runs the same code as the CLI.
- A documentation site, https://manunicholasjacob.com/llama-roofline/, built from the
  README, `docs/`, the results gallery, CONTRIBUTING and this changelog by
  `tools/build_site.py` and published by a Pages workflow.
- GGUF Inspector, the `inspect` report in a web page
  (https://manunicholasjacob.com/llama-roofline/inspector.html): a local file's header is
  read in the browser, a Hub file's through range requests, and the page runs this
  package's `gguf.py` and `inspection.py` in Pyodide. `?model=hf:...` links share a
  result. `tools/build_site.py --space DIR` writes it as a static Hugging Face Space.
- Issue templates for a GGUF type it cannot size and for feature requests, a pull
  request template, and a CONTRIBUTING section on adding a GGUF type.

### Changed
- Errors for files that are not GGUF say what the file is: safetensors, a zip
  checkpoint, pre-GGUF GGML, a saved Hugging Face web page, a Git LFS pointer, an empty
  file, GGUF version 1, or a version newer than the tool. A file cut off inside its
  header says it looks like an incomplete download.

### Fixed
- `inspect` on the first part of a split model read that part only. It now reads every
  part, and says which part is missing if one is.

## [0.2.2] - 2026-10-12

### Added
- Ollama model names are accepted wherever a GGUF path is, as `ollama:NAME[:TAG]`
  (for example `llama-roofline inspect ollama:qwen2.5:7b`). Ollama stores each model as a
  plain GGUF blob named by its digest; the name is resolved through the local manifest
  (`$OLLAMA_MODELS`, else `~/.ollama/models`) and `inspect` prints the name rather than
  the digest. An unknown name is reported by name, with the directory searched,
  and exit code 2.
- README: a quickstart that needs no llama.cpp build (`inspect` reads only the file
  header), pipx and uv install lines, and a BibTeX entry. PyPI page: project URLs,
  keywords and a classifier.

### Changed
- `advise` ships the Raspberry Pi 5 (Cortex-A76) measurements only. On any other CPU,
  including Intel hybrid laptops, it says there is no shipped measurement for that core
  and points at `advise --measure`, which benchmarks the user's own files.
- The `advise` report for the shipped measurements prints speed, streamed bandwidth and
  perplexity. The streaming-envelope column, the energy column and the bullets built on
  them are no longer printed or written to `--json`; the data file is unchanged.
- `advise --measure` points at the one core the shipped rankings cover.
- `advise --plan` explains its refusal of an already quantized source with llama.cpp's
  own `--allow-requantize` warning.
- `inspect`: when the labelled type is the majority of a file, the note says "The rest
  is ..." instead of "Most of it is ...".
- Report, README and documentation wording tightened to what the shipped measurements
  show.

### Fixed
- On Windows 11 24H2 and later, which no longer ship `wmic`, the CPU was reported as
  "Intel64 Family 6 Model ..." and the physical core count was missing, so `advise` could
  not identify the core and the default thread sweep was built from logical cores only
  (1, 10, 20 instead of 1, 7, 14, 20 on an i7-12700H). The brand string now comes from the
  registry and the core count from CIM.
- `advise` on a detected core with no rows in the matrix raised `IndexError`; it now
  falls through to the measure-it-yourself report.

### Removed
- Launch-planning notes and the 0.2.0 release-notes essay under `docs/`. The changelog is
  the record of changes.

## [0.2.1] - 2026-10-03

A patch release: no change to any measurement, fit or advice. It makes bad input fail
at once with a usage message, tests and declares Python 3.14, and corrects the citation
metadata.

### Fixed
- Numeric options are checked when the command line is parsed. `--threads 1,x` raised a
  `ValueError` traceback; `--threads 0` was dropped without a word; `--n-prompt 0` was
  accepted and only failed inside the benchmark loop, after the bandwidth measurement had
  already run. Worst, `membw --membw-reps 0` printed a ceiling of 0.00 GB/s with no kernel
  and no thread count and told the user to pass `--peak-bw 0.00` to `run`. Thread lists
  (`--threads`, `--threads-sweep`) must now be comma-separated whole numbers of at least 1,
  and `--reps`, `--n-prompt`, `--n-gen`, `--membw-mb`, `--membw-reps` and
  `advise --threads` must be at least 1. Each is a one-line usage error with exit code 2.
  `membw.measure()` also refuses `reps < 1`, for callers using it as a library.
- `llama-roofline report` on a JSON file that is not a results file failed with an
  `AttributeError` traceback when the file held a list, and printed a blank report card
  with exit code 0 when it held an empty object. It now says the file is not a
  llama-roofline results file and exits 2.
- `silicon.detect(cpu_override=...)` answered from the host instead of from the name it was
  given. It replaced the CPU string but still picked its branch from `platform.machine()`, so
  an Intel override on an Apple silicon runner fell into the arm64 path and went looking for
  `/proc/cpuinfo`. Three tests failed on `macos-latest` and three more passed there for the
  wrong reason, asserting that a CPU yields no cores while never reaching the x86 logic they
  exist to check. An override now settles the ISA from the name and reads nothing off the
  running system. No user-visible change: `cpu_override` is not exposed on the CLI.
- The README and CITATION.cff gave the v0.1.0 version DOI as the one to cite for the
  v0.2.0 release. They now give `10.5281/zenodo.22131998`, the DOI Zenodo minted for
  v0.2.0. The concept DOI is unchanged.

### Changed
- Python 3.14 is tested in CI on Linux, macOS and Windows and declared in the package
  classifiers. The suite also passes on 3.14 with `DeprecationWarning` raised as an error.
  CI moves to `actions/checkout@v5` and `actions/setup-python@v6`.
- `tools/verify_wheel.py` reads the expected version from `pyproject.toml` instead of
  hard-coding it, so the installed-package job checks the version being released rather
  than the previous one.
- Releases are uploaded to PyPI by `.github/workflows/publish.yml` through Trusted
  Publishing, with no API token anywhere. It refuses to publish when the release tag and
  `pyproject.toml` disagree.

## [0.2.0] - 2026-08-23

### Added, second pass
- `llama-roofline advise --plan <model-f16.gguf>` prints the `llama-quantize` commands
  that build a set worth comparing, and refuses when pointed at a file that is already
  quantized. Most people have one file per model rather than several formats of one, which
  makes `--measure` a size comparison; it now says so instead of presenting one as the
  other. The source has to be unquantized so that every file in the set comes from the same
  weights.
- `tools/verify_wheel.py` installs the built wheel and sdist into throwaway environments,
  runs the commands that need packaged data from a directory that is not the checkout, and
  asserts the matrix was loaded from site-packages. A new CI job runs it on Linux and
  Windows. A checkout has that CSV on disk whether or not the build carries it, so the
  whole suite can pass while every installed copy fails.
- Both the measured report and the unmeasured-silicon path now end with where to send the
  result.

### Added, third pass
- `gguf.composition()` measures how much of a file is the type its name claims, over the
  repeating layers, plus the effective bits per weight. `inspect` leads with it. At 0.5B,
  of eight formats built from one FP16 source, three contain the type they are named after;
  Q3_K_M and Q2_K contain none of it; Q4_K_M is 12% Q4_K and stores 5.52 bits per weight
  against a nominal 4.5. At 1.5B the same recipes reach 58 to 100% on-label.
- The shipped matrix gains `on_label_pct` and `bits_per_weight` per row, and `advise` warns
  where it prints a table that effect confounds.
- The 1.5B rows are normalised by streamed bytes parsed from the artifacts rather than by
  file size, so the two scales are finally comparable. The builder cross-checks its own
  parse against the study's published 0.5B streamed sizes and refuses to build if they
  disagree by more than 1%.

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
  energy and perplexity on the same artifacts. The report states its evidence level per core (measured here, same core design,
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
- 172 tests, up from 77. The new ones cover the tensor-table parser against synthetic GGUF
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
