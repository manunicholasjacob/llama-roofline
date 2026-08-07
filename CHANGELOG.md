# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
