# Show HN draft

Post at least a day apart from the Reddit thread so you can be present for both.

---

**Title:** Show HN: Llama-roofline, find out if your local LLM is memory-bandwidth-bound

**URL:** https://github.com/manunicholasjacob/llama-roofline

**First comment (post this immediately after submitting):**

A dense transformer generating one token at a time reads every weight in the model from
memory, once, per token. Nothing is reused. So decode throughput is not a compute problem,
it is a bandwidth problem:

    tok/s = BW_eff / model_bytes

I kept wanting to know where a given machine actually sat on that line, so I wrote a CLI
that measures both sides. It measures the machine's sustainable memory read bandwidth,
benchmarks your own GGUF models through llama.cpp's `llama-bench` across a thread sweep,
fits `tok/s = BW * (1/bytes)` through the origin, and prints a report card.

The interesting part is not the fit, it is what falls out of it. Prefill and decode sit on
opposite sides of the roofline: prompt processing is a matrix-matrix product and scales
with cores, generation is memory-bound and stops improving early. On my laptop prefill kept
scaling to 14 threads while decode peaked at 8 and lost 63% of its throughput at 20. That
is one setting, and it looks free.

Two things I tried hard to get right:

The bandwidth ceiling is a **lower bound** and the tool says so. It uses numpy kernels over
a thread pool for portability, and llama.cpp's hand-written SIMD kernels can genuinely
stream faster. When measured decode exceeds the measured ceiling, the report explains that
the percentages are lower bounds instead of printing ">100% of peak". You can supply a real
STREAM number with `--peak-bw`.

Mixture-of-experts models break the bytes-per-token assumption, because only the routed
experts are read. Rather than let them bend the fit, it parses the GGUF header, detects
`expert_count`, flags them, and excludes them.

It reproduces the published fit from the study it came out of, exactly, when handed that
study's raw data, which is the closest thing to a correctness test I could construct for a
measurement tool. That conversion script is in the repo.

Core is pure standard library; numpy and matplotlib are optional extras and CI proves the
tool runs without either. MIT. All throughput numbers come from llama.cpp, which does the
hard part.

I would especially like results from hardware I do not own, particularly Apple Silicon with
unified memory. There is a gallery in the repo and results that contradict the model are
the most useful ones.

---

## Notes

- HN reads titles literally. Do not oversell.
- Be in the thread. A Show HN with an absent author dies.
- Expect pushback on the numpy ceiling. It is a fair criticism, the limitation is already
  documented, and "yes, and here is exactly why, and here is how to override it" is the
  right answer. If someone contributes a better ceiling, that is a win.
