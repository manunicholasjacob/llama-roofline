# Show HN draft

**Order of operations, per the decision of record:** CNX-Software first, then this. A Show
HN thread is hours of live methodology defence and it goes better once a write-up exists to
point at.

Post on a day you can sit in the thread. Post after 0.2.0 is on PyPI.

---

**Title:** Show HN: Most of my quantized model files were not the format their name claimed

**URL:** https://github.com/manunicholasjacob/llama-roofline

**First comment, posted immediately after submitting:**

I built eight quantizations of Qwen2.5-0.5B from one FP16 source, intending to compare
decode speed across formats. Before running anything I parsed the tensor tables to work out
how many bytes each file streams per token, and the parse said something I did not expect:
most of the files were mostly not the format they were named after.

Measured over the repeating layers, which is where the substitution happens and where
decode spends its bytes:

    Q4_0      100% Q4_0        4.50 bits/weight
    Q8_0      100% Q8_0        8.50
    IQ4_NL     95% IQ4_NL      4.55
    IQ4_XS     24% IQ4_XS      4.48   (70% is IQ4_NL)
    Q6_K       24% Q6_K        7.93   (76% is Q8_0)
    Q4_K_M     12% Q4_K        5.52   (70% is Q5_0)
    Q3_K_M      0% Q3_K        4.57   (64% Q4_0, 28% Q4_K)
    Q2_K        0% Q2_K        4.20   (75% Q4_0, 24% Q3_K)

This is not a bug and llama-quantize is not doing anything wrong. K-quants operate on
blocks of 256 elements, Qwen2.5-0.5B has an embedding dimension of 896, and 896 is not
divisible by 256, so the quantizer substitutes a type that fits, per tensor. It warns per
tensor and prints no summary, so what you get is a file that is mostly Q5_0 and a filename
that says Q4_K_M, and nothing in between tells you.

It resolves with scale. The same recipes on Qwen2.5-1.5B: Q4_K_M goes from 12% to 79%
on-label, Q3_K_M from 0% to 58%, IQ4_XS from 24% to 95%.

That is the part I think is worth the attention. Small models are where quantization gets
compared, because a 0.5B sweep is an afternoon and a 70B sweep is not, and small models are
where the file is least likely to be what it says. A format comparison at that scale is
partly a comparison of what the quantizer substituted, and I have not seen anyone report
the aggregate.

The tool is how I got there and it started somewhere narrower. Dense decode reads every
weight from memory once per token, so tok/s is bandwidth over bytes, and both sides are
measurable. `llama-roofline diagnose` takes no arguments: it finds your GGUFs and
llama-bench, measures the machine's sustainable read bandwidth, sweeps threads, and says
which knob moves your number. On my laptop it found that running a 3B at all 20 threads
costs a quarter of the decode throughput against running it at 8.

Three things I tried to get right, because they are where a measurement tool usually cheats.

The bandwidth ceiling is a lower bound and the report says so. numpy kernels over a thread
pool are portable and llama.cpp's hand-written SIMD can stream faster, so when decode
exceeds the ceiling it explains that the percentages are lower bounds rather than printing
an impossible number.

Bytes per token is the tensors decode actually reads. The embedding is a row lookup rather
than a stream, which is 18 to 22% of these files. The parser that computes it was checked
against an outside result: given eight artifacts whose streamed sizes were reported
independently, it returns all eight within 0.2%. That check runs every time the shipped data
is rebuilt and refuses to build if it fails.

`advise` refuses to guess. Three CPU cores have been measured, which is a narrow base for
"which quant should I run", so it states its evidence level per core and on unmeasured
silicon prints the commands to measure rather than handing you a laptop's ranking for a
phone. It also now warns, on its own data, that a 0.5B ranking is partly a ranking of
substitutions.

    pip install llama-roofline
    llama-roofline inspect ~/models/whatever.gguf

`inspect` is a header read, about a second, touches nothing.

Limits: one model family, two sizes, files I built rather than downloaded. Quantizers using
an importance matrix produce a different artifact and I have not measured those.

MIT. The analysis, the advisor and the GGUF reader are pure standard library. Every
throughput number comes from llama.cpp, which does the hard part.

---

## Notes

- The title claims something specific and checkable, which is the right kind of thing to
  defend in an HN thread.
- Someone will say this is known. Partly true: the divisibility rule is documented behaviour
  and people have hit it individually. The aggregate is not published anywhere I can find,
  and 0% on a file named Q3_K_M is the number that makes it concrete.
- Someone will say the substitution is correct and the tool should not imply otherwise.
  Agree immediately. The finding is about names and comparisons, not about llama-quantize.
- Expect the numpy ceiling objection. Fair, documented, and a better ceiling is a welcome
  contribution.
- Do not let it drift into "i-quants beat k-quants on CPU". That claim does not survive at
  1.5B and defending it would cost the thread.
