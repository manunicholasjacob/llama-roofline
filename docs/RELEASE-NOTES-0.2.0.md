# llama-roofline 0.2.0

Draft. Every public word here is yours to rewrite before it goes anywhere.

---

v0.1.0 could tell you whether your decode was memory-bandwidth-bound, if you already knew
what a roofline was, had at least two GGUF files, and were willing to pass a glob. This
release is about the people who arrive with a different question: my tokens per second
feel low, what do I change?

```
pip install llama-roofline
llama-roofline diagnose
```

No flags. It finds your models, finds `llama-bench`, measures what your memory actually
sustains, sweeps thread counts, and tells you in words which knob moves your number.

On the laptop I test on, running a 3B at every one of its 20 threads costs 25% of the
decode throughput versus running it at 8. That is the sort of thing this now says out
loud, rather than leaving in a table for you to notice.

## Which quantization should you run

The second command is the one I think is actually new.

`llama-roofline advise` reads the core you are on and gives that core's format ranking. It
does not give a global one, because there is no global one to give. Q4_0 leads Golden Cove
at two threads. IQ4_NL leads Gracemont at four, on the same laptop, same memory, same
binary. On a Raspberry Pi 5's Cortex-A76 the whole 4-bit class converges to within 4% and
the choice comes down to energy and quality instead.

Q4_K_M, which almost every model card recommends, sits below the streaming envelope on all
three cores at 0.5B: 16% on the A76, 17% on Golden Cove and 30% on Gracemont, each at that
core's fastest thread setting. On the Pi that is 206 mJ per token against IQ4_XS's 143, for
0.58 perplexity points.

Read that carefully, because I nearly published a stronger version of it than the data
supports. A good part of that deficit at 0.5B is not the format. It is that the Q4_K_M file
at this size is 12% Q4_K and 70% Q5_0, for the reason in the next section, and it streams
374 MiB per token where the other 4-bit files stream about 330. At 1.5B, where the
substitution mostly stops, the gap narrows and reverses on two of the three cores. So the
honest statement is that the ranking is core-specific, that it is confounded at small model
sizes, and that `advise` says so where it prints such a table.

The rankings come from a controlled study of eight formats built from one FP16 source,
measured on three microarchitectures with PMIC energy and perplexity on the same
artifacts. They ship as a CSV with the measurement each row came from in a `source`
column, so you can check any number I quote.

On silicon nobody measured, the tool says so and shows you how to measure it, rather than
handing you a laptop's ranking for a phone:

```
llama-roofline advise --measure --models ~/models/qwen0.5b-*.gguf
```

## What is in the file you downloaded

`llama-roofline inspect` prints the per-tensor type map. This is the part that surprised me
most and it is the reason the other numbers here need reading carefully.

I built eight formats of Qwen2.5-0.5B from one FP16 source. Three of them contain the format
they are named after. Percentages are of repeating-layer bytes:

```
Q4_0      100% Q4_0        4.50 bits/weight
Q8_0      100% Q8_0        8.50
IQ4_NL     95% IQ4_NL      4.55
IQ4_XS     24% IQ4_XS      4.48   (70% is IQ4_NL)
Q6_K       24% Q6_K        7.93   (76% is Q8_0)
Q4_K_M     12% Q4_K        5.52   (70% is Q5_0)
Q3_K_M      0% Q3_K        4.57
Q2_K        0% Q2_K        4.20
```

K-quants operate on blocks of 256 and this model's embedding dimension is 896, so
llama-quantize substitutes a type that fits, per tensor, with a warning per tensor and no
summary. At 1.5B the dimension is wide enough that it mostly stops: Q4_K_M goes to 79%
on-label, Q3_K_M to 58%, IQ4_XS to 95%.

Small models are where people compare quantization formats, because it is cheap. Small
models are where the file is least likely to be what it says. `advise` now says so where it
prints a table at that scale, because a ranking between those files is partly a ranking of
what the quantizer substituted.

Provenance does the same thing from a different direction. The same eight targets
requantized from a Q8_0 intermediate instead of FP16 carry identical labels and identical
nominal bit widths, and decoded up to 38% slower on the A76.

## Bytes per token, and why two commands disagree

`diagnose` now counts what decode actually reads: the repeating layers plus the output
head, with the token embedding excluded, because generating a token looks up one row of it
rather than streaming the matrix. On a 0.5B model that is 18 to 22% below the file size,
and it varies by format.

`run` keeps the old convention, the resident size, because that is what the published fits
this tool reproduces were computed with. `diagnose --file-bytes` makes them agree.

The tensor parser is checked against something outside itself: given eight artifacts whose
streamed sizes a published study reported independently, it returns all eight to within
0.2%.

## Everything else

172 tests, up from 77. Reports are asserted to be plain ASCII. The no-dependency CI job now
also proves the advisor and the GGUF reader run without numpy.

Full detail in [CHANGELOG.md](../CHANGELOG.md).

## What I would most like

Results from hardware I do not own. The advisor knows three cores, which is a narrow base
for a question this broad, and the shape of the answer changes with the kernel coverage a
runtime happens to have for your silicon. If `advise --measure` gives you an ordering that
does not match anything above, that is the interesting case, and the results-gallery issue
template exists for it.
