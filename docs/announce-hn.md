# Show HN draft

**Order of operations, per the decision of record:** CNX-Software goes first, then this.
The media criterion needs a third party writing about the work, and a Show HN thread is
hours of live methodology defence that would be better spent once a write-up exists to
point at.

Post on a day you can sit in the thread. A Show HN with an absent author dies, and this
one will draw specific technical objections that deserve specific answers.

Post after 0.2.0 is on PyPI so the install line is real.

---

**Title:** Show HN: Which quantization to run is a property of your CPU, not the model

**URL:** https://github.com/manunicholasjacob/llama-roofline

**First comment, posted immediately after submitting:**

Every GGUF model card carries roughly the same guidance for choosing a file, and the line
about CPUs says i-quants will be slower than the equivalent k-quant. I repeated that advice
for months. Then I built eight quantizations from one FP16 source and measured them on
three CPU cores, and IQ4_NL beat Q4_K_M on all three: by 33% on a Raspberry Pi 5's
Cortex-A76, 17% on an i7-12700H's performance cores, and 48% on the same laptop's
efficiency cores. On the Pi it also used 29% less energy per token.

The reason turned out to be more interesting than the speed. At 0.5B the embedding
dimension is 896, which does not divide by 256, so llama-quantize substitutes a different
type per tensor without saying so. In the Q4_K_M file, 12% of the repeating-layer bytes are
actually Q4_K and 70% are Q5_0. It streams 374 MiB per token against about 330 for the
other 4-bit files. The label describes a recipe, not the contents.

Two things follow that I did not expect.

The ranking is not a property of the format. Q4_0 leads the performance cores of that
laptop, IQ4_NL leads its efficiency cores, and the A76 converges the whole 4-bit class to
within 4%. Same binary, same DRAM controller, same files. It tracks which formats
llama.cpp has architecture-tuned repacked kernels for on that core, which is visible in
`llama-bench -v` load logs and which nobody publishes.

The ranking is not a property of the filename either. Requantizing the same eight targets
from a Q8_0 intermediate rather than FP16 yields files with identical labels and identical
nominal bit widths that decoded up to 38% slower on the A76. Any benchmark that does not
pin artifact provenance is measuring something unnamed.

The tool is how I got there. It started as one question, whether a llama.cpp decode is
memory-bandwidth-bound, which is answerable because dense decode reads every weight per
token, so tok/s is bandwidth over bytes and both sides are measurable. `diagnose` measures
the machine's sustainable read bandwidth, benchmarks your own GGUFs through `llama-bench`
across a thread sweep, fits the roofline, and says which knob moves your number. On my
laptop it found that running a 3B at all 20 threads costs a quarter of the decode
throughput against running it at 8.

Three things I tried to get right, because they are where a measurement tool usually
cheats:

The bandwidth ceiling is a lower bound and the report says so. It uses numpy kernels over a
thread pool for portability and llama.cpp's hand-written SIMD can genuinely stream faster,
so when decode exceeds the ceiling it explains that the percentages are lower bounds
instead of printing an impossible number. `--peak-bw` takes a real STREAM figure.

Bytes per token is the tensors decode actually reads, not the file size. The token
embedding is a row lookup rather than a stream, which is 18 to 22% of these files. The
parser that computes this was checked against an outside result: given eight artifacts
whose streamed sizes were reported independently, it returns all eight within 0.2%.

`advise` refuses to guess. Three cores is a narrow base for a question this broad, so it
states its evidence level per core, and on silicon nobody has measured it says so and
prints the commands to measure it rather than handing you a laptop's ranking for a phone.

MIT. The analysis core, the advisor and the GGUF reader are pure standard library; numpy
and matplotlib are only needed to measure the ceiling and draw the figure, and CI proves it
runs without either. Every throughput number comes from llama.cpp, which does the hard part.

What I would most like from here is a fourth core. Apple Silicon especially, where unified
memory changes the whole balance and where I have no access.

---

## Notes

- HN reads titles literally. The title claims something specific and falsifiable, which is
  the right kind of claim to defend in a thread.
- Expect the numpy ceiling objection. It is fair, it is already documented, and the answer
  is "yes, here is exactly why, here is how to override it, and a better ceiling is a
  welcome contribution".
- Expect "0.5B is a toy". The 1.5B rows are the answer, and they narrow the gap and flip
  Gracemont, which is worth conceding before someone finds it.
- Expect somebody who knows llama.cpp internals to have a better explanation of the repack
  coverage than mine. That is the best possible outcome and should be treated as one.
- Do not argue with anyone about whether the model cards are wrong. The claim is what three
  CPUs did, not what all CPUs do.
