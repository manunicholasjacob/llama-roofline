# Is your llama.cpp memory-bound? Here is how to tell.

You bought more cores and your tokens per second did not move. You set `-t` to every
thread the machine has and it got *slower*. You dropped from Q8 to Q4 and got almost
exactly the speedup the file size predicted, which felt like a coincidence.

None of that is a coincidence. There is one equation behind all three, and you can measure
your side of it in about five minutes.

## The equation

```
decode tok/s  =  BW_eff / model_bytes
```

That is it. Your token generation rate is the memory bandwidth your machine can actually
sustain, divided by the number of bytes in your model.

Here is why. When a dense transformer generates text one token at a time, the forward pass
for that token touches every weight in the model exactly once. There is no reuse to
exploit, no data to keep in cache across the computation, nothing to amortise. Every
weight travels from DRAM to the CPU, gets used for one multiply-accumulate, and is gone.

So the CPU is not really computing. It is waiting. The arithmetic is trivial next to the
traffic, and the thing that sets your speed is how fast bytes arrive.

## Prefill is the opposite, and that is the confusing part

Processing your prompt is a different computation with a different shape. A prompt of 512
tokens is a matrix-matrix product: the same weights get reused across all 512 positions.
Arithmetic intensity scales with the prompt length, and prefill lands squarely in the
compute-bound region.

This is why threading advice for local LLMs is so contradictory. Both camps are right.
Prefill wants every core you have. Decode saturates at a low thread count and then goes
backwards, because more threads waiting on the same memory bus does not make the memory
faster; it just adds contention.

On my laptop, prefill kept improving all the way out to 14 threads. Decode peaked at 8 and
then fell off a cliff: at 20 threads a 0.5B model dropped from 89 tok/s to 33 tok/s. That
is a 63% loss from a setting that looks like it should be free.

## So I built a tool that measures it

[`llama-roofline`](https://github.com/manunicholasjacob/llama-roofline) measures both sides
of the equation on your machine:

```bash
pip install git+https://github.com/manunicholasjacob/llama-roofline
llama-roofline run --models ~/models/*.gguf
```

It measures the memory bandwidth your machine actually sustains, benchmarks your own GGUF
models through llama.cpp's `llama-bench` across a thread sweep, fits the roofline, and
prints a report card in English rather than a table of numbers you have to interpret.

The output tells you four things a raw tok/s number cannot:

**Are you at the wall?** If decode is running at 85% of your memory ceiling, a faster CPU
will do nothing for you. Only fewer bytes or faster RAM will move that number. If you are
at 30%, something else is wrong, and the report lists what to check.

**How much will a smaller quant actually buy?** Not a guess. The fit predicts it from the
byte count, and the report shows the ratio measured on your own models.

**How many threads should you use?** Separately for prefill and decode, because the answer
is different.

**Where does the model stop applying?** Mixture-of-experts models only read the routed
experts per token, so they sit off the dense roofline. The tool reads the GGUF header,
detects them, and excludes them from the fit rather than quietly bending the line.

Here is what it printed for me, across seven models from 0.5B to 7B on one laptop:

```
  IS YOUR DECODE MEMORY-BOUND?
------------------------------------------------------------------------
  YES -- your decode is memory-bandwidth-bound.

  Token generation is running at 72% of the memory bandwidth this machine
  can actually sustain.

  THE ROOFLINE
------------------------------------------------------------------------
    decode tok/s  =  37.65 GB/s  /  model bytes
    fitted across 7 models, R^2 = 0.9874
    that effective bandwidth is 70% of your 53.9 GB/s ceiling
```

An R² of 0.987 across a 14x range of model sizes is the part I find genuinely useful. It
means one number, the byte count, predicts my generation speed well enough to plan with. I
can work out what a model will run at before I download it.

## The Raspberry Pi and the laptop agree

I ran this on two machines 3.9x apart in measured memory ceiling and roughly 20x apart in
price: an Intel i7-12700H laptop with DDR5, and a 2 GB Raspberry Pi 5 with LPDDR4X.

On the Pi, across 7 models spanning three parameter counts and five quantization levels,
decode throughput fits `10.69 GB/s / model_bytes` with an R² of 0.980, sitting at 73% to
86% of the board's 13.98 GB/s memory ceiling.

On the laptop, across seven models from 0.5B to 7B, decode fits `37.65 GB/s / model_bytes`
with an R² of 0.987, at 65% to 97% of its measured 53.9 GB/s ceiling. The 7B model, the
one with the largest working set, runs closest to the wall at 97%.

The two fitted bandwidths differ by 3.5x. So does the memory.

Two very different machines, the same behaviour. That is what makes the roofline useful
rather than a curiosity: it is not a property of one device, so the prediction transfers.
If you know your machine's bandwidth and your model's size, you can estimate your tokens
per second before you download anything.

## The honest part

I would rather you trust the tool than be impressed by it, so here is what it does not do.

The bandwidth ceiling it measures is a **lower bound**. It uses numpy kernels over a thread
pool, because shipping a compiled STREAM benchmark to every platform is not practical, and
llama.cpp's hand-written SIMD kernels can genuinely stream faster than numpy can. When
measured decode bandwidth comes out above the ceiling, the report says the percentages are
lower bounds instead of printing an impossible ">100% of peak". If you have a real STREAM
number, pass `--peak-bw` and everything tightens.

Bytes-per-token is the model's resident size. That is exact for a dense transformer at
short context and an overestimate once the KV cache grows large, so treat the result as a
short-context measurement.

It measures throughput and nothing else. A Q2 model is faster than a Q8 model, and that
tells you exactly nothing about whether it is still worth using. Quality is a separate
question and this tool does not touch it.

And batching is the real escape hatch from this roofline: serving several sequences at
once amortises the weight read across all of them. This measures single-stream decode,
which is the worst case, and also the case most of us are actually in when running a model
locally.

## Try it, and send me your numbers

```bash
pip install git+https://github.com/manunicholasjacob/llama-roofline
llama-roofline run --models ~/models/*.gguf
```

It works on Linux, macOS and Windows, needs Python 3.9 or newer, and needs a llama.cpp
build you already have. The analysis core is pure standard library.

The most useful thing you can do is run it on hardware I do not own, and open an issue
with the report. Apple Silicon with its unified memory, Ryzen with dual-channel DDR5,
Snapdragon X, an old Xeon, a Jetson. I especially want the results that *disagree* with
the model, because those are where I learn something.

All the throughput numbers come from [llama.cpp](https://github.com/ggml-org/llama.cpp),
without which none of this exists. My tool drives it and does arithmetic on the output. The
hard part was already done.
