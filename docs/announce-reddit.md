# r/LocalLLaMA draft

Post this yourself, from your own account, after you have run the tool and read the
output. Do not post it as written if any number in it is not one you personally measured.

---

**Title:** I built a tool that tells you if your llama.cpp decode is memory-bandwidth-bound, and what to actually change

**Body:**

Everyone here knows tok/s drops when the model gets bigger. The reason is that dense
decode reads *every weight from RAM for every token*, so your generation speed is basically

    tok/s = (memory bandwidth you actually get) / (model bytes)

I got tired of guessing which side of that I was on, so I wrote a small CLI that measures
both. It benchmarks your own GGUFs through `llama-bench`, measures your machine's real
sustainable memory read bandwidth, fits the roofline, and then tells you in plain English
what to change.

```
pip install git+https://github.com/manunicholasjacob/llama-roofline
llama-roofline run --models ~/models/*.gguf
```

Here is what it printed on my laptop:

```
  Machine   : 12th Gen Intel(R) Core(TM) i7-12700H
              20 logical / 14 physical cores, 34.0 GB RAM
  Memory    : 53.9 GB/s sustained read (measured)

  IS YOUR DECODE MEMORY-BOUND?
------------------------------------------------------------------------
  YES -- your decode is memory-bandwidth-bound.
  Token generation is running at 72% of the memory bandwidth this machine
  can actually sustain.

  THE ROOFLINE
------------------------------------------------------------------------
    decode tok/s  =  37.65 GB/s  /  model bytes
    fitted across 7 models, R^2 = 0.9874

  YOUR MODELS
------------------------------------------------------------------------
  model                     quant         size    decode   prefill  thr    GB/s  %ceil
  ------------------------------------------------------------------------------------
  qwen0.5b-q2k              Q2_K        333 MB    113.3t      517t    8    37.7    70%
  qwen0.5b-q4km             Q4_K_M      392 MB     89.3t      351t    8    35.0    65%
  qwen0.5b-q8               Q8_0        525 MB     78.6t      294t   14    41.3    76%
  llama1b-q4km              Q4_K_M      800 MB     48.2t      245t   14    38.6    72%
  qwen1.5b-q4km             Q4_K_M      980 MB     39.3t      179t   14    38.5    71%
  qwen3b-q4km               Q4_K_M     1.92 GB     21.0t       87t    8    40.3    75%
  qwen7b-q4km               Q4_K_M     4.68 GB     11.2t       42t   20    52.2    97%
```

R² of 0.987 across a 14x range of model sizes. One number, the byte count, predicts my
generation speed well enough that I can work out what a model will run at before I
download it.

Things it tells you that a raw tok/s number does not:

- **Are you at the wall?** If you are at 85% of your memory ceiling, a faster CPU does
  nothing for you. If you are at 30%, something is wrong, and it lists what to check.
- **Your actual thread knee, separately for prefill and decode.** These are different and
  that is why thread advice here is always contradictory. Prefill is compute-bound and
  wants all your cores. Decode is memory-bound and stops improving early. On my machine
  going from 8 threads to 20 cost me 63% of my decode throughput on a 0.5B model.
- **What a smaller quant will actually buy you**, measured on your own models rather than
  guessed.
- **When the model does not apply.** MoE models only read the routed experts per token, so
  they sit off the dense roofline. It reads the GGUF header, detects them, and excludes
  them from the fit instead of quietly bending the line.

Honest limitations, up front:

- The bandwidth ceiling it measures is a **lower bound**. It uses numpy kernels for
  portability, and llama.cpp's hand-written SIMD can stream faster than numpy. If your
  decode comes out above the ceiling, the tool says the percentages are lower bounds
  rather than printing ">100% of peak". Pass `--peak-bw` if you have a real STREAM number.
- CPU inference is the target. GPU offload is detected and warned about, but the ceiling
  measured is system RAM, not VRAM.
- Bytes-per-token is the resident model size, so this is a short-context result. Long
  context adds KV traffic the model does not account for.
- It measures throughput only. It says nothing about output quality.

MIT licensed, pure standard library at its core (numpy and matplotlib are optional),
Linux/macOS/Windows. All the actual throughput numbers come from llama.cpp, which does the
hard part.

Repo: https://github.com/manunicholasjacob/llama-roofline

**What I would really like:** run it on hardware I do not have and post your report card.
Apple Silicon with unified memory, Ryzen with dual-channel DDR5, Snapdragon X, Jetson, old
Xeons. There is a results gallery in the repo. I am most interested in results that
*disagree* with the model, because that is where I learn something.

---

## Notes for posting

- Post on a weekday morning, US time.
- Lead with the report card, not the repo link. The output is the pitch.
- Answer every comment for the first 24 hours. That is where adoption happens.
- "My numbers look different" is the best possible reply. Ask them to open an issue.
- If someone finds a bug, fix it and say so in the thread.
