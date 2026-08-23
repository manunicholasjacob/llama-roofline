# r/LocalLLaMA, second attempt

Read `adoption-tracking/ADOPTION_PIPELINE.md` before posting this. The short version:

- The first version of this post went up on 17 August and AutoModerator removed it for
  insufficient subreddit karma. Nobody read it. The bot's message asks for a minimum of 5
  karma earned through comments and then explicitly invites the repost.
- His three comments there are worth roughly 3 karma. Two more useful ones clears it.
- After the karma gate a human sees it, and rule 3 bans primarily LLM-generated copy. The
  old draft has the shape that gets flagged: four consecutive bullets opening with a bolded
  phrase, an "Honest limitations, up front:" heading, a "**What I would really like:**"
  closer. That draft is kept at the bottom of this file for reference and should not go up
  as it stands.
- Post after 0.2.0 is on PyPI, so the install line is `pip install llama-roofline` rather
  than a git URL.

This draft leads with the result instead of the tool. Rewrite it in your own words before
posting, and do not post a number you have not personally re-run.

---

**Title:** I measured whether i-quants are actually slower than k-quants on CPU, and on
three cores they were not

**Body:**

Every GGUF model card I download says some version of this:

> These I-quants can also be used on CPU, but will be slower than their K-quant
> equivalent, so speed vs performance is a tradeoff you'll have to decide.

I had been repeating that to people. Then I built a set of quants from one FP16 source and
measured them, and on the three CPU cores I have access to, that is not what happened.

Qwen2.5-0.5B, one FP16 source, `llama-bench -p 128 -n 128`, five repetitions, each core at
its own best thread count:

```
                              IQ4_NL    Q4_K_M
Cortex-A76  (Pi 5),  t=2      34.2       25.7    tok/s
Golden Cove (P-core), t=6    105.5       89.9
Gracemont   (E-core), t=4     43.6       29.3
```

IQ4_NL beat Q4_K_M on all three. On the Pi it also used 29% less energy per token, from
the PMIC rails. It is worse on perplexity, 20.70 against 20.12, and that is a real trade,
but it is a different trade from the one the card describes.

My first thought was that I had built the files wrong. I had not, and working out why led
somewhere more interesting than the speed number.

At 0.5B the embedding dimension is 896, which is not divisible by 256. `llama-quantize`
falls back per tensor when a shape does not divide, silently, and in the Q4_K_M file only
12% of the repeating-layer bytes are actually Q4_K. Seventy percent are Q5_0. The file
streams 374 MiB per token where the 4-bit files stream about 330, so it is not really a
4-bit file at this size at all. None of that is visible from the filename.

Two other things fell out of the same runs:

The ranking is not the same on the two core types in one laptop. Q4_0 leads Golden Cove.
IQ4_NL leads Gracemont. Same binary, same DRAM, same files, opposite answer. On the A76 the
whole 4-bit class converges to within 4% and the choice becomes energy and quality instead.

Provenance moves the number more than I expected. The same eight targets requantized from
a Q8_0 intermediate instead of FP16 carry identical labels, identical nominal bit widths,
and decoded up to 38% slower on the A76. A benchmark that does not say where its files came
from is measuring something it has not named.

Caveats, and they matter: three cores, one model family, two sizes. At 1.5B the gap narrows
a lot and Gracemont flips to Q4_K_M by about 3%, so part of what I measured at 0.5B is the
divisibility fallback rather than the format. Perplexity is one corpus. Energy is one board
with uncalibrated rail sums, so treat it as a ratio.

The tool I used for this is `llama-roofline` and I wrote it. `llama-roofline advise` prints
the table for whichever core it detects and says plainly when it has no measurement for
your silicon, which is most silicon. `llama-roofline inspect model.gguf` prints the
per-tensor type map, which is the part I wish I had checked earlier.

```
pip install llama-roofline
llama-roofline advise
llama-roofline inspect ~/models/whatever.gguf
```

What I actually want is a fourth core. Apple Silicon, Ryzen, Snapdragon X, anything with a
different memory system. `llama-roofline advise --plan <model-f16.gguf>` prints the
llama-quantize commands to build a comparable set and `--measure` benchmarks them. If your
ordering disagrees with mine I would rather know.

---

## Notes for posting

- Do not post this the same hour as the two karma-building comments. Space it.
- The i-quant line is quoted from a live model card. Re-read the card before posting in
  case it has been edited, and quote whatever it says then.
- If someone points out the 1.5B reversal, agree immediately. It is in the post already and
  conceding a real limit is how this kind of thread goes well.
- Expect "did you use imatrix". Answer honestly: no, these are plain quantizations from
  FP16 with no importance matrix, and an imatrix i-quant is a different artifact again.
- Expect "your 0.5B is too small to matter". That is fair and the 1.5B row is the answer,
  along with saying that the divisibility fallback is a small-model effect.

---

## The removed draft, kept for reference only

Do not repost this. It is here so the difference is visible: it opens with the tool, four
bullets in a row start with a bolded phrase, and the limitations arrive under a heading
that announces its own honesty.

**Title:** I built a tool that tells you if your llama.cpp decode is memory-bandwidth-bound,
and what to actually change

The full text is in the reddit post at `/r/LocalLLaMA/comments/1vqzlto/`, which is still
visible to him while logged in.
