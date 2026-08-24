# r/LocalLLaMA, second attempt

Read `adoption-tracking/ADOPTION_PIPELINE.md` before posting. The short version:

- The first version went up on 17 August and AutoModerator removed it for insufficient
  subreddit karma. Nobody read it. The bot asks for a minimum of 5 karma earned through
  comments and then explicitly invites the repost. You are near 3.
- After the karma gate a human reads it, and rule 3 bans primarily LLM-generated copy.
- Post after 0.2.0 is on PyPI so `pip install llama-roofline` is true.

**The claim changed between drafts and you should know why.** The earlier draft led with
i-quants beating k-quants on CPU. Parsing the tensor maps at both model sizes showed that
most of that gap at 0.5B is llama-quantize substituting types, not the format, and the
comparison reverses on two cores at 1.5B. Leading with it would have been picked apart, and
the person doing the picking would have been right. What replaced it is stronger, because
it is a property of the files rather than a benchmark result, and anyone can check it in
one command.

Rewrite this in your own words. Do not post a number you have not personally re-run.

---

**Title:** I checked what is actually inside eight GGUF files. Only three contained the
format their name claims.

**Body:**

I quantized Qwen2.5-0.5B into eight formats from one FP16 source, then read the tensor
tables instead of trusting the filenames. Percentages are of repeating-layer bytes, which
is where decode spends its time:

```
Q4_0      100% Q4_0        4.50 bits/weight
Q8_0      100% Q8_0        8.50
IQ4_NL     95% IQ4_NL      4.55
IQ4_XS     24% IQ4_XS      4.48   (70% of it is IQ4_NL)
Q6_K       24% Q6_K        7.93   (76% is Q8_0)
Q4_K_M     12% Q4_K        5.52   (70% is Q5_0)
Q3_K_M      0% Q3_K        4.57   (64% Q4_0, 28% Q4_K)
Q2_K        0% Q2_K        4.20   (75% Q4_0, 24% Q3_K)
```

A file named Q3_K_M with no Q3_K tensors in it was not what I expected to find. Q4_K_M
storing 5.52 bits per weight when the name implies 4.5 was not either.

The cause is not a bug. `llama-quantize` substitutes a higher type per tensor when a shape
does not divide by the block size, and k-quants need 256. Qwen2.5-0.5B has an embedding
dimension of 896, which does not. The tool prints a warning per tensor and no summary, so
unless you are watching the scroll you get a file that is mostly something else and a name
that says otherwise.

It goes away with scale. Same recipes on Qwen2.5-1.5B:

```
              0.5B      1.5B
Q4_K_M        12%       79%
Q3_K_M         0%       58%
IQ4_XS        24%       95%
```

Which is the part I think matters here. Small models are where people benchmark
quantization, because it is cheap and fast, and small models are exactly where the file is
least likely to be what it says. If you have compared quants on a 0.5B or a 1B and drawn a
conclusion about the format, you may have measured the substitution.

Two more things fell out of the same files.

Bytes per token is not file size. Decode reads the repeating layers and the output head
every token and looks up one row of the embedding, so these files read 18 to 22% fewer
bytes than they weigh, and the gap is format-dependent. At 0.5B the output head alone is
about 40% of what moves per token.

Provenance changes the file under an unchanged name. The same eight targets requantized
from a Q8_0 intermediate instead of FP16 carry identical labels and identical nominal bit
widths, and decoded up to 38% slower on a Cortex-A76.

You can check any file you have:

```
pip install llama-roofline
llama-roofline inspect ~/models/whatever.gguf
```

It prints the per-tensor type map, how much of it is on-label, the effective bits per
weight, and the bytes decode actually reads. It is a read of the header, so it takes about
a second and touches nothing.

Caveats: one model family, two sizes, and this is a property of how these files were built
rather than a claim about every GGUF on Hugging Face. Files from a quantizer using an
importance matrix are a different artifact again and I have not measured those.

If you run it on something you downloaded rather than built, I would like to know what it
says. Especially if a file turns out to be exactly what it claims, because so far the ones
that were are the two simplest formats.

---

## Notes for posting

- The claim is about files, not about benchmarks, which is why it is hard to argue with.
  Keep it that way. Do not drift into "and therefore i-quants beat k-quants", because that
  part does not survive at 1.5B.
- Expect "this is known". Some people do know it. The reply is that the aggregate is not
  reported anywhere, and 0% on a Q3_K_M file is a number nobody has published.
- Expect "imatrix quants are different". Agree. They are, and you have not measured them.
- Expect somebody to point out that the substitution is deliberate and correct behaviour.
  It is. The finding is not that llama-quantize is broken, it is that the resulting file is
  not described by its name and that people compare files by name.
- If it goes well, the follow-up worth having is somebody running `inspect` on a
  well-known download and posting what it says.

---

## The removed draft

Do not repost it. It opened with the tool, four consecutive bullets started with a bolded
phrase, and the limitations arrived under a heading that announced its own honesty. It is
still visible to you at `/r/LocalLLaMA/comments/1vqzlto/`.
