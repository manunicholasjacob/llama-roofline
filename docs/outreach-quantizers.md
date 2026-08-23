# A note to the people whose model cards answer this question

Staged, not sent. Send it after the r/LocalLLaMA post exists, so there is something to
point at that is not just a repository.

## Who, and why them

Nearly everyone who downloads a GGUF gets it from one of a small number of quantizers, and
their model cards carry a "Which file should I choose?" section that is the community's
default answer to the question this tool exists for. `bartowski/Qwen_Qwen3.6-35B-A3B-GGUF`
alone is at 435,666 downloads, and the section is boilerplate across the cards, so the
readership is much larger than any one repo.

The last line of that section, read from the live card on 23 August 2026:

> These I-quants can also be used on CPU, but will be slower than their K-quant
> equivalent, so speed vs performance is a tradeoff you'll have to decide.

On three CPU cores, that is not what happened. That is worth telling him, carefully, with
the limits attached.

**Where to send it.** The Community tab of one of the model cards is the right venue rather
than a DM, because the correction belongs where the guidance is and because a public thread
lets other people check it. He is also reachable on r/LocalLLaMA. Pick one, not both.

**What not to do.** Do not open with the tool. Do not ask him to link anything. The message
is a measurement he might want; whether it changes his card is his call and saying so out
loud makes it easier for him to take.

---

## Draft

Subject, if it needs one: i-quants on CPU, three cores, and a Q4_K_M oddity at small sizes

---

Your model cards are where I send people who ask which file to download, so I wanted to
bring you a measurement rather than an opinion.

The "Which file should I choose?" section says i-quants can be used on CPU but will be
slower than the equivalent k-quant. I had been repeating that. Then I built eight
quantizations of Qwen2.5-0.5B from one FP16 source and benchmarked them on three CPU cores,
and IQ4_NL came out ahead of Q4_K_M on all three:

```
                              IQ4_NL    Q4_K_M
Cortex-A76  (Pi 5),  t=2      34.2       25.7    tok/s
Golden Cove (P-core), t=6    105.5       89.9
Gracemont   (E-core), t=4     43.6       29.3
```

`llama-bench -p 128 -n 128`, five repetitions, each core at its own best thread count, both
x86 rows pinned by affinity mask so the core type varies and the memory system does not. On
the Pi the i-quant also used 29% less energy per token off the PMIC rails. Q4_K_M is better
on perplexity by 0.58 points, which is a real trade, just not the trade the card describes.

Part of what I measured at 0.5B is not the format. That scale has an embedding dimension of
896, which does not divide by 256, so llama-quantize substitutes per tensor: in that
Q4_K_M file only 12% of the repeating-layer bytes are Q4_K and 70% are Q5_0, and it streams
374 MiB per token against about 330 for the other 4-bit files. At 1.5B that goes away and
the gap narrows sharply, to 8% on the A76 and 6% on Golden Cove, with Gracemont flipping to
Q4_K_M by 3%.

So I do not think the card is wrong so much as that "slower on CPU" turns out to be a
property of the kernel coverage on a particular core rather than of CPUs. The ranking moved
between the two core types inside one laptop: Q4_0 leads the P-cores, IQ4_NL leads the
E-cores, same binary and same files.

One more thing you may already know and I did not: requantizing the same targets from a
Q8_0 intermediate instead of FP16 produced files with identical labels and identical
nominal bit widths that decoded up to 38% slower on the A76. If any of your cards are built
that way for some models and not others, that difference is invisible to anyone reading the
filename.

Three cores and one model family is a narrow base and I would not want the card changed on
my say-so. If it is useful, the data and the harness are at
github.com/manunicholasjacob/llama-roofline, and `llama-roofline inspect <file>.gguf`
prints the per-tensor type map if you ever want to see what a given upload actually
contains.

---

## If he replies asking for more

- The full matrix is `src/llama_roofline/data/format_matrix.csv`, 98 rows, each with the
  measurement file it came from.
- The archived dataset is 10.5281/zenodo.21938812.
- The thing most worth asking him: whether his imatrix quantizations behave differently.
  Everything measured here is plain quantization with no importance matrix, and his cards
  ship imatrix versions. That is a genuine gap and he is one of very few people who could
  close it.
