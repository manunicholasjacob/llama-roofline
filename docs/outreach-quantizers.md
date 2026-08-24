# A note to the people whose model cards answer this question

Staged, not sent. Send it after the r/LocalLLaMA post exists, so there is something to point
at that is not just a repository. Tone is deliberately soft, per the decision of record:
first contact with someone who owes you nothing, and a card edit is worth more to the record
than to the tool anyway.

## Who, and why them

Nearly everyone who downloads a GGUF gets it from a small number of quantizers, and their
model cards carry a "Which file should I choose?" section that is the community's default
answer. `bartowski/Qwen_Qwen3.6-35B-A3B-GGUF` alone is at 435,666 downloads, and the section
is boilerplate across the cards, so the readership is much larger than any one repo.

**The claim to make is not the one I had first.** The earlier draft argued that the card's
line about i-quants being slower on CPU is wrong. Parsing the tensor maps at two model sizes
showed that most of that gap at 0.5B is llama-quantize substituting types rather than the
format itself, and the comparison reverses on two cores at 1.5B. Leading with it would have
been picked apart by someone who was right.

What replaced it is better, because it is a property of the files and he can check it in a
minute on his own uploads: at 0.5B, of eight formats built from one FP16 source, only three
contained the type their name claims.

**Where to send it.** The Community tab of one of the model cards, rather than a DM, because
the observation belongs where the guidance is and a public thread lets other people check
it. Pick one card, not several.

**What not to do.** Do not open with the tool. Do not ask him to link anything. Do not tell
him his card is wrong.

---

## Draft

Subject, if it needs one: what is actually inside the small-model GGUFs

---

Your cards are where I send people who ask which file to download, so I wanted to bring you
something I found rather than an opinion.

I quantized Qwen2.5-0.5B into eight formats from one FP16 source and parsed the tensor
tables before benchmarking anything. Percentages are of repeating-layer bytes:

```
Q4_0      100% Q4_0        4.50 bits/weight
Q8_0      100% Q8_0        8.50
IQ4_NL     95% IQ4_NL      4.55
IQ4_XS     24% IQ4_XS      4.48   (70% of it is IQ4_NL)
Q6_K       24% Q6_K        7.93   (76% is Q8_0)
Q4_K_M     12% Q4_K        5.52   (70% is Q5_0)
Q3_K_M      0% Q3_K        4.57
Q2_K        0% Q2_K        4.20
```

You will know the cause better than I do: k-quants want blocks of 256, that model's
embedding dimension is 896, and llama-quantize substitutes per tensor when a shape does not
divide. It warns per tensor and prints no summary. The part I had not appreciated is how far
it goes at small sizes, and that Q4_K_M ends up storing 5.52 bits per weight when the name
implies 4.5.

At 1.5B it mostly goes away: Q4_K_M 79% on-label, Q3_K_M 58%, IQ4_XS 95%.

I am not suggesting anything is wrong with the quantization. What made me want to write is
that people compare quants on small models precisely because it is cheap, and that is
exactly where the file is least likely to be what its name says. If someone benchmarks
Q4_K_M against IQ4_XS on a 0.5B and concludes something about the formats, a good part of
what they measured is the substitution.

If it is useful, `llama-roofline inspect <file>.gguf` prints this for any GGUF in about a
second. It is a header read and it touches nothing else. I built it while measuring decode
bandwidth and this fell out of it:
github.com/manunicholasjacob/llama-roofline

Whether any of that belongs on a card is your call and I would not want it changed on my
say-so. I mostly thought you would want to know, and I would be interested to hear whether
your imatrix builds behave differently, since everything above is plain quantization with no
importance matrix and you are one of very few people who could answer that.

---

## If he replies

- The matrix is `src/llama_roofline/data/format_matrix.csv`, 98 rows, each with the
  measurement file it came from, and now with an on-label column and effective bits per
  weight per artifact.
- The archived dataset is 10.5281/zenodo.21938812.
- The question most worth asking: whether an importance matrix changes the substitution
  pattern. It is a real gap, it is cheap for him to answer and expensive for anyone else,
  and it would make the finding considerably more useful.
- If he asks about speed, give him the streamed-normalised numbers and the 1.5B reversal in
  the same message. Do not hand him the 0.5B speed table on its own.
