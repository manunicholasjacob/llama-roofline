# LinkedIn draft

Post after the blog is live on manunicholasjacob.com, and link to the blog rather than
straight to GitHub. LinkedIn rewards the post that keeps people reading.

---

Your local LLM is probably not compute-limited. It is waiting on memory.

When a dense transformer generates text one token at a time, the forward pass for that
token reads every weight in the model out of DRAM, uses each one once, and throws it away.
Nothing is reused. So generation speed comes down to one equation:

decode tok/s = (memory bandwidth you actually sustain) / (model bytes)

Prefill is the opposite. Processing a prompt reuses the same weights across every position
in it, which makes it compute-bound and core-hungry. Two phases of the same model sitting
on opposite sides of the roofline is why threading advice for local LLMs never agrees with
itself.

I built a small open-source CLI that measures where your machine actually sits:
llama-roofline. It measures your sustainable memory bandwidth, benchmarks your own models
through llama.cpp, fits the roofline, and tells you in plain English which knob will move
your throughput and which one will not.

I ran it on two machines about 20x apart in price: an Intel i7-12700H laptop with DDR5, and
a 2 GB Raspberry Pi 5. Both land in the same place. Decode runs between 65% and 97% of the
memory ceiling, and throughput tracks 1/model_bytes with an R² of 0.987 on the laptop and
0.980 on the Pi, seven models each. The roofline is not a property of one device, which is
what makes it worth knowing.

This comes directly out of my research on the memory wall in edge AI inference. The tool is
MIT licensed and the throughput numbers all come from llama.cpp, which does the hard part.

If you run local models, I would like your report card, especially from hardware I do not
own. Apple Silicon with unified memory is the one I most want to see.

[link to blog post]

#EdgeAI #LLM #ComputerArchitecture #OpenSource #Inference

---

## Notes

- Keep the first two lines strong. LinkedIn truncates after roughly three.
- Do not use em dashes.
- Reply to every comment for the first day.
- Post mid-week, morning US time.
