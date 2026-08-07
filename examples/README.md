# Results gallery

Real output from real machines. Each directory holds exactly what `llama-roofline run`
writes: `roofline.json` (machine-readable), `report.txt` (the terminal report card),
`report.md` (shareable), and `roofline.png`.

Two devices at opposite ends of the range are included here. **If you run the tool, please
add yours**: open an issue with the "Results gallery" template. Hardware I do not own is
the most useful thing anyone can contribute, and results that *disagree* with the model
are more interesting than results that confirm it.

| device | memory | models | ceiling | fitted BW_eff | R² | utilisation | verdict |
|---|---|---:|---:|---:|---:|---|---|
| [Intel i7-12700H](x86-i7-12700H/) | DDR5-4800, 34 GB | 7 (0.5B to 7B) | 53.9 GB/s | 37.65 GB/s | 0.987 | 65-97% | memory-bound |
| [Raspberry Pi 5 (Cortex-A76)](rpi5-cortex-a76/) | LPDDR4X, 2 GB | 7 (0.5B to 1.5B) | 13.98 GB/s | 10.69 GB/s | 0.980 | 73-86% | memory-bound |

The two machines differ by 3.5x in fitted bandwidth and by roughly 20x in price, and both
land in the same place: decode sits between 65% and 97% of the memory ceiling, and
throughput tracks `1/model_bytes` with an R² above 0.98. That is the point of the tool.
The roofline is not a property of any one device.

Two details worth reading off these:

- On the laptop the **largest** model runs **closest** to the wall (the 7B at 97% of
  ceiling, the 0.5B Q4 at 65%). Bigger working sets stream more efficiently and amortise
  the per-token overhead that is not weight traffic.
- Using every hardware thread is a trap on both machines. On the laptop, `-t 20` cost up
  to 63% of decode throughput versus `-t 8`. On the Pi, using all 4 cores cost up to 21%
  versus 2. Prefill, meanwhile, kept scaling in both cases.

## About the Raspberry Pi 5 numbers

These were **not** produced by running `llama-roofline` on the Pi. They are the raw
`llama-bench` measurements from the study this tool grew out of, re-expressed in the
tool's schema and re-analysed by the tool's own code, using
[`rpi5-cortex-a76/convert_paper_data.py`](rpi5-cortex-a76/convert_paper_data.py). Read the
docstring at the top of that script for the full provenance note; the differences from a
native run (file size instead of `model_size` for bytes-per-token, an externally supplied
bandwidth ceiling) are recorded in the results JSON and printed in the report's caveats.

The conversion is also a correctness check: handed the study's raw data, the tool
recovers the published fit of 10.69 GB/s at R² = 0.980 over 7 points, exactly.

## Reproducing these

```bash
# x86 example
llama-roofline run --models /path/to/models --threads 1,4,8,14,20 --reps 3

# Pi 5 example, from the published data
cd examples/rpi5-cortex-a76
python convert_paper_data.py /path/to/edge-llm/results
```
