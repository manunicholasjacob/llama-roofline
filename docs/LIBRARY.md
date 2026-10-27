# Using llama-roofline as a library

The command line is the supported interface and the one the README documents. The functions
below are the pieces it is built from, and they are the ones to call if you want the numbers
without the report. They return plain dictionaries and lists, so nothing downstream has to
know about this package's types.

Every example here runs without numpy or matplotlib installed, which the `no-optional-deps`
CI job checks on each push.

## Read a GGUF file's metadata

`gguf.describe(path) -> dict`

Best-effort description of a GGUF file. It never raises: when the file cannot be read or the
metadata is malformed, it returns the same dictionary with `metadata_ok` false and
`metadata_error` set. Keys: `path`, `name`, `file_bytes`, `arch`, `quant`, `n_params`,
`n_expert`, `n_expert_used`, `context_length`, `is_moe`, `metadata_ok`, `metadata_error`.

```python
from llama_roofline import gguf

info = gguf.describe("qwen2.5-0.5b-instruct-q4_k_m.gguf")
print(info["arch"], info["n_params"], info["file_bytes"])
```

It reads the header only, not the weights.

## Read the per-tensor type map

`gguf.read_tensors(path, max_tensors=100000) -> list[dict]`

Returns one record per tensor with its name, ggml type, shape and size in bytes. This is the
part of the file that determines what it costs to run, and the part the filename does not
describe: two files carrying the same format label can hold different types per tensor
depending on what they were quantized from.

## Break a model down by quantization type

`gguf.composition(tensors, label=None) -> dict | None`

Takes the list from `read_tensors` and groups the repeating layers by type. Returns `None`
when there is nothing to group. Useful keys: `by_type_bytes` and `by_type_pct` (bytes and
percentage per ggml type, largest first), `bits_per_weight`, `repeating_bytes`,
`repeating_tensors`, and, when you pass a `label`, `expected_type`, `on_label_bytes` and
`on_label_pct`, which say how much of the model really is the advertised format.

```python
from llama_roofline import gguf

tensors = gguf.read_tensors("model.gguf")
comp = gguf.composition(tensors, label="Q4_K_M")
print(comp["bits_per_weight"], comp["on_label_pct"])
for ggml_type, pct in comp["by_type_pct"].items():
    print(ggml_type, pct)
```

## Measure this machine's memory-bandwidth ceiling

`membw.measure(working_set_mb=None, thread_counts=None, reps=5, ram_bytes=None, progress=None) -> dict`

Runs the read-only microbenchmark and returns the best result across kernels and thread
counts. Keys include `peak_read_GBs`, `best_kernel`, `best_threads`, `by_threads`,
`stability` and `working_set_mb`. The ceiling is a lower bound on what the hardware can do,
for the reasons given in `METHOD.md` section 2, and should be read as one.

```python
from llama_roofline import membw

ceiling = membw.measure(reps=3)
print(ceiling["peak_read_GBs"], ceiling["best_threads"])
```

### Checking the ceiling was measured on an idle machine

Two keys answer that, and they catch different things.

`stability` is the median-to-best ratio over repetitions. Below `STABILITY_THRESHOLD`
(0.75) the result carries `unstable: True` and a `warning`. This catches competition that
comes and goes, because that makes the repetitions disagree.

`foreign_cpu_fraction` is the share of the machine's total CPU capacity used by *other*
processes while the measurement ran, from `/proc` deltas. Above
`contention.CONTENTION_THRESHOLD` (0.10) the result carries `contended: True`,
`busiest_other_processes` and a `warning`. This catches the case `stability` cannot:
competition that lasts for the whole measurement depresses every repetition equally, so
stability stays high while the ceiling is wrong. On a Galaxy M21 running Android's
background dexopt job, the ceiling read 6.35 GB/s against 9.32 GB/s on the same quiet
device, 32% low, at stability 0.85.

`foreign_cpu_fraction` is `None` where `/proc` is unavailable, which includes Windows and
macOS. `None` means the check could not run, not that the machine was idle.

A `contended` warning says the machine was busy, not that the number is wrong. What it
costs depends on whether the other work touched memory: two CPU-only spinners on a Pi 5
take the fraction to 0.50 and leave the ceiling unchanged. Treat it as a reason to
re-measure and compare.

```python
ceiling = membw.measure(reps=3)
if ceiling.get("contended") or ceiling.get("unstable"):
    print(ceiling["warning"])
```

## Apply the roofline

`roofline.analyze(models, peak_read_GBs) -> dict`

Given model records and a measured ceiling, returns per-model predictions and an overall
`verdict`, plus `fit`, `models`, `n_models_ok` and `warnings`. Pass `None` as the ceiling to
get the predictions without a verdict.

```python
from llama_roofline import roofline

result = roofline.analyze(models, peak_read_GBs=54.3)
print(result["verdict"])
for m in result["models"]:
    print(m["name"], m["decode_ts"])
```

## Assemble an inspection summary

`inspection.summarize(parts, name, source, fetched=None) -> dict`

Builds the summary the `inspect` subcommand prints from the per-file parts of a model,
including split models. `source` records where the bytes came from, so a remote inspection
and a local one produce the same shape.

## Stability

These are the functions the CLI itself calls, so they change when its output changes and not
otherwise. Anything not listed here is internal. If you depend on something else and want it
kept stable, open an issue saying which function and why.
