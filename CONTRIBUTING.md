# Contributing

Two kinds of contribution are especially wanted.

## 1. Send in your report card

The most useful thing you can do is run the tool on hardware I do not have and share what
came out. Apple Silicon, Ryzen, Snapdragon X, Ampere, an old Xeon, a Jetson, a Pi Zero:
all of it is interesting, and the disagreements are the most interesting part.

```bash
llama-roofline run --models ~/models/*.gguf
```

Then open an issue with the **Results gallery** template and paste `report.md`. If the
tool told you something that turned out to be wrong for your setup, say so in the same
issue. A report that contradicts the model is worth more than one that confirms it.

## 2. Code

```bash
git clone https://github.com/manunicholasjacob/llama-roofline
cd llama-roofline
python -m pip install -e ".[dev]"
python -m pytest -q
```

House rules:

- **The core stays dependency-free.** `numpy` and `matplotlib` are installed by default so
  the tool works on first run, but nothing in the analysis path may import them: `numpy` is
  needed only to measure the bandwidth ceiling and `matplotlib` only to draw the figure. CI
  has a job that proves the tool still runs with neither installed. Do not add a third.
- **Every measurement claim needs a measurement.** If a change alters what the report
  says about a machine, show the before and after from a real run in the pull request.
- **Failures degrade, they do not crash.** A missing binary, an unparseable GGUF, a model
  that will not load: each of these gets reported and the run continues. Look at how
  `gguf.describe` and the plotting call in `cli.py` handle their failure cases.
- **Be honest in the report text.** The tool's only real asset is that people believe its
  numbers. If a result is uncertain, the report says it is uncertain. The ">100% of peak"
  warning is the model for this: when the measurement does not support the conclusion,
  say what is actually known.
- Add a test. The suite runs in under two seconds and needs neither llama.cpp nor a GGUF
  file; synthesise what you need the way `tests/test_gguf.py` does.

## A GGUF type it cannot size

New ggml types appear in llama.cpp every few months. When `inspect` prints `type<N>` or a
`?` in the size column, the fix is usually one line in `GGML_TYPES` in
`src/llama_roofline/gguf.py`: the type id from `ggml/include/ggml.h`, and the block size and
bytes per block from the `static_assert` next to the block struct in
`ggml/src/ggml-common.h`. Then check it against a real file: for every tensor, the gap to
the next tensor's offset minus the computed size must be smaller than the alignment (32
bytes unless the file says otherwise). `inspect hf:OWNER/REPO/FILE.gguf --tensors --json`
gives the offsets without downloading the weights. Name the file you checked in the pull
request. If you only have the file and not the layout, open an issue with the
"A GGUF file it cannot read or size" template instead.

## The documentation site and the browser inspector

`python tools/build_site.py` builds the site into `_site/` (it needs `pip install
markdown`); every page is rendered from a file in this repository, so edit the source file,
not the HTML. The inspector is `web/inspector.html`; it loads `gguf.py` and
`inspection.py` from `src/llama_roofline/` at build time and runs them in Pyodide, so those
two files must stay standard library only and importable as flat modules (a test checks
both). `python -m http.server --directory _site` serves it locally.
`python tools/build_site.py --space DIR` writes the same inspector as a static Hugging Face
Space.

## Things that would genuinely help

- A better portable bandwidth ceiling. The numpy kernels are a lower bound and everyone
  knows it. A small optional C or Rust STREAM shim, or a way to read the ceiling off the
  hardware, would tighten every percentage the tool prints.
- Real bytes-per-token for mixture-of-experts models, instead of excluding them.
- KV-cache-aware bytes-per-token, so long-context runs land on the roofline too.
- A Metal or CUDA path: same analysis, with VRAM bandwidth as the ceiling.
- Energy per token on hardware with a readable power rail. The physics is already worked
  out; it is the portability that is hard.
