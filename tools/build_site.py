"""Build the documentation site, and the Hugging Face Space bundle, from the repository.

    python tools/build_site.py                 # writes _site/ (what GitHub Pages serves)
    python tools/build_site.py --space DIR     # writes a static Space: index.html + code

Every page is rendered from a file that already lives in the repository (README, docs/,
examples/, CHANGELOG, CONTRIBUTING), so the site cannot drift from the code. The browser
inspector loads ``gguf.py`` and ``inspection.py`` copied from ``src/`` at build time and
runs them in Pyodide: the page prints exactly what ``llama-roofline inspect`` prints.

Needs the ``markdown`` package for the site (``pip install markdown``); the Space bundle
needs nothing beyond the standard library.
"""

from __future__ import annotations

import argparse
import html
import pathlib
import re
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "llama_roofline"
WEB = ROOT / "web"
REPO = "https://github.com/manunicholasjacob/llama-roofline"
DOI = "10.5281/zenodo.21842493"

# (output file, source, nav title)
PAGES = [
    ("index.html", "README.md", "Overview"),
    ("inspector.html", None, "GGUF Inspector"),
    ("guide.html", "docs/BLOG.md", "Memory-bound? A guide"),
    ("method.html", "docs/METHOD.md", "Method"),
    ("examples.html", "examples/README.md", "Results gallery"),
    ("contributing.html", "CONTRIBUTING.md", "Contributing"),
    ("changelog.html", "CHANGELOG.md", "Changelog"),
]

# Repository-relative link targets that have a page of their own on the site.
LINK_MAP = {
    "README.md": "index.html",
    "docs/BLOG.md": "guide.html",
    "docs/METHOD.md": "method.html",
    "examples/README.md": "examples.html",
    "examples/": "examples.html",
    "examples": "examples.html",
    "CONTRIBUTING.md": "contributing.html",
    "CHANGELOG.md": "changelog.html",
}

CSS = """
:root { --bg:#ffffff; --fg:#1b1f24; --muted:#59636e; --line:#d0d7de; --panel:#f6f8fa; --accent:#9a4a00; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#0d1117; --fg:#e6edf3; --muted:#8b949e; --line:#30363d; --panel:#161b22; --accent:#f0a340; }
}
* { box-sizing:border-box; }
body { background:var(--bg); color:var(--fg); margin:0; padding:0 16px 56px;
       font:16px/1.6 system-ui, -apple-system, "Segoe UI", sans-serif; }
header.site { max-width:880px; margin:0 auto; padding:18px 0 10px; border-bottom:1px solid var(--line); }
header.site .brand { font-weight:700; color:var(--fg); text-decoration:none; margin-right:18px; }
header.site nav a { margin-right:14px; font-size:.95rem; white-space:nowrap; }
header.site nav a.here { color:var(--fg); font-weight:600; text-decoration:none; }
main { max-width:880px; margin:0 auto; padding-top:8px; }
a { color:var(--accent); }
h1 { font-size:1.9rem; letter-spacing:-0.01em; } h2 { margin-top:2em; border-bottom:1px solid var(--line); padding-bottom:.2em; }
pre { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:12px 14px; overflow-x:auto;
      font:13px/1.45 ui-monospace, SFMono-Regular, Consolas, monospace; }
code { font:.9em ui-monospace, SFMono-Regular, Consolas, monospace; }
pre code { font:inherit; }
table { border-collapse:collapse; display:block; overflow-x:auto; margin:1em 0; }
th, td { border:1px solid var(--line); padding:5px 10px; text-align:left; }
th { background:var(--panel); }
img { max-width:100%; height:auto; }
img[src$=".png"] { border:1px solid var(--line); border-radius:6px; background:#fff; }
blockquote { margin:1em 0; padding:0 1em; color:var(--muted); border-left:3px solid var(--line); }
footer { max-width:880px; margin:48px auto 0; color:var(--muted); font-size:.9rem;
         border-top:1px solid var(--line); padding-top:12px; }
details { margin:.6em 0; } summary { cursor:pointer; font-weight:600; }
"""


def version() -> str:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    return re.search(r'^version\s*=\s*"([^"]+)"', text, re.M).group(1)


def nav(here: str) -> str:
    links = []
    for out, _, title in PAGES:
        cls = ' class="here"' if out == here else ""
        links.append(f'<a href="{out}"{cls}>{html.escape(title)}</a>')
    links.append(f'<a href="{REPO}">GitHub</a>')
    links.append('<a href="https://pypi.org/project/llama-roofline/">PyPI</a>')
    return ('<header class="site"><a class="brand" href="index.html">llama-roofline</a>'
            f'<nav>{" ".join(links)}</nav></header>')


def rewrite_links(body: str, source: str) -> str:
    """Point repository-relative links at site pages, or at GitHub when there is none."""
    base = pathlib.PurePosixPath(source).parent

    def fix(m: re.Match) -> str:
        attr, target = m.group(1), m.group(2)
        if re.match(r"^(https?:|mailto:|#|data:)", target):
            return m.group(0)
        path, _, frag = target.partition("#")
        resolved = str(pathlib.PurePosixPath(base, path)) if path else ""
        resolved = re.sub(r"(^|/)\./", r"\1", resolved)
        while "/../" in "/" + resolved:
            resolved = re.sub(r"[^/]+/\.\./", "", resolved, count=1)
        key = resolved.rstrip("/") if resolved not in LINK_MAP else resolved
        if resolved in LINK_MAP or key in LINK_MAP:
            new = LINK_MAP.get(resolved, LINK_MAP.get(key))
        elif attr == "src":
            new = f"{REPO}/raw/main/{resolved}"
        else:
            kind = "tree" if (ROOT / resolved).is_dir() else "blob"
            new = f"{REPO}/{kind}/main/{resolved}"
        return f'{attr}="{new}{"#" + frag if frag else ""}"'

    return re.sub(r'(href|src)="([^"]+)"', fix, body)


def render_markdown(source: str) -> str:
    import markdown  # only the site needs it

    text = (ROOT / source).read_text(encoding="utf-8")
    md = markdown.Markdown(extensions=["fenced_code", "tables", "toc", "sane_lists"],
                           extension_configs={"toc": {"permalink": False}})
    return rewrite_links(md.convert(text), source)


def gallery_extra() -> str:
    """Each example machine's report and figure, under the gallery README."""
    parts = []
    for d in sorted(p for p in (ROOT / "examples").iterdir() if p.is_dir()):
        rel = f"examples/{d.name}"
        parts.append(f'<h2 id="{d.name}">{html.escape(d.name)}</h2>')
        if (d / "roofline.png").exists():
            parts.append(f'<p><img src="{rel}/roofline.png" alt="Roofline figure for {d.name}" '
                         'loading="lazy"></p>')
        if (d / "report.txt").exists():
            txt = (d / "report.txt").read_text(encoding="utf-8")
            parts.append("<details><summary>Terminal report</summary><pre>"
                         f"{html.escape(txt)}</pre></details>")
        files = ", ".join(f'<a href="{REPO}/blob/main/{rel}/{f.name}">{f.name}</a>'
                          for f in sorted(d.iterdir()) if f.is_file())
        parts.append(f"<p>Files: {files}</p>")
    return "\n".join(parts)


def page(here: str, title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)} - llama-roofline</title>
<meta name="description" content="Is your llama.cpp decode memory-bandwidth-bound? Measure it, in one command.">
<style>{CSS}</style>
</head>
<body>
{nav(here)}
<main>
{body}
</main>
<footer>llama-roofline {version()}, MIT licensed. Manu Nicholas Jacob, Independent Researcher.
Cite as <a href="https://doi.org/{DOI}">doi:{DOI}</a>. Built from
<a href="{REPO}">the repository</a>; every page here is a file there.</footer>
</body>
</html>
"""


def inspector_html(with_nav: bool) -> str:
    text = (WEB / "inspector.html").read_text(encoding="utf-8").replace("{{VERSION}}", version())
    if with_nav:
        links = " ".join(f'<a href="{o}">{html.escape(t)}</a>' for o, _, t in PAGES
                         if o != "inspector.html")
        text = text.replace("<!--NAV-->", f'<nav class="top"><b>llama-roofline</b> {links}</nav>')
    return text


def copy_code(out: pathlib.Path) -> None:
    for name in ("gguf.py", "inspection.py"):
        shutil.copyfile(SRC / name, out / name)


def build_site(out: pathlib.Path) -> None:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for here, source, title in PAGES:
        if source is None:
            (out / here).write_text(inspector_html(with_nav=True), encoding="utf-8")
            continue
        body = render_markdown(source)
        if here == "examples.html":
            body += gallery_extra()
        (out / here).write_text(page(here, title, body), encoding="utf-8")
    copy_code(out)
    for d in (ROOT / "examples").iterdir():
        if d.is_dir() and (d / "roofline.png").exists():
            (out / "examples" / d.name).mkdir(parents=True, exist_ok=True)
            shutil.copyfile(d / "roofline.png", out / "examples" / d.name / "roofline.png")
    (out / ".nojekyll").write_text("", encoding="utf-8")


SPACE_README = """---
title: GGUF Inspector
emoji: "\\U0001F50D"
colorFrom: yellow
colorTo: gray
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: What is really inside a GGUF file, read in your browser
tags:
  - gguf
  - llama.cpp
  - quantization
  - local-llm
---

# GGUF Inspector

Pick a GGUF model file, or name one on the Hub, and see what it actually contains: how much
of it is the format its name claims, its real bits per weight, how many bytes generation
reads from memory for every token, and (if you give your memory bandwidth) the ceiling that
puts on tokens per second.

Only the header is read: the first few MB of a local file, in your browser, or a few MB of
a Hub file through HTTP range requests. Nothing is uploaded and the weights are never
downloaded. A link to a Hub model can be shared as `?model=hf:OWNER/REPO/FILE.gguf`.

The Hub's own file viewer lists tensor types too. What this adds is the on-label share, the
bits per weight, the bytes read per token (embedding excluded, a tied output head counted)
and the speed ceiling.

This is the `inspect` command of
[llama-roofline](https://github.com/manunicholasjacob/llama-roofline) {version} running
unchanged in Pyodide: `gguf.py` and `inspection.py` are copied byte for byte from
`src/llama_roofline/` at that release (MIT). Gated repositories need a token, which a web
page should not hold; use the CLI for those:

```bash
pip install llama-roofline
HF_TOKEN=... llama-roofline inspect hf:OWNER/REPO/FILE.gguf
llama-roofline diagnose        # measure your own bandwidth and decode speed
```

Cite: M. N. Jacob, llama-roofline, doi:10.5281/zenodo.21842493.
"""


def build_space(out: pathlib.Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(inspector_html(with_nav=False), encoding="utf-8")
    (out / "README.md").write_text(SPACE_README.replace("{version}", version()),
                                   encoding="utf-8")
    copy_code(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(ROOT / "_site"), help="site output directory")
    ap.add_argument("--space", metavar="DIR", help="write the Hugging Face Space bundle "
                                                   "to DIR instead of building the site")
    args = ap.parse_args(argv)
    if args.space:
        build_space(pathlib.Path(args.space))
        print(f"Space bundle written to {args.space}")
    else:
        build_site(pathlib.Path(args.out))
        print(f"site written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
