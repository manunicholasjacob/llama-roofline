"""The browser inspector bundle: flat modules, same output as the CLI, version stamped."""

import importlib.util
import os
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools"


def _load_builder():
    spec = importlib.util.spec_from_file_location("build_site", TOOLS / "build_site.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_space_bundle_is_complete_and_stamped(tmp_path):
    b = _load_builder()
    b.build_space(tmp_path)
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == ["README.md", "gguf.py", "index.html", "inspection.py"]
    html = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert "{{VERSION}}" not in html
    assert f'content="{b.version()}"' in html
    assert "<!--NAV-->" in html  # the Space page has no site navigation
    readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert readme.startswith("---\ntitle: GGUF Inspector")
    assert "sdk: static" in readme and b.version() in readme
    # Byte for byte, so a reviewer can diff the Space against the tagged release.
    for name in ("gguf.py", "inspection.py"):
        assert (tmp_path / name).read_bytes() == (ROOT / "src" / "llama_roofline" / name).read_bytes()
    for f in tmp_path.iterdir():
        assert all(c < 128 for c in f.read_bytes()), f"{f.name} is not ASCII"


def test_vendored_modules_print_what_the_cli_prints(tmp_path):
    """inspection.py next to gguf.py, outside the package, as Pyodide runs it."""
    b = _load_builder()
    b.build_space(tmp_path / "space")
    model = tmp_path / "m.gguf"
    _write_model(model)

    from llama_roofline import __version__, cli, inspection
    want = inspection.render_text(cli.inspect_model(str(model)), __version__, True, 50.0)
    assert "100% Q8_0" in want and "ceiling" in want and "EVERY TENSOR" in want

    code = (
        "import io, os, sys\n"
        "sys.path.insert(0, os.getcwd())\n"
        "import gguf, inspection\n"
        f"p = {str(model)!r}\n"
        "with open(p, 'rb') as f:\n"
        "    kv, t = gguf.read_gguf(io.BytesIO(f.read()))\n"
        "r = inspection.summarize([(kv, t, os.path.getsize(p))], os.path.basename(p), p)\n"
        f"sys.stdout.write(inspection.render_text(r, {__version__!r}, True, 50.0))\n"
    )
    out = subprocess.run([sys.executable, "-I", "-c", code], cwd=tmp_path / "space",
                         capture_output=True, text=True, check=True)
    assert out.stdout.replace("\r\n", "\n") == want


def _write_model(path):
    """A small valid GGUF: one block's worth of Q8_0 and F32 tensors and a label."""
    import struct

    def s(x):
        b = x.encode()
        return struct.pack("<Q", len(b)) + b

    kv = [
        (s("general.architecture") + struct.pack("<I", 8) + s("llama")),
        (s("general.file_type") + struct.pack("<I", 4) + struct.pack("<I", 7)),  # Q8_0
    ]
    tensors = [("blk.0.attn_q.weight", 8, [64, 64]), ("blk.0.ffn_down.weight", 8, [64, 64]),
               ("output_norm.weight", 0, [64]), ("output.weight", 8, [64, 128]),
               ("token_embd.weight", 8, [64, 128])]
    sizes = {8: lambda n: n // 32 * 34, 0: lambda n: n * 4}
    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kv)) + b"".join(kv)
    off, table, blobs = 0, b"", []
    for name, tid, dims in tensors:
        n = 1
        for d in dims:
            n *= d
        size = sizes[tid](n)
        table += s(name) + struct.pack("<I", len(dims)) + struct.pack(f"<{len(dims)}Q", *dims)
        table += struct.pack("<IQ", tid, off)
        blobs.append(b"\0" * size + b"\0" * (-size % 32))
        off += size + (-size % 32)
    data = head + table
    data += b"\0" * (-len(data) % 32)
    path.write_bytes(data + b"".join(blobs))


def test_site_links_point_at_pages_or_github(tmp_path):
    pytest.importorskip("markdown")
    b = _load_builder()
    b.build_site(tmp_path / "site")
    index = (tmp_path / "site" / "index.html").read_text(encoding="utf-8")
    assert 'href="method.html' in index and 'href="examples.html"' in index
    assert f'href="{b.REPO}/blob/main/CITATION.cff"' in index
    for page in tmp_path.joinpath("site").glob("*.html"):
        text = page.read_text(encoding="utf-8")
        assert 'href="docs/' not in text and 'href="examples/"' not in text, page.name
    gallery = (tmp_path / "site" / "examples.html").read_text(encoding="utf-8")
    for d in (ROOT / "examples").iterdir():
        if (d / "roofline.png").exists():
            assert (tmp_path / "site" / "examples" / d.name / "roofline.png").exists()
            assert f'src="examples/{d.name}/roofline.png"' in gallery
    assert os.path.exists(tmp_path / "site" / ".nojekyll")
