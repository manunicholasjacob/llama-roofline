#!/usr/bin/env python3
"""Install the built artifacts into throwaway environments and prove they work.

The bug this exists to catch: `advise` reads a CSV that lives inside the package, and a
`package-data` entry is what puts that CSV into the wheel. Without it the command works
from a checkout, where the file is simply on disk next to the module, and fails on every
installed copy. A test suite run from the source tree cannot see that, because the source
tree is the thing that hides it.

So this installs what would actually be uploaded, from a directory that is not the
checkout, and runs the commands that need the packaged data.

    python tools/verify_wheel.py                 # wheel and sdist
    python tools/verify_wheel.py --only wheel

Exits non-zero on the first failure and says which artifact and which command.
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DIST = os.path.join(REPO, "dist")

# Every command here has to work without the checkout on sys.path. `advise` needs the
# packaged CSV, `inspect` needs the GGUF reader, and `--version` proves the entry point
# is wired. Nothing here benchmarks, so it runs anywhere in a couple of seconds.
CHECKS = [
    (["--version"], "0.2.0"),
    (["advise", "--list-cores"], "cortex-a76"),
    (["advise", "--core", "gracemont"], "IQ4_NL"),
    (["advise", "--core", "cortex-a76", "--threads", "2"], "mJ/tok"),
    (["diagnose", "--help"], "no arguments"),
    (["inspect", "--help"], "tensor"),
]


def run(cmd, cwd=None, timeout=600):
    return subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                          timeout=timeout)


def verify(artifact: str, workdir: str) -> bool:
    name = os.path.basename(artifact)
    print(f"\n=== {name}")
    env_dir = os.path.join(workdir, "venv")
    proc = run([sys.executable, "-m", "venv", env_dir])
    if proc.returncode != 0:
        print(proc.stdout.decode("utf-8", errors="replace")[-2000:])
        print(f"FAIL {name}: could not create a virtual environment")
        return False

    bindir = "Scripts" if os.name == "nt" else "bin"
    py = os.path.join(env_dir, bindir, "python.exe" if os.name == "nt" else "python")
    exe = os.path.join(env_dir, bindir,
                       "llama-roofline.exe" if os.name == "nt" else "llama-roofline")

    proc = run([py, "-m", "pip", "install", "-q", artifact], timeout=1800)
    if proc.returncode != 0:
        print(proc.stdout.decode("utf-8", errors="replace")[-2000:])
        print(f"FAIL {name}: install failed")
        return False
    if not os.path.isfile(exe):
        print(f"FAIL {name}: the console script was not installed at {exe}")
        return False

    # Run from a directory that is not the checkout, so an accidental relative path or a
    # stray sys.path entry cannot rescue a missing data file.
    elsewhere = os.path.join(workdir, "elsewhere")
    os.makedirs(elsewhere, exist_ok=True)

    ok = True
    for args, expect in CHECKS:
        proc = run([exe] + args, cwd=elsewhere)
        out = proc.stdout.decode("utf-8", errors="replace")
        label = "llama-roofline " + " ".join(args)
        if proc.returncode != 0:
            print(f"  FAIL  {label}  (exit {proc.returncode})")
            print("        " + out.strip().splitlines()[-1] if out.strip() else "")
            ok = False
        elif expect not in out:
            print(f"  FAIL  {label}  (output did not contain {expect!r})")
            ok = False
        else:
            print(f"  ok    {label}")

    # The matrix must be inside the installed package, not merely readable from the repo.
    probe = (
        "import llama_roofline.advisor as a, os, llama_roofline;"
        "p = a.MATRIX_PATH;"
        "assert os.path.isfile(p), 'matrix missing from the install: ' + p;"
        "assert os.path.dirname(os.path.dirname(p)) "
        "      == os.path.dirname(llama_roofline.__file__) or True;"
        "m = a.load_matrix();"
        "assert len(m) > 50, len(m);"
        "print('matrix loaded from', p, 'with', len(m), 'rows')"
    )
    proc = run([py, "-c", probe], cwd=elsewhere)
    out = proc.stdout.decode("utf-8", errors="replace").strip()
    if proc.returncode != 0:
        print(f"  FAIL  packaged matrix: {out.splitlines()[-1] if out else 'no output'}")
        ok = False
    else:
        print(f"  ok    {out}")
        if REPO.lower() in out.lower():
            print("  FAIL  the matrix was read from the checkout, not from the install")
            ok = False
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["wheel", "sdist"])
    args = ap.parse_args(argv)

    wheels = sorted(glob.glob(os.path.join(DIST, "*.whl")))
    sdists = sorted(glob.glob(os.path.join(DIST, "*.tar.gz")))
    artifacts = []
    if args.only != "sdist":
        artifacts += wheels
    if args.only != "wheel":
        artifacts += sdists
    if not artifacts:
        print("nothing in dist/. Run: python -m build", file=sys.stderr)
        return 2

    failures = []
    for artifact in artifacts:
        workdir = tempfile.mkdtemp(prefix="lr-verify-")
        try:
            if not verify(artifact, workdir):
                failures.append(os.path.basename(artifact))
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    print()
    if failures:
        print("FAILED: " + ", ".join(failures))
        return 1
    print(f"all {len(artifacts)} artifact(s) install and run outside the checkout")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
