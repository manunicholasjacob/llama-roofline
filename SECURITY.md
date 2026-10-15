# Security

`llama-roofline` reads GGUF files you give it: from a local path, or from a URL or
`hf:` reference by HTTP range requests that fetch only the header. It parses the
metadata and tensor table and never executes anything inside a model file. The parser
is pure Python and treats every file as untrusted input: a file that is not valid GGUF
should end in a short error message, never in a traceback. The browser inspector runs
the same parser in the page, so a file you open there is read in your browser and not
uploaded.

`HF_TOKEN`, when set, is sent to huggingface.co only and is not forwarded on the
redirect to the Hub's file store.

The benchmarking commands (`diagnose`, `advise`, `run`) can launch a `llama-bench`
binary: the one given by `--llama-bench`, else `$LLAMA_BENCH`, else the first found on
`PATH` or in the usual build directories. That binary runs with your permissions, like
any program you start yourself. `inspect` launches nothing.

## Reporting a vulnerability

Please report security problems privately, through GitHub's
[private vulnerability reporting](https://github.com/manunicholasjacob/llama-roofline/security/advisories/new)
for this repository, rather than in a public issue. Include the version
(`llama-roofline --version`), the command, and the file or URL that triggers it, or a
description of how to build one.

A file that makes the parser crash with a traceback instead of an error message is a
bug worth reporting too; an ordinary issue is fine for that.

This is a single-maintainer project, so expect an acknowledgement within days rather
than hours. Fixes are released as a new PyPI version and noted in `CHANGELOG.md`.

## Supported versions

Only the latest release on PyPI receives fixes.
