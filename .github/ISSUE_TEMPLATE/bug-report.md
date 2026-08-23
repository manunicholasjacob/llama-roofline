---
name: Bug report
about: Something failed, or a number looks wrong
title: ""
labels: bug
---

## What happened

<!-- What you ran, and what came back. -->

```
paste the command and its output, including any error
```

## Version and platform

<!--
  llama-roofline --version
  python --version
-->

- llama-roofline:
- Python:
- OS and CPU:
- llama.cpp build (the report prints it, or run `llama-bench --help | head -1`):

## The results file, if there is one

<!--
`diagnosis.json` or `roofline.json` from your output directory usually contains the
answer: it carries the machine description, the ceiling measurement including its
per-repetition stability, the llama.cpp build, and every raw number. Attach it if you can.

One thing worth checking first: were other processes using the machine? Background load
depresses the measured ceiling and inflates every percentage derived from it. The tool
flags this when it can see it, but constant load is invisible to a best-of-N measurement.
-->
