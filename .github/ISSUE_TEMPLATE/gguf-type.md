---
name: A GGUF file it cannot read or size
about: inspect printed "type<N>", "?", or could not read a file that llama.cpp loads
title: "[gguf] <type name or file>"
labels: gguf
---

## The file

<!--
Where it came from, so the header can be read again. A Hub reference is best, since
`inspect` reads only the header and nobody has to download the model:

  hf:OWNER/REPO/FILE.gguf
-->

## What inspect printed

```
paste `llama-roofline inspect <file>` here (add --tensors if the problem is one tensor type)
```

## The type, if you know it

<!--
The ggml type name and id (from ggml/include/ggml.h in llama.cpp), and the block size and
bytes per block (from ggml/src/ggml-common.h) if you know them. A link to the llama.cpp
commit that added the type helps most.
-->

- llama-roofline version (`llama-roofline --version`):
- llama.cpp build that wrote the file, if known:
