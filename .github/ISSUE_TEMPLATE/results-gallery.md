---
name: Results gallery
about: Share what this said about your machine
title: "[results] <CPU / device name>"
labels: results
---

## Machine

<!-- CPU or SoC, RAM type and size, operating system. -->

## Report

<!--
Paste diagnosis.md from your output directory, or the terminal report:

  llama-roofline diagnose

If you also ran the format comparison, paste that too. It is the more useful of the two,
because the rankings are core-specific and only three cores have been measured:

  llama-roofline advise --measure --models ~/models/<several quants of one model>
-->

```
paste here
```

## Did it get anything wrong?

<!--
The interesting part. Did the verdict match your experience? Did the recommended thread
count actually turn out to be fastest in real use? If you ran the format comparison, does
your ordering match anything in `llama-roofline advise`, or does it contradict it?

A result that contradicts the model is worth more than one that confirms it. Negative
results are welcome.
-->
