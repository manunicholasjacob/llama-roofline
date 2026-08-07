# Distribution checklist

A tool nobody runs is a tool that does not exist. This is the release and outreach plan,
in the order it should happen. Everything below is for Manu to execute; nothing here is
automated, and every public post goes out under his own name and review.

## Phase 0 - before anything is public

- [ ] Read every file in the repo top to bottom. It ships under your name.
- [ ] Replace the placeholder GitHub handle if the repo lands anywhere other than
      `github.com/manunicholasjacob/llama-roofline` (it appears in `pyproject.toml`,
      `CITATION.cff`, `.zenodo.json`, and `report.py`'s Markdown footer).
- [ ] Run the suite one more time: `python -m pytest -q`.
- [ ] Run the tool on a machine you control, end to end, and confirm the report reads
      the way you would say it out loud.
- [ ] Decide whether the Raspberry Pi 5 example stays as converted data or gets replaced
      by a native run on the Pi. A native run is better; the converted file is labelled
      honestly either way.

## Phase 1 - ship the repo

- [ ] `git init`, initial commit, push to a new **public** repo named `llama-roofline`.
- [ ] Repo topics: `llama-cpp`, `llm`, `roofline`, `memory-bandwidth`, `benchmark`,
      `quantization`, `gguf`, `local-llm`, `edge-ai`.
- [ ] Repo description: "Is your llama.cpp decode memory-bandwidth-bound? Find out in one
      command."
- [ ] Enable Issues and Discussions. Discussions is where the results gallery grows.
- [ ] Confirm the CI badge goes green on all three operating systems.
- [ ] Tag `v0.1.0` and cut a GitHub Release with the report card for your own machine in
      the release notes. The report card *is* the marketing.

## Phase 2 - make it citable  (DONE 2026-08-07)

- [x] Log in to Zenodo with GitHub, flip the switch for the `llama-roofline` repo.
- [x] Cut the `v0.1.0` release. **Note for next time:** the first attempt archived as
      *Failed* with no error shown in Zenodo's UI. Cause was `.zenodo.json`: the relation
      `isBasedOn` is not in Zenodo's vocabulary, and the license needs the lowercase SPDX
      id `mit`. Fixed, then the release was deleted and re-cut, which Zenodo reprocessed
      cleanly. If a future release shows Failed, look there first.
- [x] Add the DOI badge to the README and the DOIs to `CITATION.cff`.
      Concept DOI `10.5281/zenodo.21842493` (always latest);
      v0.1.0 DOI `10.5281/zenodo.21842494`.
- [ ] Add the software DOI to ORCID (0009-0007-6589-6572).
- [ ] Optional: submit the record to a relevant Zenodo community.

## Phase 3 - PyPI

- [ ] `python -m build` then `twine check dist/*`.
- [ ] Upload to TestPyPI first, install from it in a clean virtualenv, run
      `llama-roofline membw`.
- [ ] Upload to PyPI. Confirm `pip install llama-roofline` then `llama-roofline --version`
      works on a machine that has never seen the source.
- [ ] Add the PyPI badge back to the README (it was removed for the initial push, because a
      badge for a package that does not exist yet renders broken):
      `[![PyPI](https://img.shields.io/pypi/v/llama-roofline.svg)](https://pypi.org/project/llama-roofline/)`
- [ ] Switch the README's two install commands from `pip install git+https://...` to
      `pip install llama-roofline`, and delete the "A PyPI release is coming" line.
- [ ] Only now are the Phase 4 drafts accurate: they all say `pip install llama-roofline`.
      Do not post before this phase is done.

## Phase 4 - distribution

Order matters. Lead with r/LocalLLaMA: it is the single highest-signal audience for this
tool, and a good reception there gives the other channels something to point at.

- [ ] **r/LocalLLaMA** - the key channel. Draft is in `announce-reddit.md`. Post on a
      weekday morning US time. Lead with the report card, not the repo link. Answer every
      comment for the first 24 hours; comment replies are where adoption actually happens.
      Expect and welcome "my numbers look different" replies: that is the gallery filling
      itself.
- [ ] **Hacker News (Show HN)** - draft in `announce-hn.md`. Title: `Show HN: llama-roofline
      - find out if your local LLM is memory-bandwidth-bound`. Post separately from Reddit,
      at least a day apart, so you can be present for both.
- [ ] **r/MachineLearning** - only if the Reddit post lands well; frame it as tooling, not
      as a paper.
- [ ] **llama.cpp Discord** - share in the appropriate channel with a one-line description
      and your own report card. Do not cross-post the full Reddit text.
- [ ] **Blog post** - `docs/BLOG.md`, published on manunicholasjacob.com, then cross-posted
      to LinkedIn (draft in `announce-linkedin.md`).
- [ ] **Awesome lists** - open PRs adding the tool to `awesome-local-llm` style lists and
      any `awesome-llama.cpp` equivalent. One PR per list, each with a one-line entry that
      says what the tool does, not how great it is.

## Phase 5 - track adoption

Screenshot and date-stamp each of these for the evidence vault
(`PROFILE-BUILDING` / `EB1_O1_CRITERIA_MAP`):

- [ ] GitHub stars, forks, and unique clones (Insights > Traffic; this data only goes back
      14 days, so capture it on a schedule, not once).
- [ ] PyPI download counts (pypistats.org).
- [ ] Zenodo views and downloads.
- [ ] The Reddit and HN threads themselves, including notable comments.
- [ ] Any third-party mention: a blog post, an issue that says "I ran this and", a
      citation, an inclusion in someone else's benchmark writeup.
- [ ] Reports contributed by other people to the results gallery. These are the strongest
      evidence: someone ran your tool on hardware you have never touched.

## Rules for the posts

- Every number in a post must come from a run you can reproduce. No rounding up.
- Say what the tool does not do. The caveats section in the report card is a feature;
  repeat it in the posts.
- Credit llama.cpp explicitly and early. The tool is worthless without it.
- Do not use em dashes.
- No AI-generated-sounding prose. Short sentences, concrete numbers, no adjective stacks.
- If someone finds a bug, fix it and say so in the thread. That is the best possible
  advertisement.
