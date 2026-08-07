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
- [x] Add the software DOI to ORCID (0009-0007-6589-6572). Added via Works > Add >
      "Add work with a DOI", which pulls the metadata from DataCite rather than
      hand-typing it. Registered as type Software, visibility Everyone.
- [ ] Optional: submit the record to a relevant Zenodo community.

## Phase 3 - PyPI  (DECIDED AGAINST, 2026-08-07)

**Not shipping to PyPI.** PyPI has mandated two-factor authentication on every account
since 2024, and without it you cannot create an API token, so you cannot upload at all.
That was not a tradeoff worth making for this project, so the install path is the
repository:

```
pip install git+https://github.com/manunicholasjacob/llama-roofline
```

Every install command in the README, the blog and the announcement drafts already says
this, so nothing gates the launch.

The packaging is still correct and PyPI-ready if you change your mind later: `pyproject.toml`
is complete, `python -m build` produces a wheel and sdist, and both pass `twine check`.
To do it then:

- [ ] Enable 2FA on PyPI (authenticator app), create an account-scoped API token.
- [ ] Put the token in `~/.pypirc` under `[pypi]` with `username = __token__`.
- [ ] `python -m build`, `twine check dist/*`, upload to TestPyPI first, then PyPI.
- [ ] Add the badge back:
      `[![PyPI](https://img.shields.io/pypi/v/llama-roofline.svg)](https://pypi.org/project/llama-roofline/)`
- [ ] Switch the install commands in README, `docs/BLOG.md` and `docs/announce-reddit.md`
      from the git URL to `pip install llama-roofline`.

Consequence for Phase 5: there are no PyPI download numbers to track. GitHub clones and
Zenodo downloads are the adoption evidence instead.

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
- [ ] Zenodo record views and downloads (`10.5281/zenodo.21842493`). With no PyPI, this
      and GitHub traffic are the install-side evidence.
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
