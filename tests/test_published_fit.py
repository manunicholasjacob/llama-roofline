"""The one external correctness check this tool has.

The analysis is portable arithmetic over someone else's measurements, so there is no
ground truth inside the repository to test it against. There is one outside it: a
published study of LLM decode on a Raspberry Pi 5 reported an effective bandwidth of
10.69 GB/s at R^2 = 0.9800 over seven models. `examples/rpi5-cortex-a76/roofline.json`
holds that study's raw llama-bench numbers in this tool's schema, and re-analysing them
through `roofline.analyze` has to return the published figures.

If a change to the analysis moves these numbers, either the change is wrong or the
published result was, and neither should pass quietly.
"""

import json
import os

import pytest

from llama_roofline import roofline

EXAMPLES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "examples")

# study, effective GB/s, R^2, models in the fit
PUBLISHED = [
    ("rpi5-cortex-a76", 10.69, 0.9800, 7),
    ("x86-i7-12700H", 37.65, 0.9874, 7),
]


def _load(name):
    path = os.path.join(EXAMPLES, name, "roofline.json")
    if not os.path.isfile(path):
        pytest.skip(f"{name} example not present")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@pytest.mark.parametrize("name,bw,r2,n", PUBLISHED)
def test_reanalysis_returns_the_published_fit(name, bw, r2, n):
    results = _load(name)
    stored = results["analysis"]
    fresh = roofline.analyze(stored["models"], stored.get("peak_read_GBs"))
    assert fresh["fit"]["n_points"] == n
    assert round(fresh["fit"]["bw_eff_GBs"], 2) == bw
    assert round(fresh["fit"]["r2"], 4) == r2


@pytest.mark.parametrize("name,bw,r2,n", PUBLISHED)
def test_the_shipped_example_still_says_what_it_said(name, bw, r2, n):
    # Catches an example file edited by hand as well as an analysis that drifted.
    fit = _load(name)["analysis"]["fit"]
    assert round(fit["bw_eff_GBs"], 2) == bw
    assert round(fit["r2"], 4) == r2


def test_the_examples_use_the_resident_size_convention():
    # `diagnose` counts streamed bytes, which is the more accurate number and a different
    # one. The published fits were computed the other way, so the examples and `run` have
    # to stay on the old convention or the reproduction above stops meaning anything.
    for name, *_ in PUBLISHED:
        for model in _load(name)["analysis"]["models"]:
            source = model.get("bytes_source", "")
            assert not source.startswith("streamed"), (name, model["name"], source)


def test_every_renderer_states_the_operating_point():
    """A fitted bandwidth from this tool is not comparable without it.

    The same seven models on the same laptop give 35.73 GB/s at a fixed thread count and
    37.65 here, because this fit takes each model at its own best thread count and so sits
    on the upper envelope. Both numbers are right. A surface that prints one without
    saying which is inviting somebody to conclude the two disagree.
    """
    from llama_roofline import report

    results = _load("x86-i7-12700H")
    for renderer in (report.render, report.render_markdown,
                     report.render_diagnosis, report.render_diagnosis_markdown):
        text = " ".join(renderer(results).split())
        assert "37.65" in text, renderer.__name__
        assert report.FIT_OPERATING_POINT in text, renderer.__name__


def test_the_operating_point_claim_is_true_of_the_shipped_example():
    # Every model in the example is reported at whichever thread count was fastest for it,
    # and they are not all the same, which is the whole point of the qualifier.
    models = [m for m in _load("x86-i7-12700H")["analysis"]["models"] if m.get("ok")]
    threads = {m["decode_threads"] for m in models}
    assert len(threads) > 1, threads
    for m in models:
        best = max(r["decode_ts"] for r in m["runs"] if r.get("decode_ts"))
        assert m["decode_ts"] == best, m["name"]
