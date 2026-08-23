"""The quantization advisor, and the measurements it ships.

Two kinds of test here. The first kind checks the logic. The second kind pins numbers
from the study the matrix came from, so that a regenerated matrix that quietly disagrees
with the published result fails instead of shipping.
"""

import math

import pytest

import llama_roofline.advisor as advisor


@pytest.fixture(scope="module")
def matrix():
    return advisor.load_matrix()


# ------------------------------------------------------------------ the shipped data

def test_matrix_ships_and_parses(matrix):
    assert len(matrix) > 50
    assert advisor.cores(matrix) == ["cortex-a76", "golden-cove", "gracemont"]


def test_every_row_says_where_it_came_from(matrix):
    for r in matrix:
        assert r["basis"] in ("measured", "extrapolated")
        assert r["source"], f"row without a source: {r}"
        assert r["normalization"] in ("streamed_bytes", "file_bytes")


def test_energy_exists_only_where_a_rail_sensor_does(matrix):
    # The Pi 5 has a PMIC. The laptop runs Windows and reports no package power, so a
    # number there would have to be invented.
    for r in matrix:
        if r["mJ_per_token"] is not None:
            assert r["core"] == "cortex-a76"


def test_perplexity_is_a_property_of_the_file_not_the_core(matrix):
    by_format = {}
    for r in matrix:
        if r["perplexity"] is None:
            continue
        by_format.setdefault(r["format"], set()).add(r["perplexity"])
    for fmt, values in by_format.items():
        assert len(values) == 1, f"{fmt} has more than one perplexity: {values}"


def test_streamed_bandwidth_is_consistent_with_its_own_inputs(matrix):
    for r in matrix:
        if r["streamed_GBs"] and r["streamed_MiB"]:
            expected = r["tok_s"] * r["streamed_MiB"] * 1024 * 1024 / 1e9
            assert math.isclose(r["streamed_GBs"], expected, rel_tol=1e-3)


# --------------------------------------------------------- pinned published results

def test_a76_reproduces_the_published_table(matrix):
    rows = {r["format"]: r for r in advisor.rows_for(matrix, "cortex-a76", 0.5, 2)}
    # Table 1 of the study as printed. Compared with a tolerance rather than pinned
    # digit for digit, because the table is rounded for print and one entry rounds up
    # where the raw record rounds down.
    for fmt, tok_s, mJ in [("Q4_0", 34.7, 142), ("IQ4_XS", 34.2, 143),
                           ("IQ4_NL", 34.2, 147), ("Q3_K_M", 33.4, 147),
                           ("Q2_K", 31.8, 155), ("Q4_K_M", 25.7, 206),
                           ("Q6_K", 23.4, 204), ("Q8_0", 22.2, 216)]:
        assert abs(rows[fmt]["tok_s"] - tok_s) <= 0.1, fmt
        assert abs(rows[fmt]["mJ_per_token"] - mJ) <= 1.0, fmt


def test_the_default_format_is_off_the_envelope_on_every_core(matrix):
    # The headline finding. If a regenerated matrix stops showing it, that is either a
    # real change or a mistake, and either way it should not pass silently.
    for core in advisor.cores(matrix):
        threads = advisor.thread_counts(matrix, core, 0.5)[0]
        rows = advisor.rows_for(matrix, core, 0.5, threads)
        deficit = advisor.envelope_deficit(rows)
        assert deficit["Q4_K_M"] > 10, core


def test_a76_deficit_matches_the_published_sixteen_percent(matrix):
    rows = advisor.rows_for(matrix, "cortex-a76", 0.5, 2)
    assert round(advisor.envelope_deficit(rows)["Q4_K_M"]) == 16


def test_golden_cove_matched_byte_spread_matches_the_paper(matrix):
    spread = advisor.matched_byte_spread(matrix, "golden-cove", 0.5)
    assert round(spread[2]["spread_pct"]) == 25
    assert round(spread[6]["spread_pct"]) == 11


def test_the_a76_four_bit_class_converges(matrix):
    spread = advisor.matched_byte_spread(matrix, "cortex-a76", 0.5)
    assert spread[2]["spread_pct"] < 5


def test_the_winner_is_not_the_same_on_every_core(matrix):
    winners = {advisor.best_operating_point(matrix, c, 0.5)[1]["format"]
               for c in advisor.cores(matrix)}
    assert len(winners) > 1


# ---------------------------------------------------------------------- the logic

def test_matched_byte_group_holds_traffic_constant(matrix):
    rows = advisor.rows_for(matrix, "cortex-a76", 0.5, 2)
    group = advisor.matched_byte_group(rows)
    sizes = [r["streamed_MiB"] for r in group]
    assert max(sizes) / min(sizes) <= advisor.MATCHED_BYTE_TOL
    assert len(group) >= 3


def test_pareto_drops_a_format_that_is_both_slower_and_worse():
    rows = [
        {"format": "fast_good", "tok_s": 100.0, "perplexity": 20.0},
        {"format": "slow_bad", "tok_s": 50.0, "perplexity": 21.0},
        {"format": "slow_good", "tok_s": 40.0, "perplexity": 19.0},
    ]
    kept = {r["format"] for r in advisor.pareto(rows)}
    assert kept == {"fast_good", "slow_good"}


def test_quality_pick_breaks_a_perplexity_tie_on_energy(matrix):
    # IQ4_NL and IQ4_XS land 0.001 perplexity apart, which decides nothing. IQ4_XS uses
    # less energy, so it should win, and that matches the study's own recommendation.
    adv = advisor.advise_core(matrix, "cortex-a76", params_b=0.5, threads=2)
    assert adv["best_quality_at_speed"]["format"] == "IQ4_XS"


def test_advice_names_its_thread_count_and_the_ones_it_had(matrix):
    adv = advisor.advise_core(matrix, "cortex-a76", params_b=0.5, threads=3)
    assert adv["threads"] == 3
    assert adv["threads_available"] == [2, 3, 4]
    assert adv["thread_choice"] == "as requested"


def test_a_requested_thread_count_that_was_not_measured_snaps_and_says_so(matrix):
    adv = advisor.advise_core(matrix, "cortex-a76", params_b=0.5, threads=16)
    assert adv["threads"] == 4
    assert "nearest measured" in adv["thread_choice"]


def test_scale_snaps_to_a_measured_one(matrix):
    assert advisor.nearest_scale(matrix, "cortex-a76", 7.0) == 1.5
    assert advisor.nearest_scale(matrix, "cortex-a76", 0.6) == 0.5


def test_1_5b_rows_do_not_pretend_to_have_streamed_bytes(matrix):
    for r in matrix:
        if r["params_b"] == 1.5:
            assert r["normalization"] == "file_bytes"
            assert r["streamed_MiB"] is None


def test_missing_matrix_is_an_error_not_a_silent_empty(tmp_path):
    with pytest.raises(advisor.MatrixMissing):
        advisor.load_matrix(str(tmp_path / "nope.csv"))


# ---------------------------------------------------------------------- rendering

def test_report_states_the_evidence_level(matrix):
    adv = advisor.advise_core(matrix, "cortex-a76")
    text = "\n".join(advisor.render_core(adv, "related", "an A78 is not an A76"))
    assert "EXTRAPOLATED" in text


def test_unknown_silicon_gets_a_recipe_not_a_ranking(matrix):
    detection = {"cpu": "Some Unmeasured CPU", "cores": [], "logical_cores": 8}
    text = advisor.render(detection, [], matrix, "0.2.0")
    assert "NO MEASUREMENT FOR THIS SILICON" in text
    assert "advise --measure" in text
    # It must not print a table it cannot justify.
    assert "Fastest:" not in text


def test_reports_are_plain_ascii(matrix):
    detection = {"cpu": "x", "cores": [], "logical_cores": 4}
    adv = advisor.advise_core(matrix, "cortex-a76")
    for text in (advisor.render(detection, [], matrix, "0.2.0"),
                 advisor.render(detection, [({"core": "cortex-a76",
                                              "confidence": "exact", "why": ""}, adv)],
                                matrix, "0.2.0"),
                 advisor.render_markdown(detection, [({"core": "cortex-a76",
                                                       "confidence": "exact",
                                                       "why": ""}, adv)],
                                         matrix, "0.2.0")):
        text.encode("ascii")
        assert "—" not in text


def test_measured_report_reads_back_its_own_numbers():
    measured = [
        {"format": "Q4_0", "name": "a", "threads": 4, "tok_s": 100.0,
         "streamed_MiB": 330, "streamed_GBs": 34.6, "file_MiB": 409},
        {"format": "IQ4_XS", "name": "b", "threads": 4, "tok_s": 90.0,
         "streamed_MiB": 330, "streamed_GBs": 31.1, "file_MiB": 408},
    ]
    text = advisor.render_measured({"cpu": "test", "logical_cores": 8}, measured, [4])
    assert "Q4_0" in text and "11%" in text  # 100/90 - 1
    text.encode("ascii")


def test_measured_spread_needs_matched_bytes():
    measured = [
        {"format": "Q4_0", "name": "a", "threads": 4, "tok_s": 100.0,
         "streamed_MiB": 330, "streamed_GBs": 34.6},
        {"format": "Q8_0", "name": "b", "threads": 4, "tok_s": 50.0,
         "streamed_MiB": 660, "streamed_GBs": 34.6},
    ]
    assert advisor.local_matched_spread(measured) is None
