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


def test_a_row_claiming_streamed_bytes_has_them(matrix):
    # The 1.5B rows normalised by file size until the artifacts were parsed. Either
    # convention is defensible; claiming one and carrying the other is not.
    for r in matrix:
        if r["normalization"] == "streamed_bytes":
            assert r["streamed_MiB"], r
            assert r["streamed_GBs"], r
        else:
            assert r["streamed_MiB"] is None, r


def test_both_scales_are_now_streamed_normalised(matrix):
    # Comparing a streamed-byte number at one scale against a file-byte number at another
    # is the mistake this makes impossible.
    assert {r["normalization"] for r in matrix} == {"streamed_bytes"}


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


# ------------------------------------------------------------------ the comparison plan

def test_plan_refuses_an_already_quantized_source():
    # Building the comparison set from a quantized file needs --allow-requantize and
    # produces artifacts that share a label with a proper set while holding different
    # types. That is the exact confound the shipped study spent an arm measuring.
    text = "\n".join(advisor.quantize_plan("models/thing-Q4_K_M.gguf", "Q4_K_M"))
    assert "WRONG STARTING POINT" in text
    assert "--allow-requantize" in text
    assert "38%" in text
    assert "llama-quantize models" not in text  # no commands offered


def test_plan_from_an_unquantized_source_gives_runnable_commands():
    text = "\n".join(advisor.quantize_plan("/models/qwen-fp16.gguf", "F16"))
    for fmt in advisor.PLAN_FORMATS:
        assert f".gguf {fmt}" in text, fmt
    assert "advise --measure" in text
    assert advisor.GALLERY_URL in text
    text.encode("ascii")


def test_plan_strips_the_precision_suffix_from_the_output_names():
    text = "\n".join(advisor.quantize_plan("/m/qwen0.5b-fp16.gguf", "F16"))
    assert "qwen0.5b-Q4_0.gguf" in text
    assert "qwen0.5b-fp16-Q4_0.gguf" not in text


def test_plan_keeps_the_separator_style_it_was_given():
    posix = "\n".join(advisor.quantize_plan("/home/me/models/m-f16.gguf", "F16"))
    assert "/home/me/models/m-Q4_0.gguf" in posix
    assert "\\" not in posix


def test_a_size_only_comparison_says_so_instead_of_implying_a_format_result():
    measured = [
        {"format": "Q4_0", "name": "small", "threads": 4, "tok_s": 100.0,
         "streamed_MiB": 330, "streamed_GBs": 34.6},
        {"format": "Q4_0", "name": "large", "threads": 4, "tok_s": 50.0,
         "streamed_MiB": 660, "streamed_GBs": 34.6},
    ]
    text = advisor.render_measured({"cpu": "test", "logical_cores": 8}, measured, [4])
    assert "SIZE COMPARISON, NOT A FORMAT COMPARISON" in text
    assert "advise --plan" in text


def test_a_matched_byte_comparison_does_not_get_the_size_warning():
    measured = [
        {"format": "Q4_0", "name": "a", "threads": 4, "tok_s": 100.0,
         "streamed_MiB": 330, "streamed_GBs": 34.6},
        {"format": "IQ4_XS", "name": "b", "threads": 4, "tok_s": 90.0,
         "streamed_MiB": 331, "streamed_GBs": 31.2},
    ]
    text = advisor.render_measured({"cpu": "test", "logical_cores": 8}, measured, [4])
    assert "SIZE COMPARISON" not in text
    assert "Same bytes:" in text


def test_the_measured_report_says_where_to_send_the_result():
    measured = [
        {"format": "Q4_0", "name": "a", "threads": 4, "tok_s": 100.0,
         "streamed_MiB": 330, "streamed_GBs": 34.6},
        {"format": "IQ4_XS", "name": "b", "threads": 4, "tok_s": 90.0,
         "streamed_MiB": 331, "streamed_GBs": 31.2},
    ]
    text = advisor.render_measured({"cpu": "test", "logical_cores": 8}, measured, [4])
    assert advisor.GALLERY_URL in text


# ---------------------------------------------------- what is actually in the files

def test_every_row_says_how_much_of_the_file_is_the_format_it_names(matrix):
    for r in matrix:
        assert r["on_label_pct"] is not None, r
        assert 0.0 <= r["on_label_pct"] <= 100.0, r
        assert r["bits_per_weight"] and 1.0 < r["bits_per_weight"] < 33.0, r


def test_the_on_label_share_is_a_property_of_the_file_not_the_core(matrix):
    by_file = {}
    for r in matrix:
        by_file.setdefault((r["params_b"], r["format"]), set()).add(r["on_label_pct"])
    for key, values in by_file.items():
        assert len(values) == 1, (key, values)


def test_the_small_model_fallback_is_in_the_data(matrix):
    # The finding that makes the whole comparison contestable, and therefore the one that
    # has to survive a regenerated matrix. At 0.5B most of these files are mostly not the
    # format they name; at 1.5B the same recipes are mostly on-label.
    def on_label(scale, fmt):
        return next(r["on_label_pct"] for r in matrix
                    if r["params_b"] == scale and r["format"] == fmt)

    assert on_label(0.5, "Q3_K_M") == 0.0
    assert on_label(0.5, "Q2_K") == 0.0
    assert on_label(0.5, "Q4_K_M") < 15
    assert on_label(1.5, "Q4_K_M") > 75
    assert on_label(0.5, "Q4_0") == 100.0


def test_the_default_format_stores_more_bits_than_its_name_implies_at_small_scale(matrix):
    small = next(r for r in matrix
                 if r["params_b"] == 0.5 and r["format"] == "Q4_K_M")
    large = next(r for r in matrix
                 if r["params_b"] == 1.5 and r["format"] == "Q4_K_M")
    assert small["bits_per_weight"] > 5.4      # nominal is 4.5
    assert large["bits_per_weight"] < small["bits_per_weight"]


def test_the_report_warns_when_the_files_are_mostly_not_what_they_say(matrix):
    adv = advisor.advise_core(matrix, "cortex-a76", params_b=0.5, threads=2)
    formats = [f for f, _, _ in adv["mislabelled"]]
    assert "Q3_K_M" in formats and "Q4_K_M" in formats
    text = "\n".join(advisor.render_core(adv))
    assert "Read this table knowing what is in the files" in " ".join(text.split())
    assert "contains no Q3_K tensors" in " ".join(text.split())


def test_the_warning_goes_quiet_where_the_files_are_honest(matrix):
    adv = advisor.advise_core(matrix, "cortex-a76", params_b=1.5)
    assert [f for f, _, _ in adv["mislabelled"]] == ["Q3_K_M"]


def test_the_label_finding_leads_the_cross_core_facts(matrix):
    first = advisor.cross_core_facts(matrix)[0]
    assert first.startswith("The label does not describe the file")
    assert "only 3 contain the type their name claims" in first
