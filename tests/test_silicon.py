"""Device detection: does it name the right core, and does it stay quiet otherwise."""

import llama_roofline.silicon as silicon


def _cores(det):
    return [(c["core"], c["confidence"]) for c in det["cores"]]


def test_the_measured_laptop_is_recognised_exactly():
    det = silicon.detect(cpu_override="12th Gen Intel(R) Core(TM) i7-12700H")
    assert _cores(det) == [("golden-cove", "exact"), ("gracemont", "exact")]
    assert det["hybrid"]


def test_another_alder_lake_sku_is_same_uarch_not_exact():
    det = silicon.detect(cpu_override="Intel(R) Core(TM) i5-12400")
    assert all(conf == "same-uarch" for _, conf in _cores(det))


def test_raptor_lake_p_core_is_extrapolation_but_its_e_core_is_not():
    det = silicon.detect(cpu_override="13th Gen Intel(R) Core(TM) i9-13900K")
    by_core = dict(_cores(det))
    assert by_core["golden-cove"] == "related"
    assert by_core["gracemont"] == "same-uarch"


def test_core_ultra_is_not_claimed_as_alder_lake():
    # Meteor Lake onward is hybrid, but with different cores. Claiming the Golden Cove
    # ranking for it would be the exact mistake this tool exists to argue against.
    det = silicon.detect(cpu_override="Intel(R) Core(TM) Ultra 7 155H")
    assert det["cores"] == []
    assert "Redwood Cove" in (det["detail"] or "")


def test_pre_hybrid_intel_gets_nothing():
    det = silicon.detect(cpu_override="11th Gen Intel(R) Core(TM) i7-1165G7")
    assert det["cores"] == []


def test_amd_gets_nothing():
    det = silicon.detect(cpu_override="AMD Ryzen 7 5800X")
    assert det["cores"] == []


def test_arm_parts_table_covers_the_measured_core():
    assert silicon.ARM_PARTS[0xd0b] == "Cortex-A76"


def test_a76_relatives_exclude_the_little_cores():
    # A55 and A510 are in-order designs. Their behaviour is not the A76's, and pretending
    # otherwise would put a confident number in front of a phone user who deserves better.
    for little in ("Cortex-A55", "Cortex-A510", "Cortex-A520", "Cortex-A53"):
        assert little not in silicon.A76_RELATIVES


def test_summary_line_survives_missing_fields():
    line = silicon.summary_line({"cpu": None, "logical_cores": None, "ram_bytes": None})
    assert "unknown CPU" in line


def test_intel_generation_parsing():
    assert silicon._intel_generation("12th Gen Intel(R) Core(TM) i7-12700H") == 12
    assert silicon._intel_generation("Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz") == 9
    assert silicon._intel_generation("Intel(R) Core(TM) i5-12400") == 12
    assert silicon._intel_generation("AMD Ryzen 7 5800X") is None
