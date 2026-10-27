"""Tests for the foreign-CPU check.

The case that matters is the one the stability check cannot see: competition that lasts for
the whole measurement, so every repetition is depressed equally and stability looks fine.
That is what happened on a Galaxy M21 with Android's background dexopt job running, where
the ceiling read 32% low at stability 0.85. test_constant_contention_is_caught_where_
stability_would_pass reproduces that signature.
"""

from __future__ import annotations

import os

import pytest

from llama_roofline import contention


def snap(monotonic: float, total: int, own: int, hz: int = 100, ncpu: int = 8) -> dict:
    return {"hz": hz, "total_busy_ticks": total, "own_ticks": own,
            "monotonic": monotonic, "ncpu": ncpu}


# ---------------------------------------------------------------- foreign_fraction


def test_idle_machine_reports_no_foreign_use():
    # 1 second, 8 cores at 100 Hz = 800 ticks of capacity. All the busy time is ours.
    before = snap(0.0, total=1000, own=500)
    after = snap(1.0, total=1800, own=1300)
    assert contention.foreign_fraction(before, after) == pytest.approx(0.0)


def test_half_the_machine_taken_by_others():
    # 400 of 800 ticks of capacity went to somebody else.
    before = snap(0.0, total=0, own=0)
    after = snap(1.0, total=400, own=0)
    assert contention.foreign_fraction(before, after) == pytest.approx(0.5)


def test_fraction_is_capped_at_one():
    # Counters can overshoot capacity when the sampling interval is estimated; a number
    # above 1.0 would be reported as "140% of the machine", which is not a useful thing
    # to tell anyone.
    before = snap(0.0, total=0, own=0)
    after = snap(1.0, total=5000, own=0)
    assert contention.foreign_fraction(before, after) == 1.0


def test_negative_delta_is_floored_not_propagated():
    # /proc/stat and /proc/self/stat are read a moment apart, so our own usage can appear
    # to exceed the total. That is rounding, not a machine where others used negative CPU.
    before = snap(0.0, total=100, own=0)
    after = snap(1.0, total=150, own=80)
    assert contention.foreign_fraction(before, after) == 0.0


def test_missing_snapshot_returns_none_not_zero():
    # None means "could not check". Zero would mean "checked, and the machine was clean",
    # which is the one thing an unsupported platform must not claim.
    assert contention.foreign_fraction(None, snap(1.0, 1, 1)) is None
    assert contention.foreign_fraction(snap(0.0, 1, 1), None) is None
    assert contention.foreign_fraction(None, None) is None


def test_zero_length_window_returns_none():
    assert contention.foreign_fraction(snap(1.0, 0, 0), snap(1.0, 400, 0)) is None


def test_more_cores_means_the_same_foreign_ticks_matter_less():
    before = snap(0.0, total=0, own=0, ncpu=2)
    after = snap(1.0, total=100, own=0, ncpu=2)
    two_cores = contention.foreign_fraction(before, after)
    before16 = snap(0.0, total=0, own=0, ncpu=16)
    after16 = snap(1.0, total=100, own=0, ncpu=16)
    assert two_cores > contention.foreign_fraction(before16, after16)


# ---------------------------------------------------------------- describe


def test_constant_contention_is_caught_where_stability_would_pass():
    # The Android dexopt case: about 60% of an 8-core phone taken by five compiler threads
    # for the whole run. Stability was 0.85, so the existing check passed it.
    msg = contention.describe(0.6, [("dex2oat", 500.0), ("Compiler driver", 480.0)])
    assert msg is not None
    assert "60%" in msg
    assert "dex2oat" in msg
    # It must say why a good stability figure is not reassurance, or the reader will trust
    # the stability number and discard this one.
    assert "does not show up as instability" in msg


def test_warning_does_not_claim_the_ceiling_is_definitely_wrong():
    # Measured on a Pi 5: two CPU-only spinners take the foreign fraction to 0.50 and leave
    # the ceiling at 13.6 GB/s, unchanged, because nothing else touched memory. The warning
    # has to survive that case without having lied, so it says "unreliable" and not "too low".
    msg = contention.describe(0.5, [("python3", 400.0)])
    assert msg is not None
    assert "unreliable" in msg
    assert "too low" not in msg.replace("measured 32% low", "")
    # and it should tell the reader what to do instead of leaving them with a scary number
    assert "pass --peak-bw" in msg


def test_quiet_machine_produces_no_warning():
    assert contention.describe(0.01) is None
    assert contention.describe(0.0, []) is None


def test_unsupported_platform_produces_no_warning():
    assert contention.describe(None) is None


def test_threshold_boundary_is_inclusive():
    assert contention.describe(contention.CONTENTION_THRESHOLD - 1e-9) is None
    assert contention.describe(contention.CONTENTION_THRESHOLD) is not None


def test_describe_without_offender_names_still_warns():
    msg = contention.describe(0.4, [])
    assert msg is not None and "busiest" not in msg


# ---------------------------------------------------------------- top_foreign


def test_top_foreign_ranks_by_cpu_used_during_the_window():
    before = {1: ("quiet", 100), 2: ("busy", 100), 3: ("medium", 100)}
    after = {1: ("quiet", 101), 2: ("busy", 900), 3: ("medium", 400)}
    assert [n for n, _ in contention.top_foreign(before, after)] == ["busy", "medium", "quiet"]


def test_top_foreign_respects_the_limit():
    before = {i: (f"p{i}", 0) for i in range(10)}
    after = {i: (f"p{i}", 100 * (10 - i)) for i in range(10)}
    assert len(contention.top_foreign(before, after, limit=2)) == 2


def test_a_process_that_started_mid_window_counts_all_its_usage():
    # It was not there before and it was busy after, so all of its time is foreign time.
    assert contention.top_foreign({}, {7: ("latecomer", 300)}) == [("latecomer", 300.0)]


def test_a_recycled_pid_is_not_credited_with_the_previous_owners_time():
    # Same pid, different name: treat it as new rather than computing a bogus delta, which
    # for a reused pid with a lower tick count would otherwise go negative and vanish.
    before = {7: ("old-process", 5000)}
    after = {7: ("new-process", 40)}
    assert contention.top_foreign(before, after) == [("new-process", 40.0)]


def test_a_process_that_exited_is_simply_absent():
    assert contention.top_foreign({1: ("gone", 100)}, {}) == []


def test_top_foreign_handles_missing_tables():
    assert contention.top_foreign(None, {1: ("x", 1)}) == []
    assert contention.top_foreign({1: ("x", 1)}, None) == []


# ---------------------------------------------------------------- live /proc reads


@pytest.mark.skipif(not os.path.exists("/proc/stat"), reason="needs Linux-style /proc")
def test_snapshot_on_this_machine_is_self_consistent():
    first = contention.snapshot()
    assert first is not None
    assert first["hz"] > 0 and first["ncpu"] >= 1
    # Burn a little CPU so both counters have somewhere to go.
    total = 0
    for i in range(400000):
        total += i
    second = contention.snapshot()
    assert second is not None
    assert second["monotonic"] > first["monotonic"]
    assert second["total_busy_ticks"] >= first["total_busy_ticks"]
    assert second["own_ticks"] >= first["own_ticks"]
    frac = contention.foreign_fraction(first, second)
    assert frac is not None and 0.0 <= frac <= 1.0


@pytest.mark.skipif(not os.path.exists("/proc/stat"), reason="needs Linux-style /proc")
def test_process_table_finds_processes_and_excludes_us():
    table = contention.process_table()
    assert table is not None and len(table) > 0
    assert os.getpid() not in table
    for name, ticks in table.values():
        assert isinstance(name, str) and ticks >= 0


def test_snapshot_returns_none_when_proc_is_unavailable(monkeypatch):
    # Windows and macOS take this path, and the contract is None rather than a fake zero.
    monkeypatch.setattr(contention, "_total_busy_ticks", lambda: None)
    assert contention.snapshot() is None
    monkeypatch.setattr(contention, "_total_busy_ticks", lambda: 10)
    monkeypatch.setattr(contention, "_own_ticks", lambda: None)
    assert contention.snapshot() is None


def test_own_ticks_parses_an_executable_name_containing_spaces_and_brackets(tmp_path,
                                                                           monkeypatch):
    # A comm of "(my prog) (x)" breaks any parser that splits the whole line on spaces or
    # finds the first ')'. Fields 14 and 15 here are 111 and 222.
    fake = tmp_path / "stat"
    tail = " ".join(str(i) for i in range(3, 14))  # fields 3..13
    fake.write_text(f"42 (my prog) (x)) {tail} 111 222 0 0 20 0 1 0 0\n")

    real_open = open

    def fake_open(path, *a, **kw):
        if path == "/proc/self/stat":
            return real_open(str(fake), *a, **kw)
        return real_open(path, *a, **kw)

    monkeypatch.setattr("builtins.open", fake_open)
    assert contention._own_ticks() == 333


def test_total_busy_excludes_idle_and_iowait(tmp_path, monkeypatch):
    fake = tmp_path / "stat"
    # user nice system idle iowait irq softirq steal
    fake.write_text("cpu  10 20 30 1000 500 1 2 3\ncpu0 1 1 1 1 1 1 1 1\n")

    real_open = open

    def fake_open(path, *a, **kw):
        if path == "/proc/stat":
            return real_open(str(fake), *a, **kw)
        return real_open(path, *a, **kw)

    monkeypatch.setattr("builtins.open", fake_open)
    # 10+20+30+1+2+3 = 66, with idle 1000 and iowait 500 left out.
    assert contention._total_busy_ticks() == 66


def test_total_busy_includes_nice_time(tmp_path, monkeypatch):
    # The Android job ran at nice 10. A nice'd process yields the CPU but not memory
    # bandwidth, so excluding nice time would miss exactly the case this module exists for.
    fake = tmp_path / "stat"
    fake.write_text("cpu  0 400 0 0 0 0 0 0\n")

    real_open = open

    def fake_open(path, *a, **kw):
        if path == "/proc/stat":
            return real_open(str(fake), *a, **kw)
        return real_open(path, *a, **kw)

    monkeypatch.setattr("builtins.open", fake_open)
    assert contention._total_busy_ticks() == 400


def test_malformed_proc_stat_returns_none(tmp_path, monkeypatch):
    fake = tmp_path / "stat"
    fake.write_text("intr 1 2 3\n")

    real_open = open

    def fake_open(path, *a, **kw):
        if path == "/proc/stat":
            return real_open(str(fake), *a, **kw)
        return real_open(path, *a, **kw)

    monkeypatch.setattr("builtins.open", fake_open)
    assert contention._total_busy_ticks() is None
