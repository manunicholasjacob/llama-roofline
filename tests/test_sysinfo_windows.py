"""Windows CPU naming without wmic, which Windows 11 24H2 no longer ships."""

from llama_roofline import silicon, sysinfo


def test_registry_brand_string_wins_over_processor_identifier(monkeypatch):
    monkeypatch.setattr(sysinfo.platform, "system", lambda: "Windows")
    monkeypatch.setattr(sysinfo, "_windows_cpu_name",
                        lambda: "12th Gen Intel(R) Core(TM) i7-12700H")
    monkeypatch.setenv("PROCESSOR_IDENTIFIER",
                       "Intel64 Family 6 Model 154 Stepping 3, GenuineIntel")
    assert sysinfo.cpu_model() == "12th Gen Intel(R) Core(TM) i7-12700H"


def test_without_registry_or_wmic_it_degrades_to_the_identifier(monkeypatch):
    monkeypatch.setattr(sysinfo.platform, "system", lambda: "Windows")
    monkeypatch.setattr(sysinfo, "_windows_cpu_name", lambda: None)
    monkeypatch.setattr(sysinfo, "_run", lambda *a, **k: None)
    monkeypatch.setenv("PROCESSOR_IDENTIFIER", "Intel64 Family 6 Model 154")
    assert sysinfo.cpu_model() == "Intel64 Family 6 Model 154"


def test_physical_cores_falls_back_to_cim_when_wmic_is_missing(monkeypatch):
    monkeypatch.setattr(sysinfo.platform, "system", lambda: "Windows")

    def fake_run(cmd, timeout=5.0):
        return None if cmd[0] == "wmic" else "14\r\n"

    monkeypatch.setattr(sysinfo, "_run", fake_run)
    assert sysinfo.physical_cores() == 14


def test_the_brand_string_is_what_core_detection_needs():
    det = silicon.detect("12th Gen Intel(R) Core(TM) i7-12700H")
    assert [c["core"] for c in det["cores"]] == ["golden-cove", "gracemont"]
