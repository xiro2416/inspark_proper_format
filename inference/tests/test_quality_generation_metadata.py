from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import generate_quality_corpus as generator


def test_host_and_device_native_target_counters_are_summed_once():
    target = NS(native_full_steps=17, stats=lambda: {"native_full_steps": 17})
    draft = NS(native_full_steps=29, stats=lambda: {})
    engine = NS(rt=NS(target=target, backbone=draft, native_target_steps=11, device_target_steps=13),
                device_round_attempts=3, device_round_successes=2, device_round_fallbacks=1)
    counters = generator.runtime_counters(engine)
    assert counters["native_target_steps"] == 28
    assert counters["native_target_host_steps"] == 17
    assert counters["native_target_device_steps"] == 11
    assert counters["native_draft_steps"] == 29


def test_preflight_is_the_actual_lease_snapshot_without_querying_gpu():
    result = generator.gpu_preflight(NS(index="6", initial_memory_mib=7200,
                                        initial_utilization=0, shared=True))
    assert result["physical_gpu"] == 6 and result["existing_memory_mib"] == 7200
    assert result["explicit_shared_run"] and result["external_processes_preserved"]


def test_admission_emotions_are_independent_exact_copies():
    case = {"id": "request", "emotion": [0., .1, .2, .3, .4, .5, .6, .7]}
    result = generator.admission_emotions([case])
    assert result["request"] == case["emotion"]
    case["emotion"][0] = 1.
    assert result["request"][0] == 0.


@pytest.mark.parametrize("verified", [False, True])
def test_native_identity_reuses_actual_paths_and_never_promotes_legacy(monkeypatch, verified):
    import audit_real_acoustics
    import validate_trt113_ar
    observed = []
    acoustic = {component: {"has_native_engine": True, "weight_identity_verified": verified,
                            "provenance_status": "recorded_not_audited" if verified else "legacy_unverified"}
                for component in ("cfm", "vocoder")}
    monkeypatch.setattr(audit_real_acoustics, "capture_acoustic_engine_evidence", lambda *args: acoustic)
    def evidence(path, batch, component, paths):
        observed.append((path, batch, component, paths))
        return {"sha256": "engine-sha", "weight_identity_verified": verified}
    monkeypatch.setattr(validate_trt113_ar, "engine_evidence", evidence)
    bank = NS(artifacts={4: {"path": "actual.engine", "sha256": "engine-sha"}})
    engine = NS(rt=NS(target=NS(native_full_bank=bank), backbone=NS(native_full_bank=bank)))
    models = {component: {"model_sources": [{"role": component + "_checkpoint", "path": "/workspace/actual-" + component}]}
              for component in ("target", "draft")}
    result = generator.native_engine_evidence(engine, models)
    assert all(row["weight_identity_verified"] is verified for row in result.values())
    assert observed[0][3] == {"target_checkpoint": "/workspace/actual-target"}
    assert result["target"]["engines"]["4"]["sha256"] == "engine-sha"
    bank.artifacts[4]["sha256"] = "different-loaded-engine"
    with pytest.raises(ValueError, match="already loaded"):
        generator.native_engine_evidence(engine, models)


def test_eager_has_no_native_engine_identity_instead_of_vacuous_verified(monkeypatch):
    import audit_real_acoustics
    monkeypatch.setattr(audit_real_acoustics, "capture_acoustic_engine_evidence", lambda *args: {
        name: {"has_native_engine": False, "weight_identity_verified": False} for name in ("cfm", "vocoder")})
    engine = NS(rt=NS(target=NS(), backbone=NS()))
    result = generator.native_engine_evidence(engine, {})
    assert set(result) == {"target", "draft", "cfm", "vocoder"}
    assert all(row["provenance_status"] == "not_applicable" and row["weight_identity_verified"] is None
               for row in result.values())
