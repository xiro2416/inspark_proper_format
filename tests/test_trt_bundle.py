"""CPU-only checks for the exact-shape, four-component bundle contract."""
import hashlib
import json
from pathlib import Path

import pytest

from inspark_infer.build.trt113 import (
    PROFILE, _engine_record, parse_batches, unsupported, validate_bundle,
)
from inspark_infer.build import trt113
from inspark_infer.build.hf_cache import _attestation, _remote_path, _repo_id


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_exact_batch_and_unsupported_handoff():
    assert parse_batches("1,4,8") == (1, 4, 8)
    for value in ("", "1,1", "1,3x", "0"):
        with pytest.raises(ValueError):
            parse_batches(value)
    result = unsupported({"sm": 120, "name": "future"}, PROFILE, (3,))
    assert result["status"] == "unsupported"
    assert len(result["reasons"]) == 2
    assert "codex_task" in result


def test_preflight_never_needs_models_or_trt(monkeypatch, capsys):
    monkeypatch.setattr(trt113, "gpu_info", lambda index: {"physical_gpu": index,
                        "sm": 89, "name": "GPU", "memory_total_mib": 49140,
                        "memory_used_mib": 0})
    assert trt113.main(["--preflight-only", "--gpu", "6", "--batches", "1,4"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "supported_build_candidate"
    assert trt113.main(["--preflight-only", "--gpu", "6", "--batches", "3"]) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "unsupported"


def test_bundle_inventory_requires_all_components_and_exact_hashes(tmp_path):
    root = tmp_path / "bundle"
    root.mkdir()
    components = {}
    files = {}
    for name in ("target", "draft", "cfm", "vocoder"):
        child = root / name
        child.mkdir()
        engine = child / "engine"
        engine.write_bytes(name.encode())
        plan = child / "plan.json"
        plan.write_text("{}")
        components[name] = {"engine": f"{name}/engine", "engine_sha256": digest(engine),
                            "plan": f"{name}/plan.json", "plan_sha256": digest(plan)}
        files[f"{name}/engine"] = digest(engine)
        files[f"{name}/plan.json"] = digest(plan)
    deployment = root / "deployment.json"
    deployment.write_text("{}")
    files["deployment.json"] = digest(deployment)
    route = root / "route_report.json"
    route.write_text("{}")
    files["route_report.json"] = digest(route)
    identity = {"schema": 1, "model": "indextts2", "profile": PROFILE, "batch": 4,
                "precision_policy": "mixed_bf16_fp32", "quantization": "none",
                "hardware": {"name": "GPU", "sm": 89, "memory_total_mib": 49140},
                "trt": "11.3.0", "source_sha256": "source",
                "engines": {name: row["engine_sha256"] for name, row in components.items()}}
    bundle_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    manifest = {**identity, "bundle_id": bundle_id,
                "components": components, "deployment": "deployment.json",
                "deployment_sha256": digest(deployment), "route_report": "route_report.json",
                "files": files}
    (root / "manifest.json").write_text(json.dumps(manifest))
    assert validate_bundle(root)["batch"] == 4
    manifest["certified_for_production"] = True
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Production certification"):
        validate_bundle(root)
    manifest["certified_for_production"] = False
    (root / "manifest.json").write_text(json.dumps(manifest))
    (root / "vocoder/engine").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash"):
        validate_bundle(root)
    (root / "vocoder/engine").write_bytes(b"vocoder")
    manifest["components"].pop("vocoder")
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="four components"):
        validate_bundle(root)


def test_engine_record_enforces_shape_and_hardware(tmp_path):
    child = tmp_path / "cfm"
    child.mkdir()
    engine = child / "solver.engine"
    engine.write_bytes(b"engine")
    plan = {"engine": "solver.engine", "sha256": digest(engine), "batch": 1,
            "frames": 310, "sm": 89, "trt": "11.3.0", "gpu_name": "GPU",
            "tensors": [{"name": "x", "shape": [1, 80, 310]}],
            "provenance": {"status": "recorded_not_audited",
                           "source": {"source_sha256": "abc"},
                           "model_sources": [{"role": "checkpoint", "sha256": "weight"}]}}
    (child / "plan.json").write_text(json.dumps(plan))
    assert _engine_record(tmp_path, "cfm", 1, {"sm": 89, "name": "GPU"})["engine"] == "cfm/solver.engine"
    for field, value in (("frames", 309), ("sm", 120), ("trt", "10.0")):
        bad = dict(plan, **{field: value})
        (child / "plan.json").write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            _engine_record(tmp_path, "cfm", 1, {"sm": 89, "name": "GPU"})


def test_private_cache_paths_and_attestation(tmp_path):
    assert _repo_id("user/private-model") == "user/private-model"
    assert _remote_path("bundles/sm89/b1") == "bundles/sm89/b1"
    for path in ("../bundle", "bundles/../secret", "/bundles/x", "other/x"):
        with pytest.raises(ValueError):
            _remote_path(path)
    with pytest.raises(ValueError):
        _repo_id("https://huggingface.co/user/repo")
    attestation = tmp_path / "attestation.json"
    attestation.write_text(json.dumps({"schema": 1, "reviewed": True,
                                       "redistribution_permitted": True, "sources": {}}))
    with pytest.raises(ValueError, match="every pinned source"):
        _attestation(attestation)
