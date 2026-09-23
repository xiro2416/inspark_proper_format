"""CPU-only checks for the exact-shape, four-component bundle contract."""
import hashlib
import json
from pathlib import Path

import pytest

from inspark_infer.build.trt113 import (
    PROFILE, _deployment, _engine_record, parse_batches, unsupported, validate_bundle,
)
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
    manifest = {"schema": 1, "profile": PROFILE, "batch": 4, "bundle_id": "test",
                "components": components, "deployment": "deployment.json",
                "deployment_sha256": digest(deployment), "files": files}
    (root / "manifest.json").write_text(json.dumps(manifest))
    assert validate_bundle(root)["batch"] == 4
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
                           "source": {"source_sha256": "abc"}}}
    (child / "plan.json").write_text(json.dumps(plan))
    assert _engine_record(tmp_path, "cfm", 1, {"sm": 89})["engine"] == "cfm/solver.engine"
    for field, value in (("frames", 309), ("sm", 120), ("trt", "10.0")):
        bad = dict(plan, **{field: value})
        (child / "plan.json").write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            _engine_record(tmp_path, "cfm", 1, {"sm": 89})


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
