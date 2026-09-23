"""CPU-only checks for the exact-shape, four-component bundle contract."""
import hashlib
import json
from pathlib import Path
import tempfile

import pytest
import huggingface_hub

from inspark_infer.build.trt113 import (
    PROFILE, ROOT, _engine_record, parse_batches, unsupported, validate_bundle,
)
from inspark_infer.build import trt113
from inspark_infer.build import hf_cache
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
    assert len(result["reasons"]) == 1
    assert "codex_task" in result
    assert unsupported({"sm": 120}, PROFILE, (1, 4, 8))["reasons"] == []


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


def test_private_cache_paths_and_attestation(tmp_path, monkeypatch):
    assert _repo_id("user/private-model") == "user/private-model"
    assert _remote_path("bundles/sm89/b1") == "bundles/sm89/b1"
    for path in ("../bundle", "bundles/../secret", "/bundles/x", "other/x"):
        with pytest.raises(ValueError):
            _remote_path(path)
    with pytest.raises(ValueError):
        _repo_id("https://huggingface.co/user/repo")
    with monkeypatch.context() as offline:
        offline.setenv("HF_HUB_OFFLINE", "1")
        with pytest.raises(ValueError, match="HF_HUB_OFFLINE=0"):
            hf_cache._token()
    attestation = tmp_path / "attestation.json"
    attestation.write_text(json.dumps({"schema": 1, "reviewed": True,
                                       "redistribution_permitted": True, "sources": {}}))
    manifest = {"components": {"target": {"model_sources": [
        {"role": "target_checkpoint", "sha256": "baaaeb8b56328da81731dc540a85a7dee32eca9da28f174b05757cb651c602a4"}]}}}
    with pytest.raises(ValueError, match="every source embedded"):
        _attestation(attestation, manifest)
    attestation.write_text(json.dumps({"schema": 1, "reviewed": True,
        "redistribution_permitted": True,
        "sources": {"IndexTeam/IndexTTS-2": {"permitted": True, "evidence": "local license review"}}}))
    assert _attestation(attestation, manifest)["reviewed"] is True
    manifest["components"]["target"]["model_sources"][0]["sha256"] = "unknown"
    with pytest.raises(ValueError, match="not pinned"):
        _attestation(attestation, manifest)


def test_private_cache_rejects_public_repository_before_upload_or_download(monkeypatch):
    class PublicApi:
        committed = False

        def __init__(self, **kwargs):
            pass

        def model_info(self, *args, **kwargs):
            return type("Info", (), {"private": False})()

        def create_commit(self, **kwargs):
            self.committed = True
            raise AssertionError("public upload must never start")

    monkeypatch.setattr(huggingface_hub, "HfApi", PublicApi)
    monkeypatch.setattr(hf_cache, "_token", lambda: "dummy-local-test-token")
    monkeypatch.setattr(hf_cache, "validate_bundle", lambda root: {"batch": 1, "bundle_id": "id"})
    monkeypatch.setattr(hf_cache, "_attestation", lambda path, manifest: {})
    monkeypatch.setattr(hf_cache, "gpu_info", lambda index: {
        "physical_gpu": index, "sm": 89, "name": "GPU", "memory_total_mib": 49140})
    work = ROOT / ".work"
    work.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=work) as directory:
        root = Path(directory)
        reference = root / "reference.wav"
        reference.write_bytes(b"wav")
        with pytest.raises(ValueError, match="public repository"):
            hf_cache.publish(root, "user/public", reference)
        with monkeypatch.context() as missing_site:
            missing_site.setenv("ACC_TRT113_SITE", str(root / "missing-trt-site"))
            with pytest.raises(ValueError, match="TensorRT 11.3 environment missing"):
                hf_cache.fetch("user/public", "a" * 40,
                               f"bundles/sm89/{PROFILE}/b1/id", 6, reference,
                               root / "bundles", "https://hf-mirror.com")
        with pytest.raises(ValueError, match="public repository"):
            hf_cache.fetch("user/public", "a" * 40,
                           f"bundles/sm89/{PROFILE}/b1/id", 6, reference,
                           root / "bundles", "https://hf-mirror.com")


def test_private_cache_publish_is_explicit_and_revision_pinned(monkeypatch):
    class Operation:
        def __init__(self, *, path_in_repo, path_or_fileobj):
            self.path_in_repo = path_in_repo
            self.path_or_fileobj = path_or_fileobj

    class PrivateApi:
        def __init__(self, **kwargs):
            self.uploaded = None

        def model_info(self, *args, **kwargs):
            return type("Info", (), {"private": True})()

        def create_commit(self, *, operations, **kwargs):
            self.uploaded = operations
            assert len(operations) == 6
            assert all(operation.path_or_fileobj.is_file() for operation in operations)
            return type("Commit", (), {"oid": "b" * 40})()

    monkeypatch.setattr(huggingface_hub, "HfApi", PrivateApi)
    monkeypatch.setattr(huggingface_hub, "CommitOperationAdd", Operation)
    monkeypatch.setattr(hf_cache, "_token", lambda: "dummy-local-test-token")
    monkeypatch.setattr(hf_cache, "validate_bundle", lambda root: {
        "batch": 4, "bundle_id": "example-id", "hardware": {"sm": 89},
        "files": {"target/engine": "digest"}})
    monkeypatch.setattr(hf_cache, "_attestation", lambda path, manifest: {})
    work = ROOT / ".work"
    work.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=work) as directory:
        root = Path(directory)
        (root / "target").mkdir()
        (root / "target/engine").write_bytes(b"engine")
        (root / "manifest.json").write_text("{}")
        attestation = root / "attestation.json"
        attestation.write_text("{}")
        result = hf_cache.publish(root, "user/private", attestation)
    assert result["status"] == "published_private"
    assert result["revision"] == "b" * 40
    assert result["bundle_path"] == f"bundles/sm89/{PROFILE}/b4/example-id"


def test_private_cache_fetch_retries_failed_private_mirror_on_official_hub(monkeypatch):
    from huggingface_hub.errors import LocalEntryNotFoundError

    class PrivateApi:
        def __init__(self, **kwargs):
            pass

        def model_info(self, *args, **kwargs):
            return type("Info", (), {"private": True})()

    monkeypatch.setattr(huggingface_hub, "HfApi", PrivateApi)
    monkeypatch.setattr(hf_cache, "_token", lambda: "dummy-local-test-token")
    monkeypatch.setattr(hf_cache, "gpu_info", lambda index: {
        "physical_gpu": index, "sm": 89, "name": "GPU", "memory_total_mib": 49140})
    monkeypatch.setattr(hf_cache, "ensure_trt113_site", lambda: None)
    monkeypatch.setattr(hf_cache, "_attestation", lambda path, manifest: {})
    monkeypatch.setattr(hf_cache, "validate_bundle", lambda root: json.loads((root / "manifest.json").read_text()))
    work = ROOT / ".work"
    work.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=work) as directory:
        root = Path(directory)
        remote = root / "remote"
        (remote / "licenses").mkdir(parents=True)
        manifest = {"schema": 1, "profile": PROFILE, "batch": 1, "bundle_id": "id",
                    "hardware": {"sm": 89, "name": "GPU", "memory_total_mib": 49140},
                    "files": {"deployment.json": "unused"}, "deployment": "deployment.json"}
        (remote / "manifest.json").write_text(json.dumps(manifest))
        for name in ("LICENSE", "THIRD_PARTY_NOTICES.md", "distribution_attestation.json",
                     "licenses/BigVGAN.txt", "deployment.json"):
            (remote / name).write_text(name)
        endpoints = []

        def download(*, filename, endpoint, **kwargs):
            endpoints.append(endpoint)
            if endpoint == "https://hf-mirror.com":
                raise LocalEntryNotFoundError("private mirror metadata unavailable")
            return str(remote / filename.rsplit("/", 1)[-1]) if not filename.endswith("licenses/BigVGAN.txt") else str(remote / "licenses/BigVGAN.txt")

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
        monkeypatch.setattr(hf_cache, "_run", lambda name, command, *, stage, gpu: (stage / "route_report_local.json").write_text('{"status":"passed"}'))
        reference = root / "reference.wav"
        reference.write_bytes(b"wav")
        result = hf_cache.fetch("user/private", "a" * 40,
                                f"bundles/sm89/{PROFILE}/b1/id", 4, reference,
                                root / "download", "https://hf-mirror.com")
        assert result["download_endpoint"] == "https://huggingface.co"
        assert endpoints[0:2] == ["https://hf-mirror.com", "https://huggingface.co"]
        assert (Path(result["bundle"]) / "licenses/BigVGAN.txt").is_file()
