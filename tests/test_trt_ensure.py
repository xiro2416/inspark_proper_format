"""CPU-only decision tests: cache, hardware identity, and explicit failure policy."""
from pathlib import Path
import tempfile

import pytest

from inspark_infer.build import ensure as resolver, trt113
from inspark_infer.quantization.trt113 import load_policy


BUILDER = trt113.builder_policy(trt113.ROOT / trt113.BUILDER_DEFAULT)
QUANT = load_policy(trt113.ROOT / trt113.QUANTIZATION_DEFAULT)
GPU = {"physical_gpu": 4, "name": "NVIDIA GeForce RTX 4090", "sm": 89,
       "memory_total_mib": 49140, "memory_used_mib": 0}


@pytest.fixture
def tmp_path():
    work = trt113.ROOT / ".work"
    work.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=work) as directory:
        yield Path(directory)


def test_policy_validation_and_quantization_rejection(tmp_path):
    assert BUILDER["optimization_level"] == {"target": 3, "draft": 5, "cfm": 5, "vocoder": 5}
    assert BUILDER["tiling_optimization_level"] == "none"
    assert QUANT["scheme"] == "none"
    policy = tmp_path / "quant.json"
    policy.write_text('{"schema":1,"scheme":"nvfp4","calibration":null,"scale_format":null,"qdq_graph_sha256":null}')
    with pytest.raises(NotImplementedError, match="explicit Q/DQ"):
        load_policy(policy)


def test_registry_is_exact_hardware_and_baseline():
    assert resolver._registry_entry(GPU, 1, BUILDER)["bundle_id"] == "b62db855b4239970035f"
    assert resolver._registry_entry({**GPU, "memory_total_mib": 24564}, 1, BUILDER) is None
    assert resolver._registry_entry({**GPU, "sm": 120}, 1, BUILDER) is None
    assert resolver._registry_entry(GPU, 1, {**BUILDER, "workspace_bytes": 1 << 30}) is None


def test_legacy_bundle_only_registered_id():
    manifest = {"schema": 1, "profile": trt113.PROFILE, "batch": 1,
                "quantization": "none", "trt": "11.3.0.99",
                "hardware": {key: GPU[key] for key in ("name", "sm", "memory_total_mib")},
                "bundle_id": "b62db855b4239970035f"}
    assert resolver._compatible(manifest, GPU, 1, BUILDER, "new-source")
    assert not resolver._compatible({**manifest, "bundle_id": "other"}, GPU, 1, BUILDER, "new-source")
    assert not resolver._compatible(manifest, {**GPU, "memory_total_mib": 24564}, 1, BUILDER, "new-source")
    modern = {**manifest, "schema": 2, "source_sha256": "new-source", "builder": BUILDER}
    assert resolver._compatible(modern, GPU, 1, BUILDER, "new-source", "11.3.0.99")
    assert not resolver._compatible(modern, GPU, 1, BUILDER, "changed-source", "11.3.0.99")
    assert not resolver._compatible(modern, GPU, 1, BUILDER, "new-source", "11.3.1")


def test_all_component_steps_use_requested_builder(tmp_path):
    steps = trt113._steps(4, tmp_path, 4, BUILDER)
    by_name = dict(steps)
    for name in trt113.COMPONENTS:
        command = by_name[name]
        assert command[command.index("--optimization-level") + 1] == str(BUILDER["optimization_level"][name])
        assert command[command.index("--workspace-bytes") + 1] == str(BUILDER["workspace_bytes"])
        assert command[command.index("--tiling-optimization-level") + 1] == "none"


def test_deployment_template_is_not_sm89_bound(tmp_path):
    deployment = trt113._deployment(tmp_path, 4, 120)
    from inspark_infer.runtime.deployment import load
    data = load(deployment)
    assert data["status"].startswith("sm120_trt113_b4_")
    assert data["tensorrt113_target_full_plan"] == str(tmp_path / "target/plan.json")


def _exercise(monkeypatch, tmp_path, gpu, *, mode="auto", local=None, fetched=None):
    reference = tmp_path / "reference.wav"
    reference.write_bytes(b"wav")
    monkeypatch.setattr(trt113, "ensure_trt113_site", lambda: None)
    monkeypatch.setattr("inspark_infer.ops.tensorrt.native113._import_trt113",
                        lambda: type("TRT", (), {"__version__": "11.3.0.99"})())
    monkeypatch.setattr(resolver, "_local", lambda *args: local)
    monkeypatch.setattr(resolver, "_registry_entry", lambda *args: fetched)
    monkeypatch.setattr("scripts.trt113_provenance.source_identity", lambda root: {"source_sha256": "source"})
    return reference


def test_cache_hit_does_not_fetch_or_build(monkeypatch, tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    reference = _exercise(monkeypatch, tmp_path, GPU, local=bundle)
    monkeypatch.setattr(resolver.hf_cache, "fetch", lambda *args: pytest.fail("unexpected fetch"))
    monkeypatch.setattr(trt113, "build_one", lambda *args: pytest.fail("unexpected build"))
    result = resolver.ensure(GPU, (1,), reference, tmp_path, "auto", BUILDER, QUANT,
                             allow_experimental=True, endpoint="https://hf-mirror.com")
    assert result["bundles"][0]["origin"] == "local"


def test_private_auth_failure_does_not_build(monkeypatch, tmp_path):
    reference = _exercise(monkeypatch, tmp_path, GPU, fetched={"bundle_id": "id", "revision": "a" * 40,
                                                          "repo_id": "user/private"})
    monkeypatch.setattr(resolver.hf_cache, "fetch", lambda *args: (_ for _ in ()).throw(ValueError("invalid HF_TOKEN")))
    monkeypatch.setattr(trt113, "build_one", lambda *args: pytest.fail("auth failure rebuilt"))
    with pytest.raises(ValueError, match="invalid HF_TOKEN"):
        resolver.ensure(GPU, (1,), reference, tmp_path, "auto", BUILDER, QUANT,
                        allow_experimental=True, endpoint="https://hf-mirror.com")


@pytest.mark.parametrize("sm", [89, 90, 100, 120])
def test_new_gpu_build_is_explicit_and_unverified(monkeypatch, tmp_path, sm):
    gpu = {**GPU, "sm": sm, "memory_total_mib": 24564}
    reference = _exercise(monkeypatch, tmp_path, gpu)
    calls = []
    monkeypatch.setattr(trt113, "build_one", lambda *args: calls.append(args) or tmp_path / "built")
    result = resolver.ensure(gpu, (4,), reference, tmp_path, "auto", BUILDER, QUANT,
                             allow_experimental=True, endpoint="https://hf-mirror.com")
    assert result["bundles"][0]["origin"] == "built"
    assert result["status"] == "experimental_not_numerically_certified"
    assert len(calls) == 1


def test_experimental_gate_and_reuse_only(monkeypatch, tmp_path):
    reference = _exercise(monkeypatch, tmp_path, {**GPU, "sm": 120})
    with pytest.raises(ValueError, match="allow-experimental"):
        resolver.ensure(GPU, (1,), reference, tmp_path, "auto", BUILDER, QUANT,
                        allow_experimental=False, endpoint="https://hf-mirror.com")
    with pytest.raises(FileNotFoundError, match="No matching"):
        resolver.ensure({**GPU, "sm": 120}, (1,), reference, tmp_path, "reuse-only", BUILDER, QUANT,
                        allow_experimental=True, endpoint="https://hf-mirror.com")
