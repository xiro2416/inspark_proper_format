import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from audit_real_ar import (BoundedARRecorder, GraphProxy, ProposalProxy, ProposalCallProxy,
    actual_target_reference, new_target_kv, ar_coverage, bitwise_equal,
    target_untouched_cache_equal, replay_ar_engine_evidence)


def test_target_kv_extracts_each_real_row_position():
    cache = torch.arange(2 * 2 * 2 * 3 * 16 * 4).reshape(2, 2, 2, 3, 16, 4)
    result = new_target_kv(cache, [1, 7])
    assert result.shape == (2, 2, 2, 3, 8, 4)
    assert torch.equal(result[:, :, 0], cache[:, :, 0, :, 1:9])
    assert torch.equal(result[:, :, 1], cache[:, :, 1, :, 7:15])


def test_native_proxy_preserves_shared_runtime_bank_identity(tmp_path, monkeypatch):
    import validate_trt113_ar
    monkeypatch.setattr(validate_trt113_ar, "engine_evidence", lambda *args: {
        "weight_identity_verified": False, "sha256": "loaded-engine"})
    graph = SimpleNamespace(inputs=(torch.ones(1),), outputs=())
    bank = SimpleNamespace(graphs={(1, 128): graph}, backends={1: object()},
                           artifacts={1: {"path": "unused", "sha256": "loaded-engine"}})
    owner = SimpleNamespace(native_full_bank=bank, graphs={(1, 128): graph})
    engine = SimpleNamespace(rt=SimpleNamespace(target=owner, backbone=SimpleNamespace(native_full_bank=None),
                                               proposal=SimpleNamespace(graphs={})))
    recorder = BoundedARRecorder(engine, tmp_path, {"model_provenance": {"target": {"model_sources": []}}}, 2)
    recorder.install()
    assert owner.graphs[1, 128] is bank.graphs[1, 128]
    assert isinstance(owner.graphs[1, 128], GraphProxy)
    assert owner.graphs[1, 128].inputs is graph.inputs


def test_proposal_observer_clones_real_tokens_without_replay_injection():
    recorder = SimpleNamespace(verify_tokens={})
    graph = SimpleNamespace(inputs=(None, None, None, torch.tensor([3])),
                            outputs=(torch.tensor([[4, 5, 6, 7, 8, 9, 10]]),),
                            graph=SimpleNamespace(replay=lambda: None))
    proxy = ProposalProxy(graph, recorder)
    proxy.graph.replay()
    graph.outputs[0].zero_()
    assert recorder.verify_tokens[1].tolist() == [[3, 4, 5, 6, 7, 8, 9, 10]]


def test_ragged_target_reference_uses_exact_native_stored_values_promoted_fp32():
    seen = []
    class Target:
        def _block_forward_with_hidden_states(self, x, past, keep, position):
            assert position is None
            seen.append((past[0][0].clone(), keep.clone()))
            assert past[0][0].dtype == torch.float32
            assert keep.shape[-1] == past[0][0].shape[-2] + 8
            result = x.sum(-1, keepdim=True)
            present = tuple((torch.cat((k, torch.ones(1, 1, 8, 2)), 2),
                             torch.cat((v, torch.ones(1, 1, 8, 2) * 2), 2)) for k, v in past)
            return result, present, result, result
    raw = torch.full((1, 2, 2, 1, 16, 2), 1.234567)
    stored = raw.bfloat16()
    bundle = {"inputs": (torch.zeros(2, 8, 3), torch.arange(2), torch.tensor([2, 5])),
              "cache_before": stored, "keep": torch.ones(2, 16, dtype=torch.int32)}
    engine = SimpleNamespace(rt=SimpleNamespace(engine=SimpleNamespace(target=Target())))
    result = actual_target_reference(engine, bundle)
    assert result["new_kv"].shape == (1, 2, 2, 1, 8, 2)
    assert seen[0][0].shape[-2] == 2 and seen[1][0].shape[-2] == 5
    assert seen[0][0].flatten()[0].item() == stored.float().flatten()[0].item()
    assert seen[0][0].flatten()[0].item() != raw.flatten()[0].item()


def test_generic_proposal_observes_actual_positions_even_without_native_draft():
    result = [(torch.tensor([[4, 5, 6, 7, 8, 9, 10]]), "probability", "logits")]
    jobs = [{"anchor_token": torch.tensor([3]), "first_position": 17}]
    recorder = SimpleNamespace(verify_tokens={}, draft_positions={})
    class Proposal:
        def __call__(self, observed_jobs, observed_tasks):
            assert observed_jobs is jobs and observed_tasks == ["task"]
            return result
    proxy = ProposalCallProxy(Proposal(), recorder)
    assert proxy(jobs, ["task"]) is result
    assert recorder.verify_tokens[1].tolist() == [[3, 4, 5, 6, 7, 8, 9, 10]]
    assert recorder.draft_positions[1].tolist() == [[17, 18, 19, 20, 21, 22, 23]]


def test_native_coverage_needs_both_actual_components_not_merely_an_engine():
    assert not ar_coverage([])["pass_gate"]
    assert not ar_coverage([{"component": "draft", "batch": 4}])["pass_gate"]
    coverage = ar_coverage([{"component": "draft", "batch": 4}, {"component": "target", "batch": 4}])
    assert coverage["pass_gate"] and coverage["calls"] == {"target": 1, "draft": 1}


def test_untouched_cache_compares_bytes_including_unused_nan_storage():
    before = torch.ones(1, 2, 1, 1, 16, 2)
    before[..., 15, :] = float("nan")
    after = before.clone(); after[..., 2:10, :] = 5
    assert bitwise_equal(before, before.clone())
    assert target_untouched_cache_equal(before, after, [2])
    after[..., 0, :] = 2
    assert not target_untouched_cache_equal(before, after, [2])


@pytest.mark.parametrize("direct_raises", [False, True])
def test_target_direct_probe_restores_graph_cache_and_returns_cloned_graph_outputs(tmp_path, monkeypatch, direct_raises):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    cache = torch.ones(1, 2, 1, 1, 16, 2, dtype=torch.bfloat16)
    class Backend:
        def __init__(self):
            self.cache = cache; self.calls = 0; self.outputs = torch.zeros(1, 8, 1)
        def run(self, x, keep, slots, lengths):
            self.calls += 1
            self.cache[..., 2:10, :] = self.calls
            self.outputs.fill_(self.calls)
            if self.calls == 2 and direct_raises:
                raise RuntimeError("direct probe failed")
            return self.outputs, self.outputs, self.outputs
    backend = Backend()
    target_model = SimpleNamespace(embeddings=lambda tokens: tokens.float().unsqueeze(-1),
        text_pos_embedding=SimpleNamespace(emb=lambda positions: torch.zeros(*positions.shape, 1)))
    target = SimpleNamespace(keep=torch.ones(1, 16, dtype=torch.int32))
    session = {"case": {"id": "real-request"}, "_row": SimpleNamespace(kv=SimpleNamespace(slot=0))}
    engine = SimpleNamespace(sessions={"real-request": session},
        rt=SimpleNamespace(target=target, engine=SimpleNamespace(target=SimpleNamespace(model=target_model))))
    manifest = {}
    recorder = BoundedARRecorder(engine, tmp_path, manifest, 2)
    recorder.verify_tokens[1] = torch.arange(8)[None]
    recorder.draft_positions[1] = torch.arange(7)[None]
    graph = lambda x, slots, lengths: backend.run(x, target.keep, slots, lengths)
    proxy = GraphProxy(graph, recorder, "target", 1, backend,
                       SimpleNamespace(artifacts={1: {"sha256": "frozen-engine"}}))
    inputs = (torch.arange(8).float().reshape(1, 8, 1), torch.tensor([0]), torch.tensor([2]))
    if direct_raises:
        with pytest.raises(RuntimeError, match="direct probe"):
            proxy(*inputs)
    else:
        result = proxy(*inputs)
        assert all(torch.equal(value, torch.ones_like(value)) for value in result)
        backend.outputs.zero_()
        assert all(torch.equal(value, torch.ones_like(value)) for value in result)
        assert manifest["ar_calls"][0]["cache_state_gate"]
        assert not manifest["ar_calls"][0]["graph_direct_exact_gate"]
    assert torch.equal(cache, torch.ones_like(cache))


def test_replay_rechecks_frozen_ar_engine_and_metadata_hashes(monkeypatch):
    import validate_trt113_ar
    frozen = {"path": "unused", "sha256": "engine", "build_metadata_sha256": "metadata",
              "weight_identity_verified": True}
    manifest = {"ar_calls": [{"component": "target", "batch": 1, "route": {"artifact": {"sha256": "engine"}}}],
                "ar_engines": {"target": {"1": frozen}}}
    models = {"target": {"model_sources": []}}
    monkeypatch.setattr(validate_trt113_ar, "engine_evidence", lambda *args: dict(frozen))
    assert replay_ar_engine_evidence(manifest, models)["target"]["1"]["weight_identity_verified"]
    for field in ("sha256", "build_metadata_sha256"):
        monkeypatch.setattr(validate_trt113_ar, "engine_evidence", lambda *args, field=field: {**frozen, field: "changed"})
        with pytest.raises(ValueError, match="changed after capture"):
            replay_ar_engine_evidence(manifest, models)
