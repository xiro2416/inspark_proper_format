import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from audit_real_ar import BoundedARRecorder, GraphProxy, ProposalProxy, actual_target_reference, new_target_kv


def test_target_kv_extracts_each_real_row_position():
    cache = torch.arange(2 * 2 * 2 * 3 * 16 * 4).reshape(2, 2, 2, 3, 16, 4)
    result = new_target_kv(cache, [1, 7])
    assert result.shape == (2, 2, 2, 3, 8, 4)
    assert torch.equal(result[:, :, 0], cache[:, :, 0, :, 1:9])
    assert torch.equal(result[:, :, 1], cache[:, :, 1, :, 7:15])


def test_native_proxy_preserves_shared_runtime_bank_identity(tmp_path, monkeypatch):
    import validate_trt113_ar
    monkeypatch.setattr(validate_trt113_ar, "engine_evidence", lambda *args: {"weight_identity_verified": False})
    graph = SimpleNamespace(inputs=(torch.ones(1),), outputs=())
    bank = SimpleNamespace(graphs={(1, 128): graph}, backends={1: object()}, artifacts={1: {"path": "unused"}})
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
