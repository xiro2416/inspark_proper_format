from types import SimpleNamespace

from acc_infer_clear.tensorrt_backend.native113 import NativeDraftFullBank113


def test_draft_cache_mirror_is_not_gated_by_committed_token_bucket():
    bank=SimpleNamespace(backends={8:object()})
    assert NativeDraftFullBank113.enabled_for_total(bank,8)
    assert NativeDraftFullBank113.enabled_for_total(bank,16)
    assert NativeDraftFullBank113.enabled_for_total(bank,56)
    assert NativeDraftFullBank113.enabled_for_total(bank,64)


def test_draft_cache_mirror_is_disabled_without_an_engine():
    bank=SimpleNamespace(backends={})
    assert not NativeDraftFullBank113.enabled_for_total(bank,64)
