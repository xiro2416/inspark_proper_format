"""Regression contracts for compact TRT caches and a fixed K128 head loop."""
from types import SimpleNamespace
from unittest.mock import patch

from inspark_infer.runtime.indextts2.slot_draft import SlotDraft
from inspark_infer.runtime.indextts2.device_round import DeviceRoundHead


def test_native_draft_requires_identity_slots_and_room_for_context_commit():
    draft = SimpleNamespace(native_full_bank=SimpleNamespace(graphs={(4, 128): object()}))
    assert SlotDraft.native_eligible(draft, 4, [0, 1, 2, 3], 120)
    assert not SlotDraft.native_eligible(draft, 4, [1, 0, 2, 3], 120)
    assert not SlotDraft.native_eligible(draft, 4, [4, 5, 6, 7], 120)
    assert not SlotDraft.native_eligible(draft, 4, [0, 1, 2, 3], 121)
    assert not SlotDraft.native_eligible(draft, 1, [0], 50)


def test_capacity_fallback_exports_progress_before_any_out_of_bounds_step():
    events = []
    runner = SimpleNamespace(
        rt=SimpleNamespace(residual=SimpleNamespace(device_failures=SimpleNamespace(item=lambda: 0))),
        ready=object(), status=SimpleNamespace(item=lambda: 4),
        past=object(), draft_lengths=object(),
        finish=lambda: events.append('export'), step=lambda: events.append('unsafe_step'),
    )
    with patch('inspark_infer.runtime.indextts2.device_round.status'):
        assert DeviceRoundHead.run(runner) == 0
    assert events == ['export']
    assert runner.fallback_reason == 'kv_capacity'
    assert not runner.failed


def test_residual_failure_does_not_commit_unvalidated_tokens():
    events = []
    runner = SimpleNamespace(
        rt=SimpleNamespace(residual=SimpleNamespace(device_failures=SimpleNamespace(item=lambda: 0))),
        ready=object(), status=SimpleNamespace(item=lambda: 2),
        past=object(), draft_lengths=object(),
        finish=lambda: events.append('export'), step=lambda: events.append('step'),
    )
    with patch('inspark_infer.runtime.indextts2.device_round.status'):
        assert DeviceRoundHead.run(runner) == 0
    assert not events
    assert runner.failed and runner.fallback_reason == 'residual'
