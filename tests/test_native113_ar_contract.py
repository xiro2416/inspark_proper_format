"""Native AR engine metadata gates, without CUDA initialization."""
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import pytest

from acc_infer_clear.tensorrt_backend.native113 import (
    _ar_io_signature, _plan_engine_identity, _validate_engine_io,
)


TRT = SimpleNamespace(
    float32='f32', bfloat16='bf16', bool='bool', int32='i32',
    TensorIOMode=SimpleNamespace(INPUT='input', OUTPUT='output'),
    TensorLocation=SimpleNamespace(DEVICE='device'),
    TensorFormat=SimpleNamespace(LINEAR='linear'),
)


@pytest.mark.parametrize('component,count', [('target', 102), ('draft', 10)])
@pytest.mark.parametrize('batch', [1, 4, 8])
def test_exact_ar_binding_contract(component, count, batch):
    expected = _ar_io_signature(TRT, component, batch)
    assert len(expected) == count
    values = dict(expected)
    engine = SimpleNamespace(
        num_io_tensors=len(values),
        get_tensor_name=lambda i: list(values)[i],
        get_tensor_shape=lambda name: values[name][0],
        get_tensor_dtype=lambda name: values[name][1],
        get_tensor_mode=lambda name: values[name][2],
        get_tensor_location=lambda name: 'device',
        get_tensor_format=lambda name: 'linear',
    )
    _validate_engine_io(engine, TRT, expected, component)
    shape, dtype, mode = values['x']
    values['x'] = ((batch + 1,) + shape[1:], dtype, mode)
    with pytest.raises(ValueError, match='I/O mismatch'):
        _validate_engine_io(engine, TRT, expected, component)


def test_plan_hash_and_legacy_status_are_distinct():
    with TemporaryDirectory(prefix='.ar-contract-', dir=Path(__file__).resolve().parents[1]) as temp:
        root = Path(temp)
        blob = b'mock-engine-not-a-real-plan'
        (root / 'mock.engine').write_bytes(blob)
        _, legacy = _plan_engine_identity(root / 'plan.json', {}, '1', 'mock.engine')
        assert not legacy['plan_hash_verified']
        assert legacy['provenance_status'] == 'legacy_unverified'
        digest = hashlib.sha256(blob).hexdigest()
        plan = {'engine_sha256': {'1': digest}, 'provenance': {'1': {'schema': 1}}}
        _, checked = _plan_engine_identity(root / 'plan.json', plan, '1', 'mock.engine')
        assert checked['plan_hash_verified']
        assert checked['provenance_status'] == 'recorded_not_audited'
        assert not checked['numerical_audit_pass']
        plan['engine_sha256']['1'] = 'bad'
        with pytest.raises(ValueError, match='hash mismatch'):
            _plan_engine_identity(root / 'plan.json', plan, '1', 'mock.engine')
