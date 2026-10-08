import pytest
from inspark_infer.api.release import engine_release

def test_independent_precision_bundle_preserves_defaults():
    info={'engine_prefix':'old','engine_revision':'oldsha','deployments':{'fp8_b64':{},'nvfp4_b64':{'engine_prefix':'new','engine_revision':'newsha'}}}
    assert engine_release(info,'fp8',64)==('old','oldsha')
    assert engine_release(info,'nvfp4',64)==('new','newsha')
    with pytest.raises(ValueError):engine_release(info,'nvfp4',8)
