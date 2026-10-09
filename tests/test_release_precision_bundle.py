import pytest
from inspark_infer.api.release import engine_release

def test_independent_precision_bundle_preserves_defaults():
    info={'engine_prefix':'old','engine_revision':'oldsha','deployments':{'fp8_b64':{},'nvfp4_b64':{'engine_prefix':'new','engine_revision':'newsha'}}}
    assert engine_release(info,'fp8',64)==('old','oldsha')
    assert engine_release(info,'nvfp4',64)==('new','newsha')
    with pytest.raises(ValueError):engine_release(info,'nvfp4',8)

def test_ready_default_retains_explicit_barrier_pin():
    info={'engine_prefix':'base','engine_revision':'base-sha','deployments':{
        'nvfp4_fp8_b64':{'engine_prefix':'old','engine_revision':'old-sha',
            'default_scheduler':'ready_c','scheduler_variants':{
                'ready_c':{'engine_prefix':'ready','engine_revision':'ready-sha'}}},
        'fp8_b8':{}}}
    assert engine_release(info,'nvfp4_fp8',64)==('ready','ready-sha')
    assert engine_release(info,'nvfp4_fp8',64,'barrier')==('old','old-sha')
    assert engine_release(info,'fp8',8)==('base','base-sha')
    with pytest.raises(ValueError):engine_release(info,'fp8',8,'ready_c')
