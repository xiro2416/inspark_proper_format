"""B128 profile contracts and inherited head/batch address coverage."""
from pathlib import Path
import pytest
from inspark_infer.runtime.zipvoice_fp8_b128.common import BATCHES,profile_manifest,workload
from inspark_infer.build.zipvoice import check_workload


def test_full_batch_profiles_and_domain():
    assert BATCHES==(128,)
    m=profile_manifest(128)
    for frames,tokens in [(600,52),(600,141),(760,78),(920,52),(920,141)]:
        check_workload(m,workload(128,frames,tokens))
    assert m['engines']['text']['shape_profile']['token_ids'][1]==[128,78]
    assert m['engines']['unique']['shape_profile']['token_ids'][1]==[1,78]
    for b in [1,64]:
        with pytest.raises(ValueError):check_workload(m,workload(b,760,78))
    for t,n in [(599,78),(921,78),(760,142)]:
        with pytest.raises(ValueError):check_workload(m,workload(128,t,n))


def test_head_batch_mapping_covers_every_row():
    offsets=[h*128+b for h in range(4) for b in range(128)]
    assert sorted(offsets)==list(range(512))
    root=Path(__file__).resolve().parents[1]/'src/inspark_infer/ops/tensorrt/zipvoice_fp8_b128/b128'
    for folder in [root,root/'geo']:
        for name in ['normal_tf32_runtime_kernel.py','online_nonlinear_rna_runtime_kernel.py']:
            source=(folder/name).read_text()
            assert 'h * 128 + b' in source and 'h * 64 + b' not in source
        for name in ['normal_tf32_plugin.py','online_nonlinear_rna_plugin.py']:
            assert 'launch.grid_y = 128' in (folder/name).read_text()

@pytest.mark.parametrize('args',[
    ['zipvoice','infer','--precision','fp8','--batch','128'],
    ['zipvoice','infer','--precision=fp8','--batch=128'],
    ['zipvoice','ensure','--precision','fp8','--batches','64,128'],
    ['trt','ensure','--model','zipvoice','--precision','fp8','--batches=128'],
])
def test_b128_dispatch_uses_independent_namespace(args,monkeypatch):
    import sys
    from types import SimpleNamespace
    from inspark_infer.command import main
    seen=[]
    monkeypatch.setitem(sys.modules,'inspark_infer.runtime.zipvoice_fp8_b128.cli',SimpleNamespace(main=lambda a:seen.append(a) or 0))
    assert main(args)==0 and seen
