"""CPU regressions for FP8 routing, engine contracts and isolation."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from inspark_infer.command import main
from inspark_infer.runtime.zipvoice_fp8.common import profile_manifest,workload
from inspark_infer.build.zipvoice_fp8 import check_workload,ensure,validate_bundle


@pytest.mark.parametrize('args',[
    ['zipvoice','infer','--precision','fp8','--batch','16'],
    ['zipvoice','infer','--precision=fp8','--batch','16'],
    ['trt','ensure','--model','zipvoice','--precision','fp8'],
])
def test_explicit_fp8_dispatch(args,monkeypatch):
    import sys
    calls=[]
    monkeypatch.setitem(sys.modules,'inspark_infer.runtime.zipvoice_fp8.cli',
                        SimpleNamespace(main=lambda values:calls.append(values) or 0))
    assert main(args)==0
    assert calls and ('infer' in calls[0] or 'ensure' in calls[0])


@pytest.mark.parametrize('batch',[1,2,4,8,16,32,64])
def test_all_components_reject_outside_profiles(batch):
    manifest=profile_manifest(batch)
    for frames,tokens in [(600,52),(760,78),(920,141)]:
        check_workload(manifest,workload(batch,frames,tokens))
    for frames,tokens in [(599,78),(921,78),(760,51),(760,142)]:
        with pytest.raises(ValueError):check_workload(manifest,workload(batch,frames,tokens))
    with pytest.raises(ValueError):check_workload(manifest,{**workload(batch,760,78),'steps':4})


def test_no_batch_substitution():
    with pytest.raises(ValueError,match='Supported FP8'):ensure(128)


def test_manifest_rejects_int8(tmp_path):
    (tmp_path/'manifest.json').write_text(json.dumps(dict(schema=1,model='zipvoice',precision='int8',batch=16)))
    with pytest.raises(ValueError,match='Unsupported'):validate_bundle(tmp_path)


def test_infer_does_not_initialize_torch_for_help():
    from inspark_infer.runtime.zipvoice_fp8.worker import main as worker
    with pytest.raises(SystemExit) as error:worker(['infer','--help'])
    assert error.value.code==0
