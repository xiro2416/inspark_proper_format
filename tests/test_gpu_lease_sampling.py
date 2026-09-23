"""A recent utilization sample must not look like persistent GPU ownership."""
import tempfile

import pytest

from inspark_infer.runtime import device
from inspark_infer.build.trt113 import ROOT


def lease_with_samples(monkeypatch, samples):
    work=ROOT/'.work'
    work.mkdir(exist_ok=True)
    directory=tempfile.TemporaryDirectory(dir=work)
    monkeypatch.setenv('ACC_GPU_LOCK_DIR',directory.name)
    monkeypatch.delenv('ACC_GPU_ALLOW_SHARED',raising=False)
    values=iter(samples)
    monkeypatch.setattr(device.subprocess,'check_output',lambda *args,**kwargs: next(values))
    monkeypatch.setattr(device.time,'sleep',lambda _:None)
    return directory


def test_lease_rechecks_recent_utilization_after_previous_child_exits(monkeypatch):
    with lease_with_samples(monkeypatch,['20, 43','20, 18','20, 0']):
        with device.GPULease(4) as lease:
            assert lease.initial_memory_mib==20
            assert lease.initial_utilization==0


def test_lease_still_rejects_persistently_busy_gpu(monkeypatch):
    with lease_with_samples(monkeypatch,['20, 43']*5):
        with pytest.raises(RuntimeError,match='GPU4 is busy'):
            with device.GPULease(4):pass


def test_lease_never_retries_or_ignores_memory_over_limit(monkeypatch):
    with lease_with_samples(monkeypatch,['21854, 0']):
        with pytest.raises(RuntimeError,match='GPU4 is busy'):
            with device.GPULease(4):pass
