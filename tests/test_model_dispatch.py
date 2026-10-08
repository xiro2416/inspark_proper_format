import sys
from types import SimpleNamespace
from inspark_infer.command import main

def test_zipvoice_dispatch_remains_independent(monkeypatch):
    seen=[]
    monkeypatch.setitem(sys.modules,'inspark_infer.runtime.zipvoice.cli',SimpleNamespace(main=lambda args:seen.append(args) or 0))
    assert main(['zipvoice','prepare','--batch','32'])==0
    assert seen==[['prepare','--batch','32']]

def test_zipvoice_ensure_keeps_its_existing_entry(monkeypatch):
    seen=[]
    monkeypatch.setitem(sys.modules,'inspark_infer.runtime.zipvoice.cli',SimpleNamespace(main=lambda args:seen.append(args) or 0))
    assert main(['trt','ensure','--model','zipvoice','--batches','16,32'])==0
    assert seen==[['ensure','--model','zipvoice','--batches','16,32']]
