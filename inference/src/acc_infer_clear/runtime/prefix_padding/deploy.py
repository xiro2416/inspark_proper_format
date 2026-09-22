"""Opt-in first-head buckets with old graph keys retained for overflow."""
import ast,json,hashlib
from pathlib import Path
from dataclasses import asdict
import torch
from acc_infer_clear.ops.planning.planner import DeviceCaps
from acc_infer_clear.runtime.prefix_padding.lab import Bank, make, bind

def identity():
 import triton
 root=Path(__file__).resolve().parents[2]
 names=['runtime/prefix_padding/lab.py','runtime/prefix_padding/deploy.py','ops/triton/fp8.py','ops/triton/stage2_gemm.py','runtime/prefix_graphs.py','ops/triton/target_seven/pointwise.py']
 return dict(device=asdict(DeviceCaps.current()),torch=torch.__version__,triton=triton.__version__,sources={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in names})

def prepare(engine,path):
 if engine.sessions or getattr(engine,'_prefix_padding_prepared',False):raise RuntimeError('Prepare padding once, before admission')
 data=json.loads(Path(path).read_text())
 if not data.get('validated') or data.get('identity')!=identity():raise ValueError('Padding plan needs validation and matching hardware/source')
 old=engine.prefix_graphs;bank=Bank(engine);bank.prefill_graphs=dict(old.prefill_graphs);bank.latent_graphs=dict(old.latent_graphs);bank.keepalive=[];added=[]
 bank.head_only=True;bank.legacy_keys={'prefill':set(),'latent':set()}
 for bs,entry in data['batches'].items():
  b=int(bs)
  if b>engine.config['max_batch']:continue
  fixed={ast.literal_eval(k):v for k,v in entry['choices'].items()}
  candidate=make(engine,b,{},'tight',fixed,True)
  bank.prefill_graphs={k:v for k,v in bank.prefill_graphs.items() if k[0]!=b}
  bank.latent_graphs={k:v for k,v in bank.latent_graphs.items() if k[0]!=b}
  bank.prefill_graphs.update(candidate.prefill_graphs);bank.latent_graphs.update(candidate.latent_graphs);bank.keepalive.extend(candidate.keepalive)
  added.extend([['prefill',b,48],['latent',b,80]])
 bind(engine,bank);engine._prefix_padding_prepared=True
 return dict(added=added,overflow='Tail or over-bucket lengths use eager math; no online capture',bf16_compatibility='Protected projections keep legacy row layout locally',online_learning=False)
