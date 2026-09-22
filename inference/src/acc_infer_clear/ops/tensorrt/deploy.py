"""Load an offline, device-bound Torch-TensorRT comparison plan.

Only pure compute subgraphs are replaced. Slot KV mutation, speculative
verification, acceptance and batching remain owned by the common scheduler.
"""
import hashlib
import json
from pathlib import Path

import torch
from torch import nn


def _sha256(path):
    digest=hashlib.sha256()
    with open(path,'rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def _resolve(engine,path):
    parts=path.split('.')
    if parts[:2]==['target','blocks']:
        return engine.tts.gpt.gpt.h[int(parts[2])],parts[3]
    if parts[:2]==['draft','blocks']:
        return engine.rt.engine.draft.layers[int(parts[2])],parts[3]
    if parts[:2]==['cfm','blocks'] and parts[3]=='feed_forward':
        return engine.student.model.transformer.layers[int(parts[2])],parts[3]
    raise ValueError('Unknown TensorRT attachment path: '+path)


class _CFMStatic(nn.Module):
    def __init__(self,eager,compiled,batch,frames):
        super().__init__();self.eager=eager;self.compiled=compiled
        self.batch=int(batch);self.frames=int(frames)
    def forward(self,x,prompt,lengths,times,style,mu):
        if x.shape[0]==self.batch and x.shape[-1]==self.frames:
            return self.compiled(x,prompt,lengths,times,style,mu)
        return self.eager(x,prompt,lengths,times,style,mu)


class _VocoderStatic(nn.Module):
    def __init__(self,eager,compiled,batch,frames):
        super().__init__();self.eager=eager;self.compiled=compiled
        self.batch=int(batch);self.frames=int(frames)
    def forward(self,x):
        if x.shape[0]==self.batch and x.shape[-1]==self.frames:return self.compiled(x)
        return self.eager(x)


class _CFMSolverStatic(nn.Module):
    def __init__(self,eager,compiled,batch,frames):
        super().__init__();self.eager=eager;self.compiled=compiled
        self.batch=int(batch);self.frames=int(frames)
        # Preserve deployment metadata expected by diagnostics and shutdown.
        self.model=eager.model;self.times=eager.times;self.identity=eager.identity
        self.observer=None
    def forward(self,x,prompt,lengths,style,mu,mask):
        if x.shape[0]==self.batch and x.shape[-1]==self.frames:
            return self.compiled(x,prompt,lengths,style,mu,mask)
        return self.eager(x,prompt,lengths,style,mu,mask)


def _prepare_v2(engine,plan,plan_path,torch_tensorrt):
    batch=int(engine.config['max_batch'])
    candidates=[entry for entry in plan['engines'] if int(entry['batch'])==batch]
    if {entry['component'] for entry in candidates}!={'cfm','vocoder'}:
        raise ValueError('TensorRT V2 requires exact CFM/Vocoder engines for configured max_batch')
    loaded={}
    for entry in candidates:
        artifact=(plan_path.parent/entry['artifact']).resolve()
        if not artifact.is_relative_to(plan_path.parent):raise ValueError('TensorRT artifact escapes plan directory')
        if _sha256(artifact)!=entry['sha256']:raise ValueError('TensorRT artifact hash mismatch: '+entry['component'])
        loaded[entry['component']]=(torch_tensorrt.load(str(artifact)).eval().cuda(),entry,artifact)
    cfm,cfm_entry,cfm_path=loaded['cfm'];vocoder,vocoder_entry,vocoder_path=loaded['vocoder']
    engine.student.model=_CFMStatic(engine.student.model,cfm,batch,cfm_entry['frames']).eval()
    engine.vocoder=_VocoderStatic(engine.tts.bigvgan,vocoder,batch,vocoder_entry['frames']).eval()
    return dict(format_version=2,backend='Torch-TensorRT Dynamo',precision=plan['precision'],
                origin='eager_after_common_precision_before_project_fusions',static_batch=batch,
                engines=[dict(component='cfm',artifact=str(cfm_path),sha256=cfm_entry['sha256'],frames=cfm_entry['frames']),
                         dict(component='vocoder',artifact=str(vocoder_path),sha256=vocoder_entry['sha256'],frames=vocoder_entry['frames'])],
                compiled_subgraphs=2,fallbacks=plan.get('fallbacks',[]),scheduler_shared=True,
                tactic_search=plan['tactic_search'],online_compilation=False,plan=str(plan_path))


def _prepare_v3(engine,plan,plan_path,torch_tensorrt):
    batch=int(engine.config['max_batch'])
    candidates=[entry for entry in plan['engines']
                if int(entry['batch'])==batch and entry['component']=='cfm_solver']
    if len(candidates)!=1:
        raise ValueError('TensorRT V3 requires one exact full-CFM Solver engine for configured max_batch')
    entry=candidates[0];artifact=(plan_path.parent/entry['artifact']).resolve()
    if not artifact.is_relative_to(plan_path.parent):raise ValueError('TensorRT artifact escapes plan directory')
    if _sha256(artifact)!=entry['sha256']:raise ValueError('TensorRT full-CFM artifact hash mismatch')
    compiled=torch_tensorrt.load(str(artifact)).eval().cuda()
    engine.student=_CFMSolverStatic(engine.student,compiled,batch,entry['frames']).eval()
    return dict(format_version=3,backend='Torch-TensorRT Dynamo',precision=plan['precision'],
                origin='eager_after_common_precision_before_project_fusions',static_batch=batch,
                engines=[dict(component='cfm_solver',artifact=str(artifact),sha256=entry['sha256'],
                              frames=entry['frames'])],compiled_subgraphs=1,
                strict_full_compilation=bool(plan.get('strict_full_compilation')),
                scheduler_shared=True,vocoder_shared=True,tactic_search=plan['tactic_search'],
                online_compilation=False,plan=str(plan_path))


def prepare(engine,plan_path):
    if engine.sessions or engine.head_graphs is not None:
        raise RuntimeError('TensorRT must be loaded before requests and graph capture')
    plan_path=Path(plan_path).resolve();plan=json.loads(plan_path.read_text())
    if plan.get('format_version') not in (1,2,3):raise ValueError('Unknown TensorRT plan format')
    if plan.get('precision')!='bf16_fp32_interfaces':raise ValueError('TensorRT precision contract changed')
    if int(plan.get('sm',-1))!=torch.cuda.get_device_capability()[0]*10+torch.cuda.get_device_capability()[1]:
        raise ValueError('TensorRT engines are device-architecture bound')
    if engine.config['max_batch']>int(plan['max_batch']):raise ValueError('TensorRT batch profile is too small')
    import tensorrt
    import torch_tensorrt
    if plan.get('tensorrt_version')!=tensorrt.__version__:raise ValueError('TensorRT version changed')
    if plan.get('torch_tensorrt_version')!=torch_tensorrt.__version__:raise ValueError('Torch-TensorRT version changed')
    if plan['format_version']==3:return _prepare_v3(engine,plan,plan_path,torch_tensorrt)
    if plan['format_version']==2:return _prepare_v2(engine,plan,plan_path,torch_tensorrt)
    attached=[]
    for entry in plan['engines']:
        artifact=(plan_path.parent/entry['artifact']).resolve()
        if not artifact.is_relative_to(plan_path.parent):raise ValueError('TensorRT artifact escapes plan directory')
        if _sha256(artifact)!=entry['sha256']:raise ValueError('TensorRT artifact hash mismatch: '+entry['path'])
        parent,name=_resolve(engine,entry['path'])
        compiled=torch_tensorrt.load(str(artifact)).eval().cuda()
        setattr(parent,name,compiled)
        attached.append(dict(path=entry['path'],artifact=str(artifact),sha256=entry['sha256'],
                             profile=entry['profile'],operators=entry['operators']))
    return dict(format_version=1,backend='Torch-TensorRT Dynamo',precision=plan['precision'],
                origin='eager_after_common_precision_before_project_fusions',
                engines=attached,compiled_subgraphs=len(attached),fallbacks=plan.get('fallbacks',[]),
                scheduler_shared=True,online_compilation=False,plan=str(plan_path))
