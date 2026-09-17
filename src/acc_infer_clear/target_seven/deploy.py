"""Per-engine installation before capture, leaving schema7 and globals untouched."""
import hashlib
import json
import types
from pathlib import Path

def identity():
    from acc_infer_clear.ar_pipeline.deploy import identity as parent
    data=parent();folder=Path(__file__).parent
    data['seven_sources']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir() if p.suffix in ('.py','.cu')}
    src=folder.parent/'dspark/slot_target.py'
    data['slot_target_source']=hashlib.sha256(src.read_bytes()).hexdigest()
    return data

def install(engine,data,*,experimental=False):
    if engine.sessions or engine.head_graphs is not None or getattr(engine.rt.target,'graph_sealed',False):
        raise RuntimeError('Target seven refinement must precede graph capture/admission')
    if getattr(engine,'_target_seven_prepared',False):raise RuntimeError('Already prepared')
    if not experimental:
        if data['identity']!=identity():raise ValueError('Seven plan identity mismatch')
        if not data.get('validated',False):raise ValueError('Seven plan requires completed validation')
    from .projections import install as projections
    from .pointwise import PlannedPointwise
    from .attention import AttentionPolicy
    result={}
    if data.get('projections',{}).get('choices'):
        result['projections']=projections(engine,data['projections'])
    changes=[]
    if data.get('exact_layernorm'):
        from .exact_norm import prepare as prepare_exact,layernorm as exact_layernorm
        prepare_exact()
        import torch
        class ExactNorm(torch.nn.Module):
            def __init__(self,base,entries):
                super().__init__();self.base=base;self.entries={tuple(e['shape']):e['mode'] for e in entries}
            def forward(self,x):
                mode=self.entries.get(tuple(x.shape))
                if mode is None or x.dtype!=torch.float32 or not x.is_contiguous():return self.base(x)
                return exact_layernorm(x,self.base.weight,self.base.bias,self.base.eps,mode)
    for i,block in enumerate(engine.tts.gpt.gpt.h):
        if data.get('gelu'):
            block.mlp.act=PlannedPointwise(block.mlp.act,'gelu',data['gelu']);changes.append(f'{i}.mlp.act')
        if data.get('layernorm'):
            for name in ('ln_1','ln_2'):
                setattr(block,name,PlannedPointwise(getattr(block,name),'layernorm',data['layernorm']));changes.append(f'{i}.{name}')
        if data.get('exact_layernorm'):
            if data.get('layernorm'):raise ValueError('Only one LayerNorm variant may be installed')
            for name in ('ln_1','ln_2'):
                setattr(block,name,ExactNorm(getattr(block,name),data['exact_layernorm']));changes.append(f'{i}.{name}.exact')
    result['pointwise']=changes
    # Original method creates a fresh SlotTarget. Bind an isolated math function
    # to that instance, then capture. Never patch the module-level attention name.
    original=engine.prepare_slot_target
    policy=AttentionPolicy(data.get('attention',[]))
    def prepare_slot_target(graphs=False):
        original(graphs=False)
        target=engine.rt.target
        if data.get('attention'):
            method=target.math.__func__
            if method.__closure__:raise RuntimeError('Unexpected SlotTarget math closure')
            env=dict(method.__globals__);env['attention']=policy
            fn=types.FunctionType(method.__code__,env,method.__name__,method.__defaults__,None)
            target.math=types.MethodType(fn,target)
        with engine.torch.cuda.stream(engine.model.stream),engine.torch.inference_mode():
            if graphs:target.prepare_graphs()
        return target.stats()
    engine.prepare_slot_target=prepare_slot_target
    engine._target_seven_prepared=True
    result['attention']=data.get('attention',[])
    return result

def attach_experiment(engine,data):
    """Diagnostic hook executes after schema7 refinements and before any graphs."""
    prior=engine.prepare_ar_pipeline
    def prepare_ar_pipeline(path):
        result=prior(path)
        with engine.torch.cuda.stream(engine.model.stream),engine.torch.inference_mode():
            result['seven_experiment']=install(engine,data,experimental=True)
        return result
    engine.prepare_ar_pipeline=prepare_ar_pipeline

def prepare(engine,path):
    return install(engine,json.loads(Path(path).read_text()))
