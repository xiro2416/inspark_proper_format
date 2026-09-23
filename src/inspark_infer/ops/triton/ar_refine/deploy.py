import ast,hashlib,json
from pathlib import Path
import torch
from inspark_infer.ops.eager.matrix import MatrixLinear
from inspark_infer.ops.triton.fp8 import linear
from inspark_infer.ops.triton.stage2_gemm import col_linear
from inspark_infer.ops.planning.planner import Tile
from inspark_infer.ops.triton.acoustic_pipeline.deploy import identity as base_identity
def identity():
    root=Path(__file__).resolve().parents[3];v=base_identity()
    files=[root/name for name in ('ops/triton/fp8.py','ops/triton/stage2_gemm.py',
           'quantization/precision.py','quantization/weights.py','ops/eager/matrix.py','ops/matrix.py')]+[Path(__file__)]
    v['ar_sources']={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files};return v
class RefinedLinear(torch.nn.Module):
    def __init__(self,old,plans):
        super().__init__();self.old=old;self.plans=plans;self.in_features=old.in_features;self.out_features=old.out_features;self.precision=old.precision
        self.register_buffer('weight_col',old.weight[:,:old.scales.numel()].t().contiguous());self.register_buffer('ones',torch.ones(1,device=old.weight.device))
    @property
    def weight(self):return self.old.weight
    @property
    def bias(self):return self.old.bias
    def forward(self,x):
        p=self.plans.get(x.numel()//self.in_features)
        if p is None:return self.old(x)
        o=self.old;tile=Tile(**p['tile'])
        if p['backend']=='row':return linear(x,o.weight,o.scales,o.bias,tile)
        return col_linear(x,self.weight_col,o.scales,o.bias,tile,'triton' if p['backend']=='col' else 'scaled_mm',self.ones)
def prepare(engine,path):
    if engine.sessions or engine.head_graphs is not None or getattr(engine.rt.target,'graph_sealed',False):raise RuntimeError('AR preparation must precede capture/admission')
    if getattr(engine,'_ar_refine_prepared',False):raise RuntimeError('AR refinement already prepared')
    plan=json.loads(Path(path).read_text())
    if plan['identity']!=identity():raise ValueError('AR plan source/device/compiler mismatch')
    changed=[]
    def walk(root,prefix,choices):
        for name,m in list(root.named_children()):
            label=prefix+'.'+name
            if isinstance(m,MatrixLinear) and m.precision=='fp8':
                pp={k[0]:v for k,v in choices.items() if k[1:]==(m.out_features,m.in_features,m.bias is not None)}
                if pp:root.add_module(name,RefinedLinear(m,pp));changed.append(label)
            else:walk(m,label,choices)
    for comp,root in [('target',engine.tts.gpt.gpt.h),('draft',engine.rt.engine.draft.layers)]:
        walk(root,comp,{ast.literal_eval(k):v for k,v in plan.get(comp,{}).items()})
    engine._ar_refine_prepared=True
    return dict(changed=changed,online_tuning=False,unknown_shapes='previous',acceptance_rng_kv_unchanged=True)
