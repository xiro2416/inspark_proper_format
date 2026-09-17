"""Optional measured down-projection pipelines over the accepted AR deployment."""
import ast,json,hashlib
from pathlib import Path
import torch
from acc_infer_clear.ar_refine.deploy import RefinedLinear,identity as parent_identity
from acc_infer_clear.kernels.stage2_gemm import col_linear
from acc_infer_clear.kernels.planner import Tile
from .gemm import run
def identity():
    v=parent_identity();root=Path(__file__).parent
    v['pipeline_sources']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'gemm.py',root/'deploy.py')};return v
class PipelineLinear(torch.nn.Module):
    def __init__(self,old,plans):
        super().__init__();self.old=old;self.plans=plans;self.in_features=old.in_features;self.out_features=old.out_features;self.precision=old.precision
    @property
    def weight(self):return self.old.weight
    @property
    def bias(self):return self.old.bias
    def forward(self,x):
        p=self.plans.get(x.numel()//self.in_features)
        if p is None or x.dtype!=torch.float32:return self.old(x)
        o=self.old
        if p['kind']=='compiler':return col_linear(x,o.weight_col,o.old.scales,o.bias,Tile(**p['tile']))
        return run(x,o.weight_col,o.old.scales,o.bias,p['plan'])
def prepare(engine,path):
    if engine.sessions or engine.head_graphs is not None or getattr(engine.rt.target,'graph_sealed',False):raise RuntimeError('AR pipeline must precede capture/admission')
    if not getattr(engine,'_ar_refine_prepared',False):raise RuntimeError('Requires the accepted AR refinement')
    if getattr(engine,'_ar_pipeline_prepared',False):raise RuntimeError('Already prepared')
    data=json.loads(Path(path).read_text())
    if data['identity']!=identity():raise ValueError('Pipeline source/device/compiler mismatch')
    changed=[]
    def walk(root,prefix,choices):
        for name,m in list(root.named_children()):
            label=prefix+'.'+name
            if isinstance(m,RefinedLinear):
                p={k[0]:v for k,v in choices.items() if k[1:]==(m.out_features,m.in_features,m.bias is not None)}
                if p:root.add_module(name,PipelineLinear(m,p));changed.append(label)
            else:walk(m,label,choices)
    for comp,root in [('target',engine.tts.gpt.gpt.h),('draft',engine.rt.engine.draft.layers)]:walk(root,comp,{ast.literal_eval(k):v for k,v in data.get(comp,{}).items()})
    engine._ar_pipeline_prepared=True
    return dict(changed=changed,plan=str(path),online_tuning=False,unknown_shapes='accepted_ar',precision_unchanged=True)
