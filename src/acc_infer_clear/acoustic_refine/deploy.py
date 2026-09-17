"""Source/device-bound offline choices with unchanged unknown-shape fallbacks."""
import ast,hashlib,json
from pathlib import Path
import torch
from acc_infer_clear.acoustic_pipeline.deploy import identity as base_identity
from acc_infer_clear.kernels.alias_free import FusedAliasFree
from acc_infer_clear.models.acoustic_stage2 import SelectedConv,SelectedLinear
from .alias import run
from .wavenet import RefinedWaveNet
def identity():
    v=base_identity();v['refine_sources']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')};return v
class RefinedAlias(torch.nn.Module):
    def __init__(self,old,plans):super().__init__();self.old=old;self.plans=plans
    def forward(self,x):
        p=self.plans.get(tuple(x.shape))
        if p is None or x.dtype!=torch.float32:return self.old(x)
        return run(self.old,x,p)
def apply(model,component,plan):
    changed=[]
    if component=='cfm':
        ffns={ast.literal_eval(k):v for k,v in plan.get('ffn',{}).items()}
        for path,m in model.model.named_modules():
            if isinstance(m,SelectedLinear) and '.feed_forward.' in path:
                for (extent,n,k),choice in ffns.items():
                    if (n,k)==(m.out_features,m.in_features) and extent in m.plans:
                        m.plans=dict(m.plans);m.plans[extent]=choice;changed.append(path+str(extent))
        if plan.get('wavenet_shapes'):
            if isinstance(model.model.wavenet,RefinedWaveNet):raise RuntimeError('WaveNet already refined')
            model.model.wavenet=RefinedWaveNet(model.model.wavenet,plan['wavenet_shapes']);changed.append('cfm.wavenet')
        return changed
    aliases={ast.literal_eval(k):v for k,v in plan.get('alias',{}).items()};convs={ast.literal_eval(k):v for k,v in plan.get('conv',{}).items()}
    def walk(root,prefix):
        for name,m in list(root.named_children()):
            path=prefix+'.'+name
            if isinstance(m,FusedAliasFree) and aliases:
                root.add_module(name,RefinedAlias(m,aliases));changed.append(path)
            elif isinstance(m,SelectedConv):
                o=m.old
                if o.transpose or o.groups!=1:continue
                for (b,t),old in list(m.plans.items()):
                    key=(b,o.in_channels,o.out_channels,o.kernel_size[0],o.stride[0],o.padding[0],o.dilation[0],t)
                    if key in convs and old['backend']=='native':
                        m.plans=dict(m.plans);m.plans[b,t]=dict(old,tile=convs[key]);changed.append(path+str((b,t)))
            else:walk(m,path)
    walk(model,'vocoder');return changed
def prepare(engine,path):
    if engine.sessions or engine.head_graphs is not None or getattr(engine.rt.target,'graph_sealed',False):raise RuntimeError('Prepare refinement before graph capture/admission')
    if getattr(engine,'_acoustic_refine_prepared',False):raise RuntimeError('Refinement already prepared')
    plan=json.loads(Path(path).read_text())
    if plan['identity']!=identity():raise ValueError('Refinement source/device/compiler mismatch')
    changed=apply(engine.student,'cfm',plan)+apply(engine.tts.bigvgan,'vocoder',plan)
    engine._acoustic_refine_prepared=True
    return dict(changed=changed,plan=str(path),online_tuning=False,precision_unchanged=True,unknown_shapes='previous_pipeline')
