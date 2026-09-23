"""Offline-only, source/device-bound dispatch for measured acoustic shapes."""
import ast,hashlib,json
from dataclasses import asdict
from pathlib import Path
import torch,triton
from inspark_infer.ops.planning.planner import DeviceCaps
from inspark_infer.ops.triton.fp8_conv_ntc import NTCConv
from inspark_infer.ops.triton.acoustic_pipeline.ops import run

def identity():
    root=Path(__file__).resolve().parent
    return dict(device=asdict(DeviceCaps.current()),torch=torch.__version__,triton=triton.__version__,cuda=torch.version.cuda,
                sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [root/'conv.py',root/'ops.py']})

def signature(o,b,t):return (o.in_channels,o.out_channels,o.kernel_size[0],o.stride[0],o.padding[0],o.dilation[0],o.transpose,o.output_padding[0],b,t)

class PipelineConv(torch.nn.Module):
    def __init__(self,old,plans):
        super().__init__();self.old=old;self.plans=plans
        self.register_buffer('weight_col',old.packed_ntc.t().contiguous())
        for key in ('in_channels','out_channels','kernel_size','stride','padding','dilation','groups','output_padding','transpose','precision'):setattr(self,key,getattr(old,key))
    def forward(self,x):
        plan=self.plans.get((x.shape[0],x.shape[-1]))
        if plan is None:return self.old(x)
        o=self.old.old
        return run(x,self.weight_col,o.bias,o.in_channels,o.out_channels,o.kernel_size[0],o.stride[0],o.padding[0],o.dilation[0],plan,o.scales,transpose=o.transpose,output_padding=o.output_padding[0])

def prepare(engine,path):
    if engine.sessions or engine.head_graphs is not None or getattr(engine.rt.target,'graph_sealed',False):raise RuntimeError('Prepare acoustic pipeline before graph capture/admission')
    if getattr(engine,'_pipeline_prepared',False):raise RuntimeError('Pipeline already prepared')
    data=json.loads(Path(path).read_text())
    if data['identity']!=identity():raise ValueError('Acoustic pipeline plan/device/compiler/source mismatch')
    if identity()['device']['sm']!=120:raise ValueError('Only SM120 is validated; prepare another device offline')
    plans={ast.literal_eval(k):v for k,v in data['plans'].items()};installed=[]
    def walk(root,prefix):
        for name,m in list(root.named_children()):
            label=prefix+'.'+name
            if isinstance(m,NTCConv):
                sig=signature(m.old,1,1)[:8];choices={k[-2:]:v for k,v in plans.items() if k[:8]==sig}
                if choices:root.add_module(name,PipelineConv(m,choices));installed.append(dict(path=label,shapes=[list(k) for k in choices]))
            else:walk(m,label)
    walk(engine.student.model,'cfm');walk(engine.tts.bigvgan,'vocoder');engine._pipeline_prepared=True
    return dict(installed=installed,plan=str(path),online_tuning=False,unknown_shapes='legacy',precision_unchanged=True)
