"""Opt-in stage2 over validated stage1; sealed shape-specific choices and CFM fusions."""
import ast,hashlib,json
from pathlib import Path
from dataclasses import asdict
import torch,triton
from acc_infer_clear.ops.eager.matrix import MatrixConv, MatrixLinear
from acc_infer_clear.ops.eager.layout_conv import LayoutConv
from acc_infer_clear.ops.planning.planner import Tile, DeviceCaps
from acc_infer_clear.ops.triton.fp8 import linear
from acc_infer_clear.ops.triton.stage2_gemm import col_linear
from acc_infer_clear.ops.triton.stage2_bf16_conv import bf16_conv

def identity():
    root=Path(__file__).resolve().parents[3]
    files=['ops/triton/stage2_gemm.py','ops/triton/stage2_bf16_conv.py','ops/triton/stage2_fusions.py',
           'ops/eager/layout_conv.py','ops/eager/matrix.py','ops/matrix.py','quantization/weights.py']
    return dict(device=asdict(DeviceCaps.current()),torch=torch.__version__,triton=triton.__version__,cuda=torch.version.cuda,
                sources={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in files})

class SelectedLinear(torch.nn.Module):
    def __init__(self,old,plans):
        super().__init__();self.old=old;self.plans=plans;self.in_features=old.in_features;self.out_features=old.out_features
        self.register_buffer('weight_col',old.weight.t().contiguous());self.register_buffer('ones',torch.ones(1,device=old.weight.device))
    def forward(self,x):
        m=x.numel()//self.in_features;choice=self.plans.get(m)
        if choice is None:return self.old(x)
        tile=Tile(**choice['tile']);o=self.old
        if choice['backend']=='row':return linear(x,o.weight,o.scales,o.bias,tile)
        return col_linear(x,self.weight_col,o.scales,o.bias,tile,'triton' if choice['backend']=='col' else 'scaled_mm',self.ones)

class SelectedConv(torch.nn.Module):
    def __init__(self,old,plans):
        super().__init__();self.old=old;self.plans=plans
        self.layouts=torch.nn.ModuleDict({p['backend']:LayoutConv(old,p['backend']) for p in plans.values() if p['backend'] in ('contiguous','channels_last')})
        if any(p['backend']=='native' for p in plans.values()):
            self.register_buffer('packed',old.weight.permute(2,1,0).contiguous().view(-1,old.out_channels))
    def forward(self,x):
        p=self.plans.get((x.shape[0],x.shape[-1]));o=self.old
        if p is None:return o(x)
        if p['backend']!='native':return self.layouts[p['backend']](x)
        return bf16_conv(x,self.packed,o.bias,o.in_channels,o.out_channels,o.kernel_size[0],o.stride[0],o.padding[0],o.dilation[0],Tile(**p['tile']))

def prepare(engine,plan_path,parts=('gemm','conv','norm','gate','rope')):
    if engine.sessions or engine.head_graphs is not None or getattr(engine.rt.target,'graph_sealed',False):raise RuntimeError('Prepare stage2 before capture/admission')
    if getattr(engine,'_acoustic_prepared',set())!={'alias','ntc'}:raise RuntimeError('Stage2 requires the validated stage1 combination')
    done=getattr(engine,'_stage2_parts',set())
    if done&set(parts):raise RuntimeError('Stage2 part already prepared')
    data=json.loads(Path(plan_path).read_text())
    if data['identity']!=identity():raise ValueError('Stage2 plan/device/source mismatch')
    result=dict(parts=list(parts),gemm=[],conv=[],fusions=[],online_tuning=False)
    gm={ast.literal_eval(k):v for k,v in data['gemm'].items()};cv={ast.literal_eval(k):v for k,v in data['conv'].items()}
    def walk(root,prefix):
        for name,m in list(root.named_children()):
            path=prefix+'.'+name
            if prefix.startswith('cfm') and 'gemm' in parts and isinstance(m,MatrixLinear) and m.precision=='fp8':
                choices={sig[0]:v for sig,v in gm.items() if sig[1:]==(m.out_features,m.in_features,m.bias is not None)}
                if choices:root.add_module(name,SelectedLinear(m,choices));result['gemm'].append(path)
            elif prefix.startswith('vocoder') and 'conv' in parts and isinstance(m,MatrixConv) and m.precision=='bf16':
                sig=(m.in_channels,m.out_channels,m.kernel_size[0],m.stride[0],m.padding[0],m.dilation[0],m.transpose,m.output_padding[0])
                choices={key[-2:]:v for key,v in cv.items() if key[:8]==sig}
                if choices:root.add_module(name,SelectedConv(m,choices));result['conv'].append(path)
            else:walk(m,path)
    walk(engine.student.model,'cfm');walk(engine.tts.bigvgan,'vocoder')
    from acc_infer_clear.ops.triton.stage2_fusions import install
    fs=set(parts)&{'norm','gate','rope'}
    if fs:result['fusions']=install(engine.student.model,fs)
    engine._stage2_parts=done|set(parts)
    return result
