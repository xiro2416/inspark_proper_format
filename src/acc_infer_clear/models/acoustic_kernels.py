"""Explicit opt-in deployment of numerically tested acoustic kernels."""
import ast,hashlib,json
from pathlib import Path
from dataclasses import asdict
import torch,triton
from .precision import MatrixConv
from acc_infer_clear.kernels.planner import DeviceCaps,Tile
from acc_infer_clear.kernels.fp8_conv_ntc import NTCConv

def identity():
    root=Path(__file__).resolve().parents[1]
    files=['kernels/fp8_conv_ntc.py','kernels/fp8_conv.py','kernels/planner.py','models/precision.py']
    return dict(device=asdict(DeviceCaps.current()),torch=torch.__version__,triton=triton.__version__,cuda=torch.version.cuda,
                sources={f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in files})

def signature(module,b,t):
    return (module.in_channels,module.out_channels,module.kernel_size[0],module.stride[0],module.padding[0],
            module.dilation[0],module.transpose,module.output_padding[0],b,t)

def prepare(engine,plan_path,mode='both'):
    if mode not in ('alias','ntc','both'):raise ValueError('Unknown acoustic mode')
    if engine.sessions or engine.head_graphs is not None or getattr(engine.rt.target,'graph_sealed',False):
        raise RuntimeError('Prepare acoustic kernels before requests and CUDA Graph capture')
    done=getattr(engine,'_acoustic_prepared',set())
    requested={'alias','ntc'} if mode=='both' else {mode}
    if done & requested:raise RuntimeError('Acoustic kernel already prepared')
    result=dict(mode=mode,convolutions=[],alias_modules=[],online_tuning=False)
    if mode in ('ntc','both'):
        data=json.loads(Path(plan_path).read_text())
        if data['identity']!=identity():raise ValueError('Acoustic plan/device/source mismatch; regenerate offline')
        plans={ast.literal_eval(k):Tile(**v) for k,v in data['plans'].items()}
        def walk(module,path):
            for name,child in list(module.named_children()):
                key=path+'.'+name
                if isinstance(child,MatrixConv) and child.precision=='fp8':
                    relevant={sig:tile for sig,tile in plans.items() if sig[:8]==signature(child,1,1)[:8]}
                    if relevant:
                        new=NTCConv(child);new.tiles={sig[-2:]:tile for sig,tile in relevant.items()}
                        # Encodec wrappers inspect convolution geometry before calling it.
                        for attr in ('in_channels','out_channels','kernel_size','stride','padding','dilation','groups','output_padding','transpose','precision'):
                            setattr(new,attr,getattr(child,attr))
                        module.add_module(name,new);result['convolutions'].append(dict(path=key,shapes=[list(k) for k in new.tiles]))
                else:walk(child,key)
        walk(engine.student.model,'cfm');walk(engine.tts.bigvgan,'vocoder')
        result['plan']=str(plan_path)
    if mode in ('alias','both'):
        from acc_infer_clear.kernels.alias_free import install
        result['alias_modules']=install(engine.tts.bigvgan)
    engine._acoustic_prepared=done|requested
    return result
