"""Offline weight conversion and representative-shape tile selection.

Compute modules live in ops; preparation is forbidden after admission/capture.
FP32 interfaces/norms are retained to isolate matrix arithmetic changes.
"""
import math
import sys
from dataclasses import asdict
import torch
from torch import nn
from transformers.pytorch_utils import Conv1D
from acc_infer_clear.ops.planning.planner import DeviceCaps, OpSignature, candidates
from acc_infer_clear.quantization.weights import pack_weight
from acc_infer_clear.ops.eager.matrix import MatrixConv, MatrixLinear

def configure_conv(self,module,precision,caps,plans):
    if precision not in ('bf16','fp8'):raise ValueError(precision)
    self.precision=precision;self.transpose=isinstance(module,nn.ConvTranspose1d)
    for name in ('in_channels','out_channels','kernel_size','stride','padding','dilation','groups'):
        setattr(self,name,getattr(module,name))
    self.output_padding=getattr(module,'output_padding',(0,))
    if self.groups!=1 or module.padding_mode!='zeros':raise ValueError('Only groups1/zero-padding learned convolutions')
    weight=module.weight.detach()
    # Old-style weight_norm must be recomputed from loaded g/v before export.
    for hook in module._forward_pre_hooks.values():
        if getattr(hook,'name',None)=='weight' and hasattr(hook,'compute_weight'):
            weight=hook.compute_weight(module).detach()
    self.register_buffer('bias',module.bias.detach().clone() if module.bias is not None else None)
    if precision=='bf16':
        self.register_buffer('weight',weight.bfloat16().contiguous())
        if self.bias is not None:self.bias=self.bias.bfloat16()
        return
    if not caps.native_fp8:raise ValueError('Native FP8 unavailable on this device')
    from acc_infer_clear.ops.triton.fp8_conv import conv1d
    ci,co,kw=self.in_channels,self.out_channels,self.kernel_size[0]
    dense=weight.permute(1,0,2) if self.transpose else weight
    packed,scales=pack_weight(dense.reshape(co,ci*kw))
    self.register_buffer('weight',packed);self.register_buffer('scales',scales)
    self.tiles={}
    for extent in (64,512,4096):
        key=('conv',extent,ci,co,kw,self.stride,self.padding,self.dilation,self.transpose)
        x=weight.new_zeros(1,ci,max(extent,kw*self.dilation[0]+1))
        if key not in plans:
            options=[]
            for tile,estimate in candidates(caps,OpSignature(extent,co,ci*kw),limit=3):
                def fn():return conv1d(x,packed,scales,self.bias,ci,co,kw,self.stride[0],self.padding[0],self.dilation[0],self.transpose,self.output_padding[0],tile)
                for _ in range(2):fn()
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(5):fn()
                end.record();end.synchronize();options.append((start.elapsed_time(end)/5,tile))
            plans[key]=min(options,key=lambda item:item[0])[1]
        self.tiles[extent]=plans[key]
        conv1d(x,packed,scales,self.bias,ci,co,kw,self.stride[0],self.padding[0],self.dilation[0],self.transpose,self.output_padding[0],self.tiles[extent])

def configure_linear(self,module,precision,caps,plans,extents):
    if precision not in ('bf16','fp8'):raise ValueError(precision)
    self.precision=precision
    weight=module.weight.detach().t() if isinstance(module,Conv1D) else module.weight.detach()
    n,k=weight.shape;self.in_features=k;self.out_features=n
    self.register_buffer('bias',module.bias.detach().clone() if module.bias is not None else None)
    if precision=='bf16':
        self.register_buffer('weight',weight.bfloat16().contiguous())
        if self.bias is not None:self.bias=self.bias.bfloat16()
        return
    if not caps.native_fp8:raise ValueError('Native FP8 unavailable on this device')
    from acc_infer_clear.ops.triton.fp8 import linear
    packed,scales=pack_weight(weight)
    self.register_buffer('weight',packed);self.register_buffer('scales',scales)
    self.tiles={}
    for extent in extents:
        key=(extent,n,k,self.bias is not None)
        if key not in plans:
            # Offline representative-shape measurement; no online learning.
            x=torch.randn(extent,k,device=weight.device,dtype=torch.float32,generator=torch.Generator(device=weight.device).manual_seed(0))
            choices=[]
            signature=OpSignature(extent,n,k)
            unsplit=candidates(caps,signature,limit=3)
            needs_more_jobs=any(info['jobs']<caps.sms*info['resident_ctas'] for _,info in unsplit)
            for tile,estimate in candidates(caps,signature,limit=6,split_k=needs_more_jobs):
                fn=lambda:linear(x,packed,scales,self.bias,tile)
                for _ in range(2):fn()
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(5):fn()
                end.record();end.synchronize()
                choices.append((start.elapsed_time(end)/5,tile))
            plans[key]=min(choices,key=lambda item:item[0])[1]
        self.tiles[extent]=plans[key]
    self._last_extent=max(self.tiles)
    # HAS_BIAS and operand signatures must be compiled even when a shape plan
    # was reused from a different module. Never leave first-use JIT to serving.
    warm=weight.new_zeros(8,k,dtype=torch.float32)
    for tile in set(self.tiles.values()):linear(warm,self.weight,self.scales,self.bias,tile)

def prepare(engine,mode,components,convolutions=False):
    if engine.sessions or engine.head_graphs is not None or getattr(engine.rt.target,'graph_sealed',False):
        raise RuntimeError('Precision must be prepared before requests and captures')
    if mode not in ('bf16','fp8'):raise ValueError(mode)
    from acc_infer_clear.ops.planning.plan_cache import PlanCache
    caps=DeviceCaps.current();plans=PlanCache(engine,'matrices_'+mode);manifest=[]
    if convolutions and mode=='fp8':
        from acc_infer_clear.ops.triton.fp8_conv import prepare_scale_kernels
        prepare_scale_kernels(next(engine.tts.gpt.parameters()).device)
    roots={'target':engine.tts.gpt.gpt.h,'draft':engine.rt.engine.draft.layers,
           'cfm':engine.student.model.transformer.layers}
    target_extents=tuple(sorted({b*q for b in range(1,engine.config['max_batch']+1) for q in (8,64,128,256)}|{4096}))
    prompt_lengths={v['values']['voice.cache_mel'].shape[-1] for v in engine.model.bank.entries.values()}
    cfm_extents=tuple(sorted({b*(p+52) for b in range(1,engine.config['max_batch']+1) for p in prompt_lengths}|{4096}))
    draft_extents=tuple(sorted({b*7 for b in range(1,engine.config['max_batch']+1)}|{64,256,512,1024,4096}))
    extents_by_component={'target':target_extents,'cfm':cfm_extents,'draft':draft_extents}
    def replace(module,prefix,precision):
        for name,child in list(module.named_children()):
            path=prefix+'.'+name
            if isinstance(child,(nn.Linear,Conv1D)):
                if '.attention_norm.' in path or '.ffn_norm.' in path:continue
                shape=list(child.weight.shape)
                module.add_module(name,MatrixLinear(child,precision,caps,plans,extents=extents_by_component.get(prefix.split('.')[0],(8,64,256,1024,4096))))
                manifest.append(dict(path=path,precision=precision,original_shape=shape))
            elif convolutions and isinstance(child,(nn.Conv1d,nn.ConvTranspose1d)):
                shape=list(child.weight.shape)
                module.add_module(name,MatrixConv(child,precision,caps,plans))
                manifest.append(dict(path=path,precision=precision,original_shape=shape,convolution=True))
            else:replace(child,path,precision)
    for component in components:
        print('PRECISION_PREP',component,mode,flush=True,file=sys.stderr)
        if component=='vocoder':
            if not convolutions:raise ValueError('Vocoder requires explicit convolution experiment')
            vocoder=engine.tts.bigvgan;cutoff=math.ceil(vocoder.num_upsamples/4)
            for index in range(vocoder.num_upsamples):
                precision='bf16' if mode=='bf16' or index<cutoff else 'fp8'
                replace(vocoder.ups[index],f'vocoder.stages.{index}.ups',precision)
                for j in range(vocoder.num_kernels):replace(vocoder.resblocks[index*vocoder.num_kernels+j],f'vocoder.stages.{index}.resblock{j}',precision)
            # Protect input/output boundaries; not counted as extra FP8 depth.
            for name in ('conv_pre','conv_post'):
                child=getattr(vocoder,name);setattr(vocoder,name,MatrixConv(child,'bf16',caps,plans))
                manifest.append(dict(path='vocoder.'+name,precision='bf16',boundary=True))
            continue
        layers=roots[component];cutoff=math.ceil(len(layers)/4)
        for index,block in enumerate(layers):
            precision='bf16' if mode=='bf16' or index<cutoff else 'fp8'
            replace(block,f'{component}.blocks.{index}',precision)
        if component=='cfm' and convolutions:
            model=engine.student.model
            for name in ('in_layers','res_skip_layers'):
                replace(getattr(model.wavenet,name),'cfm.wavenet.'+name,mode)
    return dict(mode=mode,components=components,matrices=manifest,device=asdict(caps),
                cache=plans.save(),
                plans={str(k):asdict(v) for k,v in plans.items()},interface_precision='FP32',
                observed_head_shape_extents={k:list(v) for k,v in extents_by_component.items()},
                convolution_quantized=convolutions,rnn_quantized=False,kv_quantized=False,
                calibrated_quality=False,online_tuning=False)
