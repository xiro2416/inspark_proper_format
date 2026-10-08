"""Standard Torch/ONNX geometry around NVIDIA NVFP4 Linear quantizers.

No GPU math kernel is implemented here. Learned convolution is represented as
exact padding/gather/reshape plus a Linear for TensorRT's native FP4 GEMM.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F

class MatrixWeightOp(nn.Module):
    def __init__(self,module,*,dtype=torch.float32,conv_layout="channel_tap"):
        if conv_layout not in ("channel_tap","tap_channel"):raise ValueError(conv_layout)
        self.conv_layout=conv_layout
        super().__init__()
        conv=isinstance(module,(nn.Conv1d,nn.ConvTranspose1d))
        transpose=isinstance(module,nn.ConvTranspose1d)
        weight=module.weight.detach().float()
        for hook in module._forward_pre_hooks.values():
            if getattr(hook,'name',None)=='weight' and hasattr(hook,'compute_weight'):
                weight=hook.compute_weight(module).detach().float()
        if type(module).__name__=='Conv1D':weight=weight.t()
        self.kind='conv_transpose1d' if transpose else 'conv1d' if conv else 'linear'
        self.quantized=False;self.precision='fp32';self.spec={}
        if conv:
            if module.groups!=1 or module.padding_mode!='zeros':raise ValueError('Only existing groups1/zero-padding roles are supported')
            self.stride=module.stride;self.padding=module.padding;self.dilation=module.dilation;self.groups=1;self.padding_mode="zeros"
            self.output_padding=getattr(module,'output_padding',(0,))
            self.kernel_size=module.kernel_size;self.in_channels=module.in_channels;self.out_channels=module.out_channels
            if transpose:weight=weight.transpose(0,1).flip(-1).contiguous()
            if conv_layout=="tap_channel":weight=weight.permute(0,2,1).contiguous()
            weight=weight.flatten(1)
        self.logical_k=weight.shape[1];self.padded_k=math.ceil(self.logical_k/16)*16
        self.in_features=self.logical_k;self.out_features=weight.shape[0]
        self.nvfp4_linear=nn.Linear(self.padded_k,self.out_features,bias=False,device=weight.device,dtype=dtype)
        self.nvfp4_linear.weight.data.copy_(F.pad(weight,(0,self.padded_k-self.logical_k)).to(dtype))
        self.register_buffer('bias',None if module.bias is None else module.bias.detach().float().clone())

    @property
    def weight(self):return self.nvfp4_linear.weight

    def matrix_input(self,x):
        if self.kind=='linear':return F.pad(x,(0,self.padded_k-self.logical_k))
        if self.kind=='conv_transpose1d':
            if self.stride[0]>1:
                x=F.pad(x.unsqueeze(-1),(0,self.stride[0]-1)).flatten(-2)
                x=x[...,:-(self.stride[0]-1)]
            pad=self.dilation[0]*(self.kernel_size[0]-1)-self.padding[0]
            x=F.pad(x,(pad,pad+self.output_padding[0]));stride=1;pad=0
        else:stride=self.stride[0];pad=self.padding[0]
        if pad:x=F.pad(x,(pad,pad))
        count=(x.shape[-1]-self.dilation[0]*(self.kernel_size[0]-1)-1)//stride+1
        positions=torch.arange(count,device=x.device)*stride
        if self.conv_layout=="tap_channel":
            # One channel-contiguous source, one gather and a reshape.
            # GEMM K follows [tap,channel]; weights use the identical order.
            indices=(positions[:,None]+torch.arange(self.kernel_size[0],device=x.device)[None,:]*self.dilation[0]).flatten()
            source=x.transpose(1,2).contiguous()
            matrix=source.index_select(1,indices).reshape(x.shape[0],count,-1)
            return F.pad(matrix,(0,self.padded_k-self.logical_k))
        windows=torch.stack([x.index_select(-1,positions+i*self.dilation[0]) for i in range(self.kernel_size[0])],dim=-1)
        # [B,C,L,K] -> [B,L,C*K], exactly the canonical learned weight [O,C,K].
        matrix=windows.permute(0,2,1,3).flatten(-2)
        return F.pad(matrix,(0,self.padded_k-self.logical_k))

    def forward(self,x):
        matrix=self.matrix_input(x).to(self.nvfp4_linear.weight.dtype)
        y=self.nvfp4_linear(matrix).float()
        if self.bias is not None:y=y+self.bias
        return y if self.kind=='linear' else y.transpose(1,2).contiguous()

    def apply_nvfp4(self,activation_amax):
        import modelopt.torch.quantization as mtq
        self.nvfp4_linear.half()
        a=torch.as_tensor(activation_amax,device=self.weight.device,dtype=torch.float32)
        maximum=a.max().half().float()
        statistics=torch.zeros(1,self.padded_k,device=self.weight.device,dtype=torch.float16)
        statistics[0,0]=maximum
        mtq.quantize(self,mtq.NVFP4_DEFAULT_CFG,lambda m:m.nvfp4_linear(statistics))
        self.nvfp4_linear.input_quantizer.trt_high_precision_dtype='Half'
        self.nvfp4_linear.weight_quantizer.trt_high_precision_dtype='Half'
        self.quantized=True;self.precision='nvfp4'
        self.spec={'precision':'nvfp4','block_size':16,'data_format':'e2m1','block_scale_format':'e4m3',
            'activation_amax':float(maximum),'input_layout':'last-dimension GEMM K','logical_k':self.logical_k,'padded_k':self.padded_k,
            'geometry':self.kind,'conv_layout':self.conv_layout,'output_compute_interface':'fp32','quantized_gemm_interface':'fp16'}
        return self

def canonicalize_nvfp4_matmul_interfaces(model):
    """Match ModelOpt weight DQ output to the actual FP16 Linear interface.

    TensorRT folds this conversion into native FP4 GEMM. The operator probe's
    inspector and CUDA trace must verify that fusion for each new graph family.
    """
    import onnx
    from onnx import helper
    # ModelOpt's mixed-BF16 postprocessor inserts extra *_bf16 conversions on
    # FP4 DQ consumers. Keep original protected Cast nodes, remove only those
    # generated suffix aliases: FP32 FIR/normalization interfaces must not
    # inherit a neighbouring low-precision learned operator's dtype.
    aliases={};generated=[]
    for n in model.graph.node:
        if n.op_type=='Cast' and n.output[0].endswith('_bf16') and next((a.i for a in n.attribute if a.name=='to'),None)==onnx.TensorProto.BFLOAT16:
            aliases[n.output[0]]=n.input[0];generated.append(n)
    def original(value):
        while value in aliases:value=aliases[value]
        return value
    for n in model.graph.node:
        for i,value in enumerate(n.input):n.input[i]=original(value)
    for n in generated:model.graph.node.remove(n)
    for v in list(model.graph.value_info):
        if v.name in aliases:model.graph.value_info.remove(v)
    producers={output:n for n in model.graph.node for output in n.output}
    def bypass_exporter_bf16(value):
        producer=producers.get(value)
        if producer is not None and producer.op_type=='Cast' and value.endswith('_bf16'):
            target=next((a.i for a in producer.attribute if a.name=='to'),None)
            if target==onnx.TensorProto.BFLOAT16:return producer.input[0]
        return value
    for node in list(model.graph.node):
        if node.op_type=='MatMul' and 'nvfp4_linear' in node.name:
            activation=bypass_exporter_bf16(node.input[0]);act_target=activation+'_nvfp4_interface_half'
            act_cast=helper.make_node('Cast',[activation],[act_target],name=node.name+'_activation_half',to=onnx.TensorProto.FLOAT16)
            node.input[0]=act_target;model.graph.node.insert(list(model.graph.node).index(node),act_cast)
            value=bypass_exporter_bf16(node.input[1]);target=value+'_nvfp4_interface_half'
            cast=helper.make_node('Cast',[value],[target],name=node.name+'_weight_half',to=onnx.TensorProto.FLOAT16)
            node.input[1]=target;model.graph.node.insert(list(model.graph.node).index(node),cast)
            original=node.output[0];node.output[0]=original+'_nvfp4_half'
            restore=helper.make_node('Cast',[node.output[0]],[original],name=node.name+'_fp32_boundary',to=onnx.TensorProto.FLOAT)
            model.graph.node.insert(list(model.graph.node).index(node)+1,restore)
    import re
    counter=0
    for node in list(model.graph.node):
        if node.op_type=='Conv' and ('/activations.' in node.name or 'activation_post' in node.name):
            value=node.input[0];target=value+'_fir_fp32'
            cast=helper.make_node('Cast',[value],[target],name=node.name+'_fir_f32',to=onnx.TensorProto.FLOAT)
            node.input[0]=target;model.graph.node.insert(list(model.graph.node).index(node),cast)
        if node.op_type=='Add' and (re.fullmatch(r'/resblocks\.\d+/Add(?:_\d+)?',node.name) or re.fullmatch(r'/Add(?:_\d+)?',node.name)):
            for i,value in enumerate(node.input):
                target=value+'_residual_f32_'+str(counter);counter+=1
                cast=helper.make_node('Cast',[value],[target],name=node.name+'_residual_f32_'+str(i),to=onnx.TensorProto.FLOAT)
                node.input[i]=target;model.graph.node.insert(list(model.graph.node).index(node),cast)
    return model

def export_nvfp4(model,inputs,path):
    from pathlib import Path
    import onnx
    from modelopt.torch._deploy.utils import torch_onnx as official_export
    from modelopt.torch._deploy.utils.torch_onnx import get_onnx_bytes_and_metadata,OnnxBytes
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    original=official_export.quantize_weights
    def canonical_weight_quantizers(torch_model,graph):
        # Four-step CFM reuses immutable weights. ModelOpt0.47 assigns fixed
        # scale-output names per weight; exporting four identical QDQ nodes
        # otherwise creates duplicate producers. Share only weight QDQ nodes,
        # never the dynamic activation quantizers of different Euler states.
        seen={};rename={};remove=[]
        for node in graph.graph.node:
            if node.op_type=='TRT_FP4QDQ':
                key=(tuple(node.input),tuple((a.name,a.SerializeToString()) for a in node.attribute))
                if key in seen:
                    rename.update(zip(node.output,seen[key].output));remove.append(node)
                else:seen[key]=node
        for node in graph.graph.node:
            for i,name in enumerate(node.input):node.input[i]=rename.get(name,name)
        for node in remove:graph.graph.node.remove(node)
        return original(torch_model,graph)
    official_export.quantize_weights=canonical_weight_quantizers
    try:data,metadata=get_onnx_bytes_and_metadata(model,inputs,model_name=path.stem,onnx_opset=23,weights_dtype='fp32')
    finally:official_export.quantize_weights=original
    OnnxBytes.from_bytes(data).write_to_disk(str(path.parent),clean_dir=False)
    graph=onnx.load(path)
    graph=canonicalize_nvfp4_matmul_interfaces(graph)
    onnx.save_model(graph,path,save_as_external_data=True,all_tensors_to_one_file=True,location=path.name+'.data',size_threshold=1024)
    return metadata
