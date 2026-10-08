"""Real INT8 buffers and standard decomposed Q/DQ operators for ONNX export.

ModelOpt fake quantization is used internally for calibration/reference only.
These modules represent signed INT8 weights, scaling and activation Q/DQ explicitly.
"""
import torch,math,inspect,textwrap,types
import torch.nn.functional as F
import torch.ao.quantization.fx._decomposed  # registers standard decomposed operators

class PackedWeighted(torch.nn.Module):
 def __init__(self,z,prototype=False):
  super().__init__();w=z.weight.detach().float();dims=tuple(range(1,w.ndim))
  a=w.abs().amax(dim=dims,keepdim=True) if prototype else z.weight_quantizer.amax.detach().float()
  # Tensor/Tensor division matches ModelOpt's FP32 bound/amax computation.
  # Python scalar division can lower to reciprocal*127 and change half-integer rounding.
  reciprocal=torch.where(a>2**-24,torch.tensor(127.,device=a.device)/a,torch.ones_like(a))
  q=torch.clamp(torch.round(w*reciprocal),-128,127).to(torch.int8)
  self.register_buffer('weight_int8',q.clone());self.register_buffer('weight_scale',(1/reciprocal).reshape(-1).clone());self.register_buffer('weight_zero_point',torch.zeros(w.shape[0],dtype=torch.int64,device=w.device))
  aa=torch.tensor(10.,device=w.device) if prototype else z.input_quantizer.amax.detach().float().reshape(())
  assert aa>2**-24
  self.register_buffer('input_scale',(aa/127).clone());self.register_buffer('input_zero_point',torch.zeros((),dtype=torch.int64,device=w.device))
  s=torch.ones(w.shape[1],device=w.device) if prototype and w.ndim==2 else None if prototype else z.input_quantizer.pre_quant_scale
  self.register_buffer('pre_quant_scale',s.detach().clone() if s is not None else None)
  self.register_buffer('bias',z.bias.detach().clone() if z.bias is not None else None)
  self.narrow_range=False if prototype else z.input_quantizer._narrow_range
  self.linear=(w.ndim==2)
  if not self.linear:
   self.stride=z.stride;self.padding=z.padding;self.dilation=z.dilation;self.groups=z.groups
 def forward(self,x):
  if self.pre_quant_scale is not None:x=x*self.pre_quant_scale
  if self.narrow_range:x=torch.clamp(x,min=-127*self.input_scale,max=127*self.input_scale)
  q=torch.ops.quantized_decomposed.quantize_per_tensor.tensor(x,self.input_scale,self.input_zero_point,-128,127,torch.int8)
  x=torch.ops.quantized_decomposed.dequantize_per_tensor.tensor(q,self.input_scale,self.input_zero_point,-128,127,torch.int8)
  w=torch.ops.quantized_decomposed.dequantize_per_channel.default(self.weight_int8,self.weight_scale,self.weight_zero_point,0,-128,127,torch.int8)
  return F.linear(x,w,self.bias) if self.linear else F.conv1d(x,w,self.bias,self.stride,self.padding,self.dilation,self.groups)

class DynamicPosition(torch.nn.Module):
 """Original formula, generate requested offsets without mutating a cached buffer."""
 def __init__(self,z):super().__init__();self.embed_dim=z.embed_dim;self.length_factor=z.length_factor;self.dropout=z.dropout
 def forward(self,x,left_context_len=0):
  t=x.shape[0];offsets=torch.arange(-(t+left_context_len-1),t,device=x.device,dtype=torch.float32).unsqueeze(1)
  freq=1+torch.arange(self.embed_dim//2,device=x.device);c=self.embed_dim**.5
  compressed=c*offsets.sign()*((offsets.abs()+c).log()-math.log(c));scale=self.length_factor*self.embed_dim/(2*math.pi)
  atan=(compressed/scale).atan();pe=torch.stack(((atan*freq).cos(),(atan*freq).sin()),dim=-1).flatten(-2)
  pe=torch.cat((pe[:,:-1],torch.ones_like(pe[:,-1:])),dim=1)
  return self.dropout(pe.to(x.dtype).unsqueeze(0))

class DynamicDownsample(torch.nn.Module):
 """Same repeated-last-frame padding, represented by bounded index selection."""
 def __init__(self,z):super().__init__();self.downsample=z.downsample;self.bias=z.bias
 def forward(self,src):
  t,b,c=src.shape;ds=self.downsample;n=(t+ds-1)//ds
  indices=torch.arange(n*ds,device=src.device).clamp_max(t-1)
  src=src.index_select(0,indices).reshape(n,ds,b,c)
  return (src*self.bias.softmax(0).reshape(ds,1,1)).sum(1)

def export_positions(m):
 from inspark_infer.models.zipvoice.reference.models.modules import zipformer
 from inspark_infer.models.zipvoice.reference.models.modules.zipformer import CompactRelPositionalEncoding,RelPositionMultiheadAttentionWeights,SimpleDownsample
 # TorchExport specializes frame counts for the original as_strided(storage_offset).
 # The source already defines the equivalent Gather for tracing; use it for Dynamo too.
 source=textwrap.dedent(inspect.getsource(RelPositionMultiheadAttentionWeights.forward))
 source=source.replace('torch.jit.is_scripting() or torch.jit.is_tracing()','True').replace('elif random.random() < 0.001 and not self.training:','elif False:').replace('if torch.jit.is_tracing():','if True:').replace('torch.arange(start=time1 - 1, end=-1, step=-1)','torch.arange(start=time1 - 1, end=-1, step=-1,device=pos_scores.device)').replace('torch.arange(seq_len)','torch.arange(seq_len,device=pos_scores.device)')
 start=source.index('            rows = torch.arange(');end=source.index('        else:',start)
 source=source[:start]+'''            indexes=(torch.arange(time1-1,-1,-1,device=pos_scores.device)[:,None]+torch.arange(seq_len,device=pos_scores.device)[None,:])
            pos_scores=torch.gather(pos_scores,-1,indexes[None,None].expand(num_heads,batch_size,time1,seq_len))
'''+source[end:]
 namespace=dict(vars(zipformer));exec(compile(source,str(__file__),"exec"),namespace)
 forward=namespace['forward']
 for name,z in list(m.named_modules()):
  if isinstance(z,RelPositionMultiheadAttentionWeights):z.forward=types.MethodType(forward,z)
  if isinstance(z,SimpleDownsample):
   parent,attr=name.rsplit('.',1);setattr(m.get_submodule(parent),attr,DynamicDownsample(z))
  if isinstance(z,CompactRelPositionalEncoding):
   parent,attr=name.rsplit('.',1);setattr(m.get_submodule(parent),attr,DynamicPosition(z))
 return m
