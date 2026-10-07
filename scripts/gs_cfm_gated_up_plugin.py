"""Replace audited FP8 gate/up/SwiGLU/QDQ with a paired-GEMM plugin."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
import onnx,onnx_graphsurgeon as gs
from trt113_provenance import capture_onnx_artifact,file_record

def immutable(t):
 if isinstance(t,gs.Constant):return np.asarray(t.values)
 n=t.inputs[0]
 if n.op=='Identity':return immutable(n.inputs[0])
 if n.op=='Constant':return np.asarray(n.attrs['value'].values)
 raise ValueError(n.op)

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--single',action='store_true');args=p.parse_args()
 g=gs.import_onnx(onnx.load(args.source));nodes={n.name:n for n in g.nodes};changed=[]
 ends=[n for n in g.nodes if n.op=='DequantizeLinear' and '/feed_forward/w2' in n.name and n.name.endswith('/DequantizeLinear')]
 for end in ends:
  prefix=end.name.rsplit('/w2',1)[0]+'/';suffix=end.name.split('/w2',1)[1].split('/',1)[0]
  if args.single:prefix='/model/transformer/layers.4/feed_forward/';suffix=''
  names={k:prefix+k+suffix for k in ('w1','w3','w2')}
  q1=nodes.get(names['w1']+'/QuantizeLinear_1');q3=nodes.get(names['w3']+'/QuantizeLinear_1')
  if q1 is None or q3 is None:continue
  # Never materialize lazy FP8 zero tensors: GS would change their dtype.
  zero_tensor=q1.inputs[2]
  while not isinstance(zero_tensor,gs.Constant):zero_tensor=zero_tensor.inputs[0].inputs[0]
  raw=getattr(zero_tensor._values,'tensor',None)
  zero_is_zero=(raw is not None and not any(raw.raw_data))
  # Only the declared FP8 blocks: BF16 blocks have no weight QuantizeLinear.
  ws=[immutable(q.inputs[1]).astype(np.float32) for q in (q1,q3)]
  w=[immutable(q.inputs[0]).astype(np.float32) for q in (q1,q3)]
  a=float(immutable(nodes[names['w1']+'/QuantizeLinear'].inputs[1]));a3=float(immutable(nodes[names['w3']+'/QuantizeLinear'].inputs[1]))
  if a!=a3 or any(v.shape!=(1536,512) for v in w):raise ValueError('Unexpected gate/up recipe/geometry')
  if not zero_is_zero:raise ValueError('Nonzero FP8 zero point')
  output_scale=float(immutable(nodes[names['w2']+'/QuantizeLinear'].inputs[1]))
  x=nodes[names['w1']+'/Cast'].inputs[0];old_quant=end.inputs[0].inputs[0];out=end.inputs[0]
  packed=[(torch.from_numpy(v.copy())/torch.from_numpy(s.copy())[:,None]).clamp(-448,448).to(torch.float8_e4m3fn).view(torch.int8).numpy().copy() for v,s in zip(w,ws)]
  label='custom_gated_up_'+str(len(changed));old_quant.outputs=[]
  quant_x=gs.Variable(label+'_input_fp8_bytes',dtype=np.int32,shape=[64,310,128])
  g.nodes.append(gs.Node(op='fp8_quantize',domain='inspark_custom',name=label+'_quant',attrs={'plugin_namespace':'inspark_custom','aot':1,'input_scale':a},inputs=[x],outputs=[quant_x]))
  inputs=[quant_x,gs.Constant(label+'_w1',packed[0].view(np.int32)),gs.Constant(label+'_w3',packed[1].view(np.int32)),gs.Constant(label+'_s1',ws[0]*np.float32(a)),gs.Constant(label+'_s3',ws[1]*np.float32(a))]
  g.nodes.append(gs.Node(op='fp8_gated_up',domain='inspark_custom',name=label,attrs={'plugin_namespace':'inspark_custom','aot':1,'input_scale':a,'output_scale':output_scale},inputs=inputs,outputs=[out]))
  changed.append(dict(path=prefix,suffix=suffix,input_scale=a,output_scale=output_scale,quantized_weight_bits='E4M3FN packed four-per-INT32 word as opaque bytes',shape=[64,310,512],dtype='native FP8 output; original DequantizeLinear retained'))
  if args.single:break
 if len(changed)!=(1 if args.single else 36):raise ValueError('Unexpected number of FP8 SwiGLU blocks '+str(len(changed)))
 g.cleanup().toposort();result=gs.export_onnx(g);result.opset_import.append(onnx.helper.make_opsetid("inspark_custom",1));onnx.checker.check_model(result)
 args.output.parent.mkdir(parents=True,exist_ok=True);onnx.save(result,str(args.output));artifact=capture_onnx_artifact(args.output)
 meta=json.loads(args.source.with_suffix('.export.json').read_text());meta.update(onnx=str(args.output.resolve()),onnx_sha256=artifact['sha256'],onnx_artifact=artifact,plugins=['inspark_custom::fp8_quantize','inspark_custom::fp8_gated_up'],custom_gated_up=dict(source=file_record(args.source,'source_onnx'),changed=changed,math_source=file_record('src/inspark_infer/ops/triton/cfm_gated_up.py','custom_math_source')))
 args.output.with_suffix('.export.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(dict(changed=len(changed),nodes=len(g.nodes))),flush=True)

if __name__=='__main__':main()
