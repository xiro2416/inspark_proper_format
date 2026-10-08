"""Expose BF16-cache attention to TensorRT's existing ONNX Attention operator.

Keep the original BF16 prescaled query, K/V values and verification mask.
Softmax uses framework defaults; fused output is BF16 then restored to FP32. This
rounding boundary is explicit in provenance and requires a final float audit.
"""
import argparse,json
from pathlib import Path
import numpy as np
import onnx
import onnx_graphsurgeon as gs
from trt113_provenance import capture_onnx_artifact,file_record

def rewrite(graph):
 changed=[]
 for softmax in list(graph.nodes):
  if softmax.op!='Softmax':continue
  if softmax.attrs.get('axis',-1)!=-1:raise ValueError('Expected last-axis attention softmax')
  pvs=[n for n in softmax.outputs[0].outputs if n.op=='MatMul' and n.inputs[0] is softmax.outputs[0]]
  if len(pvs)!=1:raise ValueError('Attention PV pattern ambiguous')
  pv=pvs[0];where=softmax.inputs[0].inputs[0]
  if where.op!='Where':raise ValueError('Expected original verification mask')
  sentinel=where.inputs[1]
  if isinstance(sentinel,gs.Constant):sentinel=sentinel.values
  elif len(sentinel.inputs)==1 and sentinel.inputs[0].op=='Constant':sentinel=sentinel.inputs[0].attrs['value'].values
  else:raise ValueError('Attention mask fill must be constant -inf')
  if not np.isneginf(np.asarray(sentinel)).all():raise ValueError('Attention mask fill semantics differ')
  not_mask=where.inputs[0].inputs[0];qk=where.inputs[2].inputs[0]
  if not_mask.op!='Not' or qk.op!='MatMul':raise ValueError('QK/mask contract changed')
  q_cast=qk.inputs[0].inputs[0];k_transpose=qk.inputs[1].inputs[0];k_cast=k_transpose.inputs[0].inputs[0];v_cast=pv.inputs[1].inputs[0]
  if any(n.op!='Cast' or n.attrs.get('to')!=onnx.TensorProto.FLOAT for n in [q_cast,k_cast,v_cast]):raise ValueError('Expected BF16 ->FP32 attention boundaries')
  q,k,v=q_cast.inputs[0],k_cast.inputs[0],v_cast.inputs[0];mask=not_mask.inputs[0]
  prefix=pv.name+'/official';fused=gs.Variable(prefix+'_bf16',dtype=onnx.TensorProto.BFLOAT16);outputs=list(pv.outputs);pv.outputs.clear()
  graph.nodes.append(gs.Node(op='Attention',name=prefix,inputs=[q,k,v,mask],outputs=[fused],attrs={'scale':1.0,'is_causal':0}))
  graph.nodes.append(gs.Node(op='Cast',name=prefix+'_fp32_interface',inputs=[fused],outputs=outputs,attrs={'to':onnx.TensorProto.FLOAT}));changed.append(prefix)
 graph.cleanup().toposort();return changed

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 graph=gs.import_onnx(onnx.load(a.source));changed=rewrite(graph)
 if len(changed) not in (1,24):raise ValueError('Attention coverage changed')
 model=gs.export_onnx(graph);a.out.parent.mkdir(parents=True,exist_ok=True);onnx.save_model(model,a.out,save_as_external_data=True,all_tensors_to_one_file=True,location=a.out.name+'.data',size_threshold=1024);onnx.checker.check_model(str(a.out))
 if a.source.with_suffix('.export.json').exists():
  meta=json.loads(a.source.with_suffix('.export.json').read_text());artifact=capture_onnx_artifact(a.out);meta.update(onnx=str(a.out.resolve()),onnx_sha256=artifact['sha256'],onnx_artifact=artifact,attention_rewrite={'source':file_record(a.source,'source_onnx'),'paths':changed,'QKV':'original BF16 values','query_scaling':'original prescaled query, Attention.scale=1','mask':'original bool verification mask','softmax':'TRT fused attention default; explicit ONNX softmax_precision unsupported','output_rounding':'BF16 ->FP32 interface','new_gpu_math':False});a.out.with_suffix('.export.json').write_text(json.dumps(meta,indent=2))
 print(json.dumps({'attention_paths':len(changed)}),flush=True)
if __name__=='__main__':main()
