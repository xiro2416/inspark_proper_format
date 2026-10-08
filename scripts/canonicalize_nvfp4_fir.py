"""Reuse the existing complete-halo FIR fusion; preserve native FP4 GEMMs."""
import argparse,json
from pathlib import Path
import numpy as np
import onnx
import onnx_graphsurgeon as gs
import torch
from trt113_provenance import capture_onnx_artifact,file_record

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--static-indices-from',type=Path);p.add_argument('--vocoder-config',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 graph=gs.import_onnx(onnx.load(a.source));nodes={n.name:n for n in graph.nodes};state=torch.load(a.checkpoint,map_location='cpu',weights_only=True,mmap=True);state=state.get('generator',state);changed=[]
 if a.static_indices_from and a.vocoder_config:raise ValueError('Select one source of fixed geometry')
 if a.vocoder_config:
  rates=json.loads(a.vocoder_config.read_text())['upsample_rates'];frames=int(graph.inputs[0].shape[2]);windows=[n for n in graph.nodes if n.name.endswith('/single_window')]
  if windows and len(windows)!=76:raise ValueError('Window coverage changed')
  def constant(t):
   if isinstance(t,gs.Constant):return t.values
   if len(t.inputs)==1 and t.inputs[0].op=='Constant':return t.inputs[0].attrs['value'].values
   raise ValueError('Window offset must be immutable')
  for node in windows:
   prefix=node.name.rsplit('/',2)[0];stage=int(prefix.split('.')[1].split('/')[0]);stage=stage//3 if prefix.startswith('/resblocks.') else stage
   count=frames*int(np.prod(rates[:stage+1]));offsets=[]
   for column in node.inputs[1].inputs[0].inputs:
    index=column.inputs[0].inputs[0];producer=index.inputs[0]
    if producer.op=='Range':offsets.append(0)
    elif producer.op=='Add':offsets.append(int(np.asarray(constant(producer.inputs[1])).reshape(-1)[0]))
    else:raise ValueError('Window index pattern changed')
   node.inputs[1]=gs.Constant(prefix+'/static_window_indices',np.arange(count,dtype=np.int64)[:,None]+np.asarray(offsets,dtype=np.int64)[None,:])
 if a.static_indices_from:
  constants=gs.import_onnx(onnx.load(a.static_indices_from)).tensors();windows=[n for n in graph.nodes if n.name.endswith('/single_window')]
  if windows and len(windows)!=76:raise ValueError('Window coverage changed')
  for node in windows:
   name=node.name.rsplit('/',2)[0]+'/static_window_indices';source=constants[name]
   if not isinstance(source,gs.Constant):raise ValueError('Expected audited static indices')
   node.inputs[1]=gs.Constant(name,np.asarray(source.values,dtype=np.int64).copy())
 for prefix in [f'/resblocks.{r}/activations.{v}' for r in range(18) for v in range(6)]+['/activation_post']:
  x=nodes[prefix+'/Pad'].inputs[0];tail=nodes[prefix+'/Conv_1'];outputs=list(tail.outputs);tail.outputs.clear();key=prefix.lstrip('/').replace('/','.')
  parameters=[]
  for label,suffix in [('up','upsample.filter'),('down','downsample.lowpass.filter'),('alpha','act.alpha'),('beta','act.beta')]:
   array=state[key+'.'+suffix].detach().float().numpy().reshape(-1).copy()
   if label in ['up','down'] and array.size!=12:raise ValueError('Expected complete12-tap FIR')
   parameters.append(gs.Constant(prefix+'/'+label,array))
  boundary=gs.Variable(prefix+'/fir_fp32_boundary',dtype=np.float32)
  graph.nodes.append(gs.Node(op='Cast',name=prefix+'/fir_fp32_boundary',inputs=[x],outputs=[boundary],attrs={'to':onnx.TensorProto.FLOAT}))
  graph.nodes.append(gs.Node(op='small_fir_activation_tiled',domain='inspark_custom',name=prefix+'/ExistingTiledFIR',inputs=[boundary,*parameters],outputs=outputs,attrs={'plugin_namespace':'inspark_custom','plugin_version':'1','aot':1}));changed.append(prefix)
 graph.cleanup().toposort();model=gs.export_onnx(graph);model.opset_import.append(onnx.helper.make_opsetid('inspark_custom',1));a.out.parent.mkdir(parents=True,exist_ok=True)
 onnx.save_model(model,a.out,save_as_external_data=True,all_tensors_to_one_file=True,location=a.out.name+'.data',size_threshold=1024);onnx.checker.check_model(str(a.out));artifact=capture_onnx_artifact(a.out);meta=json.loads(a.source.with_suffix('.export.json').read_text())
 meta.update(onnx=str(a.out.resolve()),onnx_sha256=artifact['sha256'],onnx_artifact=artifact,plugins=['inspark_custom::small_fir_activation_tiled'],custom_small_fir={'changed':changed,'checkpoint':file_record(a.checkpoint,'activation_parameters'),'math_source':file_record('src/inspark_infer/ops/triton/vocoder_tiled_fir.py','unchanged_existing_math'),'source':file_record(a.source,'source_onnx'),'new_gpu_math':False,'static_window_indices':file_record(a.static_indices_from,'audited_window_indices') if a.static_indices_from else file_record(a.vocoder_config,'fixed_vocoder_geometry') if a.vocoder_config else None})
 a.out.with_suffix('.export.json').write_text(json.dumps(meta,indent=2));print(json.dumps({'fused_activations':len(changed),'learned_quantization_unchanged':True}),flush=True)
if __name__=='__main__':main()
