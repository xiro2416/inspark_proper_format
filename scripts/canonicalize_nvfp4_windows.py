"""Replace per-tap tensor gathers with one indexed window gather.

Preserves [channel,tap] K order and every dynamic FP4 block quantizer. Only
index vectors are concatenated; activation windows are materialized once.
"""
import argparse,json
from pathlib import Path
import numpy as np
import onnx
import onnx_graphsurgeon as gs
from trt113_provenance import capture_onnx_artifact,file_record

def rewrite(graph):
    changed=[]
    for concat in list(graph.nodes):
        if concat.op!='Concat' or concat.attrs.get('axis')!=-1 or len(concat.inputs)<2:continue
        gathers=[]
        for value in concat.inputs:
            if len(value.inputs)!=1:break
            unsqueeze=value.inputs[0]
            if unsqueeze.op!='Unsqueeze' or len(unsqueeze.inputs[0].inputs)!=1:break
            gather=unsqueeze.inputs[0].inputs[0]
            if gather.op!='Gather' or gather.attrs.get('axis')!=-1:break
            gathers.append(gather)
        if len(gathers)!=len(concat.inputs) or len({g.inputs[0].name for g in gathers})!=1:continue
        # One source [B,C,T], one index matrix [L,K] -> exact same [B,C,L,K].
        prefix=concat.name+'/single_window';columns=[]
        axis=gs.Constant(prefix+'/axis',np.array([1],np.int64))
        for i,gather in enumerate(gathers):
            col=gs.Variable(prefix+f'/column{i}',dtype=np.int64)
            graph.nodes.append(gs.Node(op='Unsqueeze',name=prefix+f'/index{i}',inputs=[gather.inputs[1],axis],outputs=[col]));columns.append(col)
        indices=gs.Variable(prefix+'/indices',dtype=np.int64)
        graph.nodes.append(gs.Node(op='Concat',name=prefix+'/indices',attrs={'axis':1},inputs=columns,outputs=[indices]))
        outputs=list(concat.outputs);concat.outputs.clear()
        graph.nodes.append(gs.Node(op='Gather',name=prefix,attrs={'axis':-1},inputs=[gathers[0].inputs[0],indices],outputs=outputs))
        changed.append({'source':gathers[0].inputs[0].name,'taps':len(gathers),'window':prefix})
    graph.cleanup().toposort();return changed

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    graph=gs.import_onnx(onnx.load(a.source));changes=rewrite(graph)
    if not changes:raise ValueError('No eligible windows')
    a.out.parent.mkdir(parents=True,exist_ok=True)
    model=gs.export_onnx(graph)
    onnx.save_model(model,a.out,save_as_external_data=True,all_tensors_to_one_file=True,location=a.out.name+'.data',size_threshold=1024)
    onnx.checker.check_model(str(a.out))
    record=capture_onnx_artifact(a.out);meta=json.loads(a.source.with_suffix('.export.json').read_text())
    meta.update(onnx=str(a.out.resolve()),onnx_sha256=record['sha256'],onnx_artifact=record,
                window_canonicalization={'source':file_record(a.source,'source_onnx'),'changes':changes,'same_K_order':True,'quantizers_unchanged':True})
    a.out.with_suffix('.export.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps({'windows':len(changes),'taps':sum(c['taps'] for c in changes)}))
if __name__=='__main__':main()
