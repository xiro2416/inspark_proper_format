"""Fuse exclusively consumed FIR->original smooth Div->QuantizeLinear paths."""
import argparse,json
from pathlib import Path
import numpy as np
import onnx
from onnx import helper,numpy_helper
ROOT=Path(__file__).resolve().parents[2]


def main():
    from trt113_provenance import capture_onnx_artifact,file_record
    from deployment.multibatch.fir_quant_plugin import NAME
    p=argparse.ArgumentParser();p.add_argument('--source-plan',type=Path,required=True);p.add_argument('--out-dir',type=Path,required=True);a=p.parse_args()
    plan=json.loads(a.source_plan.read_text());source=Path(plan['provenance']['onnx_binding']['onnx']['path'])
    existing=a.out_dir/'model.plan.json'
    if existing.exists():
        import hashlib
        bound=json.loads(existing.read_text())['provenance']['onnx_binding']
        for item in [bound['onnx'],*bound['external_data']]:
            input_path=Path(item['path'])
            if not input_path.is_absolute():input_path=Path(bound['onnx']['path']).parent/input_path
            assert hashlib.sha256(input_path.read_bytes()).hexdigest()==item['sha256'], 'Existing FIR candidate binding mismatch; use a new directory'
        saved=json.loads((a.out_dir/'model.export.json').read_text())
        assert saved['custom_fir_quant']['source']['sha256']==hashlib.sha256(source.read_bytes()).hexdigest(), 'Do not overwrite a built FIR candidate from another source'
        print(json.dumps(dict(batch=saved['batch'],status='existing_bound_candidate_preserved')),flush=True);return
    record=json.loads(source.with_suffix('.export.json').read_text());model=onnx.load(source,load_external_data=True)
    nodes=list(model.graph.node);producer={v:n for n in nodes for v in n.output};users={};initial={v.name:v for v in model.graph.initializer}
    for n in nodes:
        for v in n.input:users.setdefault(v,[]).append(n)
    replacements={};rows=[]
    for q in nodes:
        if q.op_type!='QuantizeLinear' or len(q.input)!=3 or q.input[2] not in initial:continue
        zero=numpy_helper.to_array(initial[q.input[2]])
        if zero.dtype!=np.int8 or np.any(zero):continue
        upstream=producer.get(q.input[0]);smooth_name=None
        if upstream is not None and upstream.op_type=='Div' and upstream.input[1] in initial and len(users.get(upstream.output[0],[]))==1:
            smooth_name=upstream.input[1];fir=producer.get(upstream.input[0])
        else:fir=upstream
        if fir is None or fir.op_type!='small_fir_activation_tiled' or len(users.get(fir.output[0],[]))!=1:continue
        if any(v not in initial for v in fir.input[1:]) or q.input[1] not in initial:continue
        channels=int(initial[fir.input[3]].dims[0])
        smooth=numpy_helper.to_array(initial[smooth_name]).reshape(-1) if smooth_name else np.ones(channels,np.float32)
        if smooth.size==1:smooth=np.repeat(smooth,channels)
        if smooth.size!=channels or smooth.dtype!=np.float32 or np.any(smooth<=0):raise ValueError('Unexpected original smoothing geometry')
        name=f'fir_quant_original_smooth_{len(rows)}';model.graph.initializer.append(numpy_helper.from_array(np.ascontiguousarray(smooth),name))
        replacements[q.name]=helper.make_node(NAME.split('::')[1],[*fir.input,name,q.input[1]],list(q.output),name=f'fir_quant_{len(rows)}',domain='inspark_custom',plugin_namespace='inspark_custom')
        rows.append(dict(source_fir=fir.name,source_quantize=q.name,shape_channels=channels,original_scale=q.input[1],original_smooth=smooth_name or 'identity',parameters=[*fir.input[1:],name,q.input[1]],input=fir.input[0],output=q.output[0]))
    if not rows:raise RuntimeError('No exclusive original FIR->INT8 paths')
    new=[replacements.get(n.name,n) for n in nodes];needed={v.name for v in model.graph.output};kept=[]
    for n in reversed(new):
        if any(v in needed for v in n.output):kept.append(n);needed.update(n.input)
    new=list(reversed(kept));del model.graph.node[:];model.graph.node.extend(new)
    used={v for n in new for v in n.input};kept=[v for v in model.graph.initializer if v.name in used];del model.graph.initializer[:];model.graph.initializer.extend(kept)
    a.out_dir.mkdir(parents=True,exist_ok=True);out=a.out_dir/'model.onnx'
    onnx.save_model(model,out,save_as_external_data=True,all_tensors_to_one_file=True,location='model.onnx.data',size_threshold=1024);onnx.checker.check_model(str(out));artifact=capture_onnx_artifact(out)
    record.update(onnx=str(out.resolve()),onnx_sha256=artifact['sha256'],onnx_artifact=artifact)
    record['plugins'].append(NAME);record['graph'].update(nodes=len(new),plugins=record['plugins'])
    record['custom_fir_quant']=dict(paths=rows,source=file_record(source,'validated_current_best_source'),mechanism='Protected FP32 FIR/Snake/FIR remains unchanged; original smooth/scale/zero-point/round-even INT8 rule fused after it, eliminating FP32 store/reload only. Weight Q nodes and source recipe remain unchanged.')
    out.with_suffix('.export.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(batch=record['batch'],fused_paths=len(rows),nodes=len(new),onnx=str(out))),flush=True)


if __name__=='__main__':main()
