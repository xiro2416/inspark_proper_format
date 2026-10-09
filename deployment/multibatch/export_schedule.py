"""Apply only locally measured tile changes, preserving original quantized math."""
import argparse,json
from pathlib import Path
import onnx
from onnx import helper
ROOT=Path(__file__).resolve().parents[2]


def main():
    from trt113_provenance import capture_onnx_artifact,file_record
    from deployment.multibatch.schedule_plugin import NAME
    p=argparse.ArgumentParser();p.add_argument('--batch',type=int,required=True);p.add_argument('--probe',type=Path,required=True);p.add_argument('--residual-probe',type=Path);p.add_argument('--out-dir',type=Path);a=p.parse_args()
    probe=json.loads(a.probe.read_text());assert probe['shape']==[a.batch,192,1664]
    tiles={d:min((r for r in probe['rows'] if r['dilation']==d),key=lambda r:r['p50_ms'])['tile'] for d in (1,3,5)}
    assert all(r['bit_identical'] for r in probe['rows'])
    source=ROOT/f'artifacts/sm89/int8_smoothquant/b{a.batch}/vocoder-implicit-tuned/model.onnx'
    record=json.loads(source.with_suffix('.export.json').read_text());model=onnx.load(source,load_external_data=True)
    residual=json.loads(a.residual_probe.read_text()) if a.residual_probe else {'groups':[]}
    changed={}
    for g in residual['groups']:
        assert g['shape'][0]==a.batch and all(r['bit_identical'] for r in g['rows'])
        best=min(g['rows'],key=lambda r:r['p50_ms']);old=next(r for r in g['rows'] if r['tile']==[32,32,64])
        if best['p50_ms']<old['p50_ms']*.98 and best['tile']!=[32,32,64]:
            for i in g['indices']:changed[f'b32_implicit_conv_{i}']=best['tile']
    count=0
    for n in model.graph.node:
        if n.op_type=='implicit_int8_conv_1d_migrated' or n.name in changed:
            d=next(at.i for at in n.attribute if at.name=='dilation');tile=changed[n.name] if n.name in changed else tiles[d];n.op_type=NAME.split('::')[1]
            for name,value in zip(('bm','bn','bk'),tile):n.attribute.append(helper.make_attribute(name,value))
            count+=1
    assert count==6+len(changed)
    out=(a.out_dir or ROOT/f'artifacts/sm89/int8_smoothquant/b{a.batch}/vocoder-target-schedule')/'model.onnx';out.parent.mkdir(parents=True,exist_ok=True)
    if out.with_suffix('.engine').exists():
        import hashlib
        retained=json.loads(out.with_suffix('.plan.json').read_text())
        old=json.loads(out.with_suffix('.export.json').read_text())['custom_implicit_int8_conv']['target_schedule']
        assert hashlib.sha256(out.read_bytes()).hexdigest()==retained['provenance']['onnx_binding']['onnx']['sha256'], 'Existing ONNX/engine binding mismatch; use a new directory'
        assert old['tiles']=={str(k):v for k,v in tiles.items()} and old['residual_regions']==changed, 'Do not overwrite a built candidate; use a new directory'
        print(json.dumps(dict(batch=a.batch,status='existing_bound_candidate_preserved',onnx=str(out))),flush=True);return
    onnx.save_model(model,out,save_as_external_data=True,all_tensors_to_one_file=True,location='model.onnx.data',size_threshold=1024)
    onnx.checker.check_model(str(out));artifact=capture_onnx_artifact(out)
    record.update(onnx=str(out),onnx_sha256=artifact['sha256'],onnx_artifact=artifact)
    record['plugins']=[NAME if n=='inspark_custom::implicit_int8_conv_1d_migrated' else n for n in record['plugins']]
    record['graph']['plugins']=record['plugins']
    record['custom_implicit_int8_conv']['target_schedule']=dict(tiles=tiles,residual_regions=changed,probe=file_record(a.probe,'target_bit_identical_schedule_probe'),source=file_record(source,'validated_migration_source'))
    out.with_suffix('.export.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(batch=a.batch,regions=count,tiles=tiles,onnx=str(out))),flush=True)


if __name__=='__main__':main()
