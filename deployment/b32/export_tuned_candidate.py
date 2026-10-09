"""Apply the measured schedule only to six C192/F1664/K11 source convolutions."""
import argparse
import json
from pathlib import Path
import onnx

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch",type=int,choices=[1,2,4,8,16,32,64,128],default=32)
    parser.add_argument("--history-dir",type=Path,default=ROOT/"deployment/b32/history")
    args=parser.parse_args();batch=args.batch;args.history_dir.mkdir(parents=True,exist_ok=True)
    from trt113_provenance import capture_onnx_artifact,file_record
    source=ROOT/f'artifacts/sm89/int8_smoothquant/b{batch}/vocoder-implicit-int8/model.onnx'
    record=json.loads(source.with_suffix('.export.json').read_text());model=onnx.load(source,load_external_data=True)
    rows=record['custom_implicit_int8_conv']['rewrites'];ids=[i for i,r in enumerate(rows) if r['kernel']==11 and r['output_shape']==[batch,192,1664]]
    if ids!=[16,17,18,19,20,21]:raise RuntimeError('Measured region changed')
    names={f'b32_implicit_conv_{i}' for i in ids};count=0
    for node in model.graph.node:
        if node.name in names:
            if node.op_type!='implicit_int8_conv_1d':raise RuntimeError('Unexpected source operator')
            node.op_type='implicit_int8_conv_1d_tuned' if batch==32 else 'implicit_int8_conv_1d_migrated';count+=1
    if count!=6:raise RuntimeError('Incomplete schedule adaptation')
    output=ROOT/f'artifacts/sm89/int8_smoothquant/b{batch}/vocoder-implicit-tuned/model.onnx';output.parent.mkdir(parents=True,exist_ok=True)
    onnx.save_model(model,output,save_as_external_data=True,all_tensors_to_one_file=True,location='model.onnx.data',size_threshold=1024)
    onnx.checker.check_model(str(output));artifact=capture_onnx_artifact(output)
    record.update(onnx=str(output),onnx_sha256=artifact['sha256'],onnx_artifact=artifact)
    record['plugins'].append('inspark_custom::implicit_int8_conv_1d_tuned' if batch==32 else 'inspark_custom::implicit_int8_conv_1d_migrated')
    record['graph']['plugins']=record['plugins']
    record['custom_implicit_int8_conv'].update(tuned_regions=[rows[i] for i in ids],tuned_tile=[64,64,64],tuned_source=file_record(source,'validated_implicit_convolution_source'))
    output.with_suffix('.export.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(tuned_convolutions=count,onnx=str(output))),flush=True)


if __name__=='__main__':main()
