"""Complete source FIR+round-even quantization vs fused implementation Graph."""
import argparse,json
from pathlib import Path
import onnx
from onnx import numpy_helper
import torch,triton
from deployment.b32.probe_fir import graph_measure
from deployment.b32.implicit_int8_probe import quantize_nc
from deployment.multibatch.fir_quant_kernel import fir_quant
from inspark_infer.ops.triton.vocoder_tiled_fir import TiledFIR


def main():
    from inspark_infer.runtime.device import GPULease,select_gpu
    p=argparse.ArgumentParser();p.add_argument('--onnx',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    record=json.loads(a.onnx.with_suffix('.export.json').read_text());model=onnx.load(a.onnx,load_external_data=True)
    arrays={v.name:numpy_helper.to_array(v) for v in model.graph.initializer};infos={v.name:[d.dim_value for d in v.type.tensor_type.shape.dim] for v in model.graph.value_info}
    selected={}
    for r in record['custom_fir_quant']['paths']:
        shape=infos[r['input']]
        if len(shape)!=3 or min(shape)<=0:raise RuntimeError('Static FIR input missing')
        selected.setdefault(tuple(shape),r)
    rows=[]
    with GPULease(1),torch.inference_mode():
        select_gpu(1);torch.manual_seed(1987);stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        for shape,r in sorted(selected.items()):
            b,c,f=shape;parameters=[torch.tensor(arrays[name].copy(),device='cuda').float().contiguous() for name in r['parameters']]
            up,down,alpha,beta,smooth,scale=parameters
            current=TiledFIR(up,down,alpha,beta);x=torch.randn(b,c,f,device='cuda')*.2;reference=torch.empty_like(x,dtype=torch.int8);out=torch.empty_like(reference)
            def control():
                y=current(x)
                quantize_nc[(triton.cdiv(x.numel(),256),)](y,scale,smooth,reference,x.numel(),c,f,256,num_warps=4,enable_fp_fusion=False)
                return reference
            def fused():
                fir_quant[(b*c,triton.cdiv(f,248))](x,*parameters,out,c,f,BLOCK=256,HIGH=512,num_warps=4,enable_fp_fusion=False)
                return out
            old,expected=graph_measure(control,stream);new,actual=graph_measure(fused,stream)
            if not torch.equal(actual,expected):raise RuntimeError('Fused FIR changed precise source FP32/round-even result')
            nonzero=int(torch.count_nonzero(expected));saturated=int(((expected==-128)|(expected==127)).sum())
            if not nonzero:raise RuntimeError('Synthetic probe did not exercise nonzero INT8 outputs; choose more informative inputs')
            rows.append(dict(shape=list(shape),source_path=r['source_fir'],current_ms=old['p50_ms'],fused_ms=new['p50_ms'],gain_pct=100*(old['p50_ms']-new['p50_ms'])/old['p50_ms'],bit_identical=True,nonzero_int8_outputs=nonzero,saturated_int8_outputs=saturated))
            report=dict(batch=record['batch'],scope='Original AOT/Triton FIR math + precise source INT8 rule, complete paired Graph, real trained FIR/activation/smooth/scale parameters, synthetic inputs; not TensorRT/application E2E or native-div equivalence certification',eligible_paths=len(record['custom_fir_quant']['paths']),rows=rows)
            a.out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(rows[-1]),flush=True)


if __name__=='__main__':main()
