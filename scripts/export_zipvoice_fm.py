"""Re-export original saved INT8 tensors; does not calibrate or change scales.

The inherited exporter uses explicit legacy Q/DQ symbolics because the source
model's dynamic position/downsample path failed Dynamo shape constraints.
Re-exported graphs require mapping and execution validation before deployment.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights',type=Path,default=ROOT/'models/zipvoice')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('Keep existing source graph; export to a new destination')
    import torch
    import onnx
    from inspark_infer.models.zipvoice.weights import load_model
    from inspark_infer.models.zipvoice.packed import export_positions
    torch.set_num_threads(4)
    model,_=load_model(args.weights,'int8')
    model=export_positions(model).cpu().eval()
    class FM(torch.nn.Module):
        def __init__(self,model):super().__init__();self.model=model
        def forward(self,t,x,text_condition,speech_condition,padding_mask,guidance_scale):
            return self.model.forward_fm_decoder(t,x,text_condition,speech_condition,padding_mask,guidance_scale)
    def quant(g,x,scale,zero,*rest):return g.op('QuantizeLinear',x,scale,g.op('Cast',zero,to_i=3))
    def dequant(g,x,scale,zero,*rest):return g.op('DequantizeLinear',x,scale,g.op('Cast',zero,to_i=3))
    def dequant_channel(g,x,scale,zero,axis,*rest):
        from torch.onnx import symbolic_helper
        return g.op('DequantizeLinear',x,scale,g.op('Cast',zero,to_i=3),axis_i=symbolic_helper._parse_arg(axis,'i'))
    for name,fn in [('quantize_per_tensor',quant),('dequantize_per_tensor',dequant),('dequantize_per_channel',dequant_channel)]:
        torch.onnx.register_custom_op_symbolic('quantized_decomposed::'+name,fn,21)
    b,t=2,169
    inputs=(torch.full((b,1,1),.25),torch.randn(b,t,100),torch.randn(b,t,100),torch.randn(b,t,100),torch.zeros(b,t,dtype=torch.bool),torch.ones(b,1,1))
    names=['t','x','text_condition','speech_condition','padding_mask','guidance_scale']
    dynamic={name:{0:'batch'} for name in [*names,'velocity']}
    for name in ('x','text_condition','speech_condition','padding_mask','velocity'):dynamic[name][1]='frames'
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(FM(model).eval(),inputs,str(args.output),dynamo=False,opset_version=21,input_names=names,output_names=['velocity'],dynamic_axes=dynamic,external_data=True)
    graph=onnx.load(args.output)
    external=args.output.with_name(args.output.name+'.data')
    if external.exists():external.unlink()
    onnx.save_model(graph,str(args.output),save_as_external_data=True,all_tensors_to_one_file=True,location=external.name,size_threshold=1024)
    onnx.checker.check_model(str(args.output))
    count=sum(t.data_type==onnx.TensorProto.INT8 and t.name.endswith('.weight_int8') for t in graph.graph.initializer)
    assert count==180
    report={'status':'exported_unvalidated','int8_weights':count,'exporter':'legacy explicit QDQ compatibility route','torch':torch.__version__,
            'graph_sha256':hashlib.sha256(args.output.read_bytes()).hexdigest(),'quantization_changed':False}
    args.output.with_suffix('.export.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))

if __name__=='__main__':main()
