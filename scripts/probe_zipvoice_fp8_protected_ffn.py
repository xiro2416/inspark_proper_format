"""Focused source floating FFN fusion probe; operator timing is not E2E acceptance."""
import argparse,json,sys,importlib.util,warnings
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from inspark_infer.runtime.zipvoice_fp8.common import write,sha

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batches',type=int,nargs='+',default=[1,16,64]);p.add_argument('--frames',type=int,default=760);p.add_argument('--families',type=int,nargs='+',choices=[1,2,3]);args=p.parse_args()
 from inspark_infer.runtime.device import GPULease
 with GPULease(3):run(args)
def run(args):
 import numpy as np,torch,tensorrt as trt
 warnings.filterwarnings('error',message='The CUDA Graph is empty.*')
 from safetensors import safe_open
 from inspark_infer.ops.tensorrt.zipvoice.engine import Engine
 out=ROOT/('outputs/fp8/protected-ffn-probe'+('' if args.frames==760 else f'-t{args.frames}'));out.mkdir(parents=True,exist_ok=True)
 source=ROOT/'src/inspark_infer/build/zipvoice_templates/b16/linear_f32_activation_tf32_rna_kernel.py.template'
 local=out/'kernel.py';local.write_text(source.read_text());spec=importlib.util.spec_from_file_location('protected_ffn_probe_kernel',local);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
 stream=torch.cuda.current_stream();records=[]
 def timed(fn):
  for _ in range(5):fn()
  torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph):fn()
  events=[]
  for _ in range(30):
   a=torch.cuda.Event(enable_timing=True);z=torch.cuda.Event(enable_timing=True);a.record();graph.replay();z.record();events.append((a,z))
  torch.cuda.synchronize();return float(np.median([a.elapsed_time(z) for a,z in events]))
 for b in args.batches:
  for ff in (args.families or ([1,2,3] if b==16 else [2])):
   name=f'fm_decoder.encoders.0.layers.0.feed_forward{ff}.in_proj'
   with safe_open(ROOT/'models/zipvoice/eager/model.safetensors',framework='pt') as f:
    w=f.get_tensor(name+'.weight').T.contiguous().numpy();bias=f.get_tensor(name+'.bias').numpy()
   m=b*args.frames;n=w.shape[1];logger=trt.Logger(trt.Logger.WARNING);builder=trt.Builder(logger);net=builder.create_network(0);cfg=builder.create_builder_config();cfg.builder_optimization_level=5;cfg.max_aux_streams=0;cfg.tiling_optimization_level=trt.TilingOptimizationLevel.FULL;cfg.max_num_tactics=2**31-2
   x=net.add_input('x',trt.float32,(m,512));wc=net.add_constant(w.shape,w).get_output(0);bc=net.add_constant((1,n),bias.reshape(1,n)).get_output(0)
   value=net.add_matrix_multiply(x,trt.MatrixOperation.NONE,wc,trt.MatrixOperation.NONE).get_output(0)
   value=net.add_elementwise(value,bc,trt.ElementWiseOperation.SUM).get_output(0)
   def c(v):return net.add_constant((1,1),np.array([[v]],dtype=np.float32)).get_output(0)
   def ew(a,z,op):return net.add_elementwise(a,z,op).get_output(0)
   u=ew(value,c(4),trt.ElementWiseOperation.SUB)
   ab=net.add_unary(u,trt.UnaryOperation.ABS).get_output(0);neg=net.add_unary(ab,trt.UnaryOperation.NEG).get_output(0);exp=net.add_unary(neg,trt.UnaryOperation.EXP).get_output(0)
   log=net.add_unary(ew(exp,c(1),trt.ElementWiseOperation.SUM),trt.UnaryOperation.LOG).get_output(0)
   act=ew(ew(ew(u,c(0),trt.ElementWiseOperation.MAX),log,trt.ElementWiseOperation.SUM),ew(value,c(.08),trt.ElementWiseOperation.PROD),trt.ElementWiseOperation.SUB)
   act=ew(act,c(.035),trt.ElementWiseOperation.SUB);act.name='y';net.mark_output(act)
   plan=builder.build_serialized_network(net,cfg)
   if plan is None:raise RuntimeError('Protected FFN native build failed')
   path=out/f'b{b}-ff{ff}.plan';path.write_bytes(plan);eng=Engine(path,trt,torch)
   torch.manual_seed(772);xx=torch.randn(m,512,device='cuda');ww=torch.from_numpy(w).cuda();bb=torch.from_numpy(bias).cuda();yy=torch.empty(m,n,device='cuda')
   native_ms=timed(lambda:eng({'x':xx},torch.cuda.current_stream()));candidates=[]
   expected=eng({'x':xx},stream)['y'].clone()
   for bm,bn,bk in [(128,64,32),(64,64,32),(128,64,64)]:
    def launch():return mod.linear_f32_activation[( ((m+bm-1)//bm)*((n+bn-1)//bn),)](xx,ww,bb,yy,m,512,n,bm,bn,bk,4.,float(np.float32(.08)),float(np.float32(.035)),'tf32',num_warps=4)
    kernel=launch();ms=timed(launch)
    torch.cuda.synchronize();error=yy-expected
    candidates.append(dict(tile=[bm,bn,bk],median_ms=ms,registers=kernel.n_regs,spills=kernel.n_spills,relative_l2=float(error.norm()/expected.norm()),max_abs=float(error.abs().max())))
   records.append(dict(batch=b,frames=args.frames,ff=ff,n=n,native_ms=native_ms,candidates=candidates))
   write(out/'report.json',dict(status='operator_probe_complete',source_sha256=sha(source),weights_sha256=sha(ROOT/'models/zipvoice/eager/model.safetensors'),rows=records,limits='Isolated original protected floating in-projection plus SwooshL. CUDA Graph replay/event timing, not application E2E; no fixed numeric gate. Any useful candidate requires full graph and quality validation.'))
   print(json.dumps(records[-1]),flush=True)
if __name__=='__main__':main()
