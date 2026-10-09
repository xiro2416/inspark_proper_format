"""Isolated FP8 GEMM/residual/dual-output fusion probe; not E2E acceptance."""
import json,os,sys,warnings
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from inspark_infer.runtime.zipvoice_fp8_b128.common import write,sha


def main():
 from inspark_infer.runtime.device import GPULease
 with GPULease(3):run()

def run():
 import numpy as np,onnx,torch,tensorrt as trt,triton,triton.language as tl
 from onnx import helper as h,numpy_helper as nh,TensorProto as tp
 from safetensors import safe_open
 warnings.filterwarnings('error',message='The CUDA Graph is empty.*')
 # Separate importable kernel file lets Triton inspect its exact source.
 folder=ROOT/'outputs/fp8/b128/residual-probe';folder.mkdir(parents=True,exist_ok=True)
 kernel_path=folder/'kernel.py';kernel_path.write_text(KERNEL)
 import importlib.util
 spec=importlib.util.spec_from_file_location('b128_residual_probe_kernel',kernel_path);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
 name='fm_decoder.encoders.4.layers.0.feed_forward1.out_proj';next_name='fm_decoder.encoders.4.layers.0.nonlin_attention.in_proj'
 with safe_open(ROOT/'models/zipvoice/fp8/model.safetensors',framework='pt') as f:
  weight=f.get_tensor(name+'.weight_fp8');bias=f.get_tensor(name+'.bias');sa=float(f.get_tensor(name+'.input_scale'));sw=float(f.get_tensor(name+'.weight_scale'));so=float(f.get_tensor(next_name+'.input_scale'))
 n,k=weight.shape;w=weight.cuda();b=bias.cuda();records=[]
 def timed(fn):
  for _ in range(4):fn()
  torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
  with torch.cuda.graph(g):fn()
  pairs=[]
  for _ in range(30):
   a=torch.cuda.Event(enable_timing=True);z=torch.cuda.Event(enable_timing=True);a.record();g.replay();z.record();pairs.append((a,z))
  torch.cuda.synchronize();return float(np.median([a.elapsed_time(z) for a,z in pairs]))
 for frames in (190,380,760):
  m=128*frames;torch.manual_seed(9173);q=(torch.randn(m,k,device='cuda')*.3/sa).clamp(-448,448).to(torch.float8_e4m3fn);res=torch.randn(m,n,device='cuda')*.3
  inits=[h.make_tensor('w',tp.FLOAT8E4M3FN,[k,n],weight.T.contiguous().view(torch.uint8).numpy().tobytes(),raw=True),nh.from_array(bias.numpy(),'bias')]
  for key,value in [('sa',sa),('sw',sw),('so',so)]:inits.append(nh.from_array(np.array(value,dtype=np.float32),key))
  inits.append(h.make_tensor('zero',tp.FLOAT8E4M3FN,[],bytes([0]),raw=True))
  nodes=[h.make_node('DequantizeLinear',['q','sa','zero'],['a']),h.make_node('DequantizeLinear',['w','sw','zero'],['weight']),h.make_node('MatMul',['a','weight'],['dot']),h.make_node('Add',['dot','bias'],['y']),h.make_node('Add',['y','res'],['out']),h.make_node('QuantizeLinear',['out','so','zero'],['outq'])]
  graph=h.make_graph(nodes,'native_dual_residual',[h.make_tensor_value_info('q',tp.FLOAT8E4M3FN,[m,k]),h.make_tensor_value_info('res',tp.FLOAT,[m,n])],[h.make_tensor_value_info('out',tp.FLOAT,[m,n]),h.make_tensor_value_info('outq',tp.FLOAT8E4M3FN,[m,n])],inits)
  model=h.make_model(graph,opset_imports=[h.make_opsetid('',21)]);model.ir_version=10;onnx.checker.check_model(model);path=folder/f'native-t{frames}.onnx';onnx.save(model,path)
  logger=trt.Logger(trt.Logger.WARNING);builder=trt.Builder(logger);net=builder.create_network(0);parser=trt.OnnxParser(net,logger)
  if not parser.parse_from_file(str(path)):raise RuntimeError('\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
  cfg=builder.create_builder_config();cfg.builder_optimization_level=5;cfg.tiling_optimization_level=trt.TilingOptimizationLevel.FULL;cfg.max_num_tactics=2**31-2;cfg.max_aux_streams=0;cfg.profiling_verbosity=trt.ProfilingVerbosity.DETAILED
  plan=builder.build_serialized_network(net,cfg)
  if plan is None:raise RuntimeError('Native residual operator build failed')
  runtime=trt.Runtime(logger);engine=runtime.deserialize_cuda_engine(plan);ctx=engine.create_execution_context();native_out=torch.empty(m,n,device='cuda');native_q=torch.empty(m,n,device='cuda',dtype=torch.float8_e4m3fn)
  for key,value in [('q',q),('res',res),('out',native_out),('outq',native_q)]:assert ctx.set_tensor_address(key,value.data_ptr())
  def native():assert ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
  native_ms=timed(native);native();torch.cuda.synchronize();expected=native_out.clone();eq=native_q.clone();out=torch.empty_like(native_out);outq=torch.empty_like(native_q);rows=[]
  for bm,bn,bk,warps in [(64,64,64,4),(128,64,64,4),(128,128,64,8),(64,128,64,4)]:
   def launch():return mod.fused[((m+bm-1)//bm*((n+bn-1)//bn),)](q,w,b,res,out,outq,m,n,k,float(np.float32(sa*sw)),float(np.float32(1/so)),bm,bn,bk,num_warps=warps,num_stages=3,enable_fp_fusion=False)
   compiled=launch();candidate_ms=timed(launch);torch.cuda.synchronize();delta=out-expected
   assert torch.isfinite(out).all() and 'e4m3' in compiled.asm['ptx']
   rows.append(dict(tile=[bm,bn,bk],warps=warps,median_ms=candidate_ms,registers=compiled.n_regs,spills=compiled.n_spills,relative_l2=float(delta.norm()/expected.norm()),max_abs=float(delta.abs().max()),fp8_output_equal_fraction=float((outq.view(torch.uint8)==eq.view(torch.uint8)).float().mean())))
  records.append(dict(batch=128,frames=frames,m=m,k=k,n=n,native_ms=native_ms,candidates=rows))
  write(folder/'report.json',dict(status='operator_probe_complete',projection=name,next_projection=next_name,recipe_sha256=sha(ROOT/'models/zipvoice/fp8/quantization.json'),kernel_sha256=sha(kernel_path),rows=records,limits='Isolated constant original FP8 projection plus FLOAT32 bias/residual and dual FP8/FLOAT32 outputs; captured CUDA event timing. Any beneficial candidate needs full graph, operation/quality and matched E2E validation. No fixed numeric gate.'))
  print(json.dumps(records[-1]),flush=True)

KERNEL='''import triton
import triton.language as tl
@triton.jit
def fused(A,W,Bias,R,Y,Q,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,Alpha:tl.constexpr,Inv:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
 pid=tl.program_id(0);nr=tl.cdiv(M,BM);nc=tl.cdiv(N,BN);group=pid//(8*nc);first=group*8;size=tl.minimum(nr-first,8);pm=first+(pid%(8*nc))%size;pn=(pid%(8*nc))//size
 rows=pm*BM+tl.arange(0,BM);cols=pn*BN+tl.arange(0,BN);ks=tl.arange(0,BK);acc=tl.zeros((BM,BN),tl.float32)
 for off in range(tl.cdiv(K,BK)):
  kk=off*BK+ks;x=tl.load(A+rows[:,None]*K+kk[None,:],(rows[:,None]<M)&(kk[None,:]<K),0.0);w=tl.load(W+cols[None,:]*K+kk[:,None],(cols[None,:]<N)&(kk[:,None]<K),0.0);acc=tl.dot(x,w,acc,max_num_imprecise_acc=0)
 z=acc*Alpha+tl.load(Bias+cols,cols<N,0)[None,:];z=z+tl.load(R+rows[:,None]*N+cols[None,:],(rows[:,None]<M)&(cols[None,:]<N),0)
 q=tl.maximum(tl.minimum(z*Inv,448.),-448.).to(tl.float8e4nv,fp_downcast_rounding="rtne")
 tl.store(Y+rows[:,None]*N+cols[None,:],z,(rows[:,None]<M)&(cols[None,:]<N));tl.store(Q+rows[:,None]*N+cols[None,:],q,(rows[:,None]<M)&(cols[None,:]<N))
'''
if __name__=='__main__':main()
