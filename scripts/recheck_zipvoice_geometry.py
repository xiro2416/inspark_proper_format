"""Adapted source geometry recheck: unchanged normal attention algorithm and precision."""
import argparse,fcntl,importlib,json,os,statistics
from pathlib import Path
import torch,triton

R=Path(__file__).resolve().parents[1]
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--batch',type=int,choices=(1,2,4,8,16,32),required=True);args=parser.parse_args();B=args.batch
 online_branch_stats=importlib.import_module(f'inspark_infer.ops.tensorrt.zipvoice.a1007.b{B}.normal_tf32_runtime_kernel').online_branch_stats
 assert os.environ['CUDA_VISIBLE_DEVICES']=='1';lock=Path('/workspace/.cache/inspark/gpu-locks/1.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.manual_seed(9102);stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream());torch.cuda.set_stream(stream);rows=[]
 for t in (150,190,230,380,760):
  q=torch.randn(4,B,t,32,device='cuda')/(32**.5);k=torch.randn(4,B,32,t,device='cuda');pq=torch.randn(4,B,t,4,device='cuda');e=torch.randn(4,1,4,2*t-1,device='cuda');mask=torch.zeros(B,t,device='cuda',dtype=torch.bool);mask[:,-3:]=True;v1=torch.randn(4,B,t,12,device='cuda');v2=torch.randn_like(v1);o1=torch.full_like(v1,float('nan'));o2=torch.full_like(v2,float('nan'));stats=torch.full((4,B,t,2),float('nan'),device='cuda')
  if B>1:mask[0,:]=True
  configs=[(64,32,4),(32,32,4),(64,16,4)]+([(16,32,4)] if B<=8 else [])
  def launch(qb,kb,warps):
   grid=(triton.cdiv(t,qb),B,4);kw={'NONLIN':False,'QB':qb,'KB':kb,'AV_MODE':'tf32','num_warps':warps,'num_stages':1,'enable_fp_fusion':False}
   a=online_branch_stats[grid](q,k,pq,e,mask,v1,o1,stats,t,STATS_MODE=1,**kw);b=online_branch_stats[grid](q,k,pq,e,mask,v2,o2,stats,t,STATS_MODE=2,**kw);return a,b
  launch(64,32,4);stream.synchronize();refs=[o1.clone(),o2.clone()];baseline=torch.cuda.CUDAGraph()
  with torch.cuda.graph(baseline,stream=stream):launch(64,32,4)
  for qb,kb,warps in configs:
   row={'T':t,'B':B,'QB':qb,'KB':kb,'warps':warps}
   try:
    o1.fill_(float('nan'));o2.fill_(float('nan'));stats.fill_(float('nan'));ks=launch(qb,kb,warps);stream.synchronize();assert bool(torch.isfinite(o1).all()) and bool(torch.isfinite(o2).all()) and bool(torch.isfinite(stats).all())
    row['correctness']={'all_batch_fourheads_full12_and_rowstats_written':True,'relative_l2_vs_retained_geometry':[float((y-r).norm()/r.norm()) for y,r in zip((o1,o2),refs)]};row['resources']=[{'regs':a.n_regs,'spills':a.n_spills,'shared':a.metadata.shared,'denseTF32MMA':('mma.sync' in a.asm['ptx'] and '.tf32.' in a.asm['ptx'])} for a in ks]
    for _ in range(3):launch(qb,kb,warps)
    stream.synchronize();g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g,stream=stream):launch(qb,kb,warps)
    samples=[[],[]]
    for rep in range(6):
     for i in ((0,1,1,0) if rep%2==0 else (1,0,0,1)):
      a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record()
      for _ in range(3):(baseline if i==0 else g).replay()
      b.record();b.synchronize();samples[i].append(a.elapsed_time(b)/3)
    med=[statistics.median(v) for v in samples];row.update(status='complete',retained_candidate_ms=med,gain_percent=100*(med[0]-med[1])/med[0],samples_ms=samples);del g
   except Exception as exc:row.update(status='failed',error=str(exc))
   rows.append(row);print(json.dumps({k:v for k,v in row.items() if k!='samples_ms'}),flush=True)
  del baseline,q,k,pq,e,mask,v1,v2,o1,o2,stats,refs;torch.cuda.empty_cache()
 (R/f'reports/sm89/zipvoice/a1007/b{B}/history/007-normal-geometry.json').write_text(json.dumps({'status':'target_normal_TF32_geometry_review_complete','rows':rows,'scope':'Matched actual batch/fourhead/full12 normalwrite->read; builder Triton JIT fragment, pointer-alignment specialization may differ from deployed AOT. Requires AOT rebuild and E2E validation; not a retained gain.'},indent=2)+'\n')
if __name__=='__main__':main()
