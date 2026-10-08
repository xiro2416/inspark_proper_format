"""B64 complete fused F32 FFN tile probe after the normal-TF32 coordinator."""
import os
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='1'
os.environ['TRITON_CACHE_DIR']='/workspace/A_1007/.cache/triton-f32-ffn-tiles'
os.environ['CUDA_CACHE_PATH']='/workspace/A_1007/.cache/cuda'
os.environ['TMPDIR']='/workspace/A_1007/.cache/tmp'
os.environ['XDG_CACHE_HOME']='/workspace/A_1007/.cache'
import argparse,fcntl,importlib.util,json,statistics,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from run_zipvoice_validation import sha
from zipvoice_selected_route import idle_gpu_preflight


def live(pid):
    p=subprocess.run(['ps','-p',str(pid),'-o','stat='],capture_output=True,text=True)
    return p.returncode==0 and bool(p.stdout.strip()) and not p.stdout.strip().startswith('Z')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--after-pid',type=int,required=True);p.add_argument('--warps',type=int,choices=(4,8),default=4);p.add_argument('--report',default='029-f32-ffn-tiles.json');args=p.parse_args();assert Path(args.report).name==args.report
    print(f'Waiting for specific coordinator PID {args.after_pid}',flush=True)
    while live(args.after_pid):time.sleep(10)
    lease=Path('/workspace/.cache/inspark/gpu-locks/1.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX)
    preflight=idle_gpu_preflight()
    import torch,triton
    path=ROOT/'.work/zipvoice/b64/code/linear_f32_activation_tf32_rna_kernel.py'
    spec=importlib.util.spec_from_file_location('f32_ffn_tile_probe',path);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod);fn=mod.linear_f32_activation
    configs=[(128,64,32,2),(128,64,64,2),(128,32,64,2),(64,128,64,2),(128,64,32,3)]
    records=[];torch.manual_seed(9102)
    for t in (300,380,460,600,760,920):
        for n in (1152,1536,1920):
            m=64*t;x=torch.randn(m,512,device='cuda')*.1;w=torch.randn(512,n,device='cuda')*.02;bias=torch.randn(n,device='cuda')*.01
            y=torch.full((m,n),float('nan'),device='cuda')
            def launch(config,warps=4):
                bm,bn,bk,stages=config
                return fn[(triton.cdiv(m,bm)*triton.cdiv(n,bn),)](x,w,bias,y,m,512,n,bm,bn,bk,4.,0.07999999821186066,0.03500000014901161,'tf32',num_warps=warps,num_stages=stages,enable_fp_fusion=False)
            launch(configs[0]);torch.cuda.synchronize();reference=y.clone()
            assert torch.isfinite(reference).all().item()
            for _ in range(3):launch(configs[0])
            base=torch.cuda.CUDAGraph()
            with torch.cuda.graph(base):launch(configs[0])
            for config in configs:
                row={'batch':64,'frames_at_stage':t,'M':m,'N':n,'configuration':{'BM':config[0],'BN':config[1],'BK':config[2],'stages':config[3],'warps':args.warps},'baseline_configuration':[128,64,32,2]}
                try:
                    y.fill_(float('nan'));compiled=launch(config,args.warps);torch.cuda.synchronize()
                    assert torch.isfinite(y).all().item()
                    row['resources']={'registers':compiled.n_regs,'spills':compiled.n_spills,'shared_bytes':compiled.metadata.shared,'global_scratch_bytes':compiled.metadata.global_scratch_size}
                    assert 'cvt.rna.tf32.f32' in compiled.asm['ptx'] and '.tf32.' in compiled.asm['ptx'] and 'mma.sync' in compiled.asm['ptx']
                    row['errors']={'relative_l2':((y-reference).norm()/reference.norm().clamp_min(1e-20)).item(),'max_abs':(y-reference).abs().max().item(),'all_elements_finite':True}
                    if compiled.metadata.shared>49152 or compiled.metadata.global_scratch_size or compiled.metadata.profile_scratch_size:
                        row['status']='outside_current_plugin_resource_contract'
                    else:
                        for _ in range(3):launch(config,args.warps)
                        candidate=torch.cuda.CUDAGraph()
                        with torch.cuda.graph(candidate):launch(config,args.warps)
                        samples={'control':[],'candidate':[]}
                        for name in ('control','candidate','candidate','control','candidate','control','control','candidate'):
                            graph=base if name=='control' else candidate
                            for _ in range(5):
                                start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record();graph.replay();end.record();end.synchronize();samples[name].append(start.elapsed_time(end))
                        med={k:statistics.median(v) for k,v in samples.items()}
                        row.update(status='complete',median_ms=med,samples_ms=samples,gain_percent=100*(med['control']-med['candidate'])/med['control']);del candidate
                except Exception as exc:row.update(status='failed',error=repr(exc))
                records.append(row);print(json.dumps({k:v for k,v in row.items() if k!='samples_ms'}),flush=True)
            del base,x,w,bias,y,reference;torch.cuda.empty_cache()
    report={'status':'synthetic_full_shape_f32_ffn_tile_microprobe_complete_not_accepted','preflight':preflight,'source_sha256':sha(path),'records':records,'limits':['Synthetic complete fused FFN projection/bias/stableactivation with source TF32RNA mode, JIT graph events only','All3actual widths and6full/half-stage boundary lengths tested. No new INT8 weight/scale recipe or epilogue. BK grouping may change reduction order.','Independent AOT resource verification, full mapping/quality and currentbest E2E required to retain any shape-dependent configuration']}
    (ROOT/'reports/sm89/zipvoice/a1007/b64/history'/args.report).write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
