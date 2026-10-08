"""B64 authorized TF32 normal attention geometry probe versus original IEEE."""
import os
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='1'
os.environ.setdefault('CUDA_CACHE_PATH','/workspace/A_1007/.cache/cuda')
os.environ.setdefault('TMPDIR','/workspace/A_1007/.cache/tmp')
os.environ.setdefault('XDG_CACHE_HOME','/workspace/A_1007/.cache')
os.environ.setdefault('TRITON_CACHE_DIR','/workspace/A_1007/.cache/triton-normal-tf32-geometry')
import fcntl,importlib.util,json,statistics,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from zipvoice_selected_route import idle_gpu_preflight
from run_zipvoice_validation import sha


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    obj=importlib.util.module_from_spec(spec);sys.modules[name]=obj;spec.loader.exec_module(obj)
    return obj.online_branch_stats


def main():
    lock=Path('/workspace/.cache/inspark/gpu-locks/1.lock').open('a')
    print('Waiting for serial GPU1 lease',flush=True)
    fcntl.flock(lock,fcntl.LOCK_EX)
    preflight=idle_gpu_preflight()
    import torch,triton
    torch.manual_seed(9102)
    configs={'control':(64,32,1),'candidate':(64,32,1),'q32k64':(32,64,1),'q64k64':(64,64,1),'q64k16':(64,16,1),'q32k32':(32,32,1),'q32k64s2':(32,64,2)}
    paths={k:ROOT/f'.work/zipvoice/{"b64" if k=="control" else "b64_normtf32"}/code/online_branch_stats_runtime_kernel.py' for k in configs}
    kernels={k:module('normal_probe_'+k,p) for k,p in paths.items()}
    records=[]
    for t in (150,190,230,300,380,460,600,760,920):
        q=torch.randn(4,64,t,32,device='cuda')*.2
        k=torch.randn(4,64,32,t,device='cuda')*.2
        pq=torch.randn(4,64,t,4,device='cuda')*.1
        e=torch.randn(4,4,2*t-1,device='cuda')*.1
        mask=torch.zeros(64,t,device='cuda',dtype=torch.bool)
        mask[1,-17:]=True;mask[2,:]=True
        v1=torch.randn(4,64,t,12,device='cuda');v2=torch.randn_like(v1)
        data={};compiled={};graphs={}
        for name,fn in kernels.items():
            stats=torch.full((4,64,t,2),float('nan'),device='cuda')
            o1=torch.full_like(v1,float('nan'));o2=torch.full_like(v2,float('nan'))
            def run(fn=fn,stats=stats,o1=o1,o2=o2,name=name):
                compiled[name]=fn[(triton.cdiv(t,configs[name][0]),64,4)](q,k,pq,e,mask,v1,o1,stats,t,False,configs[name][0],configs[name][1],'ieee' if name=='control' else 'tf32',1,num_warps=4,num_stages=configs[name][2],enable_fp_fusion=False)
                fn[(triton.cdiv(t,configs[name][0]),64,4)](q,k,pq,e,mask,v2,o2,stats,t,False,configs[name][0],configs[name][1],'ieee' if name=='control' else 'tf32',2,num_warps=4,num_stages=configs[name][2],enable_fp_fusion=False)
            for _ in range(3):run()
            torch.cuda.synchronize()
            assert all(torch.isfinite(x).all().item() for x in (stats,o1,o2))
            data[name]=(stats,o1,o2)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):run()
            graphs[name]=graph
        ptx=compiled['candidate'].asm['ptx']
        assert 'cvt.rna.tf32.f32' in ptx and 'mma.sync' in ptx and '.tf32.' in ptx
        assert 'tf32' not in compiled['control'].asm['ptx'].lower()
        for name in configs:
            if name=='control':continue
            ptx=compiled[name].asm['ptx']
            assert 'cvt.rna.tf32.f32' in ptx and 'mma.sync' in ptx and '.tf32.' in ptx
            errors={}
            for i,key in enumerate(('stats','branch1','branch2')):
                a,b=data['control'][i],data[name][i]
                errors[key]={'relative_l2':((a-b).norm()/a.norm().clamp_min(1e-20)).item(),'max_abs':(a-b).abs().max().item()}
            resources={kind:{'registers':compiled[kind].n_regs,'shared_bytes':compiled[kind].metadata.shared,'global_scratch_bytes':compiled[kind].metadata.global_scratch_size} for kind in ('control',name)}
            samples={'control':[],'candidate':[]}
            for kind in ('control','candidate','candidate','control','candidate','control','control','candidate'):
                graph=graphs['control' if kind=='control' else name]
                for _ in range(5):
                    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                    start.record();graph.replay();end.record();end.synchronize();samples[kind].append(start.elapsed_time(end))
            med={kind:statistics.median(values) for kind,values in samples.items()}
            row={'frames':t,'batch':64,'heads':4,'full_value_width':12,'candidate':name,'geometry':{'query':configs[name][0],'key':configs[name][1],'warps':4,'stages':configs[name][2]},'samples_ms':samples,'median_ms':med,'gain_percent':100*(med['control']-med['candidate'])/med['control'],'errors':errors,'resources':resources,'fits_existing_shared_scratch_contract':compiled[name].metadata.shared<=49152 and not compiled[name].metadata.global_scratch_size and not compiled[name].metadata.profile_scratch_size,'all_outputs_and_stats_finite':True,'all_masked_row_included':True}
            records.append(row);print(json.dumps({k:row[k] for k in ('frames','candidate','geometry','median_ms','gain_percent')}),flush=True)
        del data,graphs,q,k,pq,e,mask,v1,v2
        torch.cuda.empty_cache()
    report={'status':'synthetic_normal_tf32_geometry_microprobe_not_deployment_acceptance','preflight':preflight,'source_sha256':{k:sha(p) for k,p in paths.items()},'records':records,'limits':['Synthetic inputs and JIT graph events, not TensorRT AOT or application E2E','No numerical error cutoff; complete path mapping and CER/UTMOS/SIM-o required before retention','Control IEEE; candidate authorized ordinary TF32 with RNA; mask, positional FMA, stats dependencies unchanged; geometry and pipeline stages vary']}
    (ROOT/'reports/sm89/zipvoice/a1007/b64/history/031-normal-tf32-geometry.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
