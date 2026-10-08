"""Queued B64 normal attention write/read microprobe; no deployment acceptance."""
import os
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='1'
os.environ.setdefault('CUDA_CACHE_PATH','/workspace/A_1007/.cache/cuda')
os.environ.setdefault('TMPDIR','/workspace/A_1007/.cache/tmp')
os.environ.setdefault('XDG_CACHE_HOME','/workspace/A_1007/.cache')
os.environ.setdefault('TRITON_CACHE_DIR','/workspace/A_1007/.cache/triton-normal-tf32-probe')
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
    paths={k:ROOT/f'.work/zipvoice/{v}/code/online_branch_stats_runtime_kernel.py' for k,v in [('control','b64'),('candidate','b64_normtf32')]}
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
                compiled[name]=fn[(triton.cdiv(t,64),64,4)](q,k,pq,e,mask,v1,o1,stats,t,False,64,32,'tf32' if name=='candidate' else 'ieee',1,num_warps=4,num_stages=1,enable_fp_fusion=False)
                fn[(triton.cdiv(t,64),64,4)](q,k,pq,e,mask,v2,o2,stats,t,False,64,32,'tf32' if name=='candidate' else 'ieee',2,num_warps=4,num_stages=1,enable_fp_fusion=False)
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
        errors={}
        for i,key in enumerate(('stats','branch1','branch2')):
            a,b=data['control'][i],data['candidate'][i]
            errors[key]={'relative_l2':((a-b).norm()/a.norm().clamp_min(1e-20)).item(),'max_abs':(a-b).abs().max().item()}
        samples={name:[] for name in kernels}
        for name in ('control','candidate','candidate','control','candidate','control','control','candidate'):
            for _ in range(10):
                start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                start.record();graphs[name].replay();end.record();end.synchronize()
                samples[name].append(start.elapsed_time(end))
        med={name:statistics.median(values) for name,values in samples.items()}
        resources={name:{'registers':obj.n_regs,'shared_bytes':obj.metadata.shared,'global_scratch_bytes':obj.metadata.global_scratch_size} for name,obj in compiled.items()}
        row={'frames':t,'batch':64,'heads':4,'full_value_width':12,'geometry':{'query':64,'key':32,'warps':4,'stages':1},'samples_ms':samples,'median_ms':med,'gain_percent':100*(med['control']-med['candidate'])/med['control'],'errors':errors,'resources':resources,'all_outputs_and_stats_finite':True,'all_masked_row_included':True}
        records.append(row);print(json.dumps({k:row[k] for k in ('frames','median_ms','gain_percent','errors')}),flush=True)
        del data,graphs,q,k,pq,e,mask,v1,v2
        torch.cuda.empty_cache()
    report={'status':'synthetic_complete_normal_write_read_microprobe_not_deployment_acceptance','preflight':preflight,'source_sha256':{k:sha(p) for k,p in paths.items()},'records':records,'limits':['Synthetic inputs and JIT graph events, not TensorRT AOT or application E2E','No numerical error cutoff; complete path mapping and CER/UTMOS/SIM-o required before retention','Control IEEE; candidate authorized ordinary TF32 with RNA; geometry, mask, positional FMA, stats dependencies unchanged']}
    (ROOT/'reports/sm89/zipvoice/a1007/b64/history/028-normal-tf32-microprobe.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
