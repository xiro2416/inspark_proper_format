"""Test moving unchanged TF32 weight rounding from FFN prologue to setup."""
import os
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID';os.environ['CUDA_VISIBLE_DEVICES']='1'
os.environ['TRITON_CACHE_DIR']='/workspace/A_1007/.cache/triton-f32-weight-prepack';os.environ['CUDA_CACHE_PATH']='/workspace/A_1007/.cache/cuda';os.environ['TMPDIR']='/workspace/A_1007/.cache/tmp';os.environ['XDG_CACHE_HOME']='/workspace/A_1007/.cache'
import argparse,fcntl,importlib.util,json,statistics,subprocess,sys,time,ast
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from run_zipvoice_validation import sha
from zipvoice_selected_route import idle_gpu_preflight


def live(pid):
    p=subprocess.run(['ps','-p',str(pid),'-o','stat='],capture_output=True,text=True)
    return p.returncode==0 and bool(p.stdout.strip()) and not p.stdout.strip().startswith('Z')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--after-pid',type=int,required=True);p.add_argument('--batch',type=int,choices=(4,8,16,32,64),default=64);p.add_argument('--report',default='034-f32-weight-prepack.json');args=p.parse_args();assert Path(args.report).name==args.report
    print(f'Waiting specific coordinator {args.after_pid}',flush=True)
    while live(args.after_pid):time.sleep(10)
    lease=Path('/workspace/.cache/inspark/gpu-locks/1.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX)
    preflight=idle_gpu_preflight()
    import torch,triton
    base=ROOT/f'.work/zipvoice/b{args.batch}{"_geo1" if args.batch==4 else ""}/code'
    source=base/'linear_f32_activation_tf32_rna_kernel.py'
    tree=ast.parse(source.read_text());removed=[]
    class Rewrite(ast.NodeTransformer):
        def visit_FunctionDef(self,node):
            if node.name=='linear_f32_activation':node.name='linear_f32_activation_prepacked'
            return self.generic_visit(node)
        def visit_Assign(self,node):
            if len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='w' and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and node.value.func.attr=='inline_asm_elementwise':
                removed.append(True);return ast.Pass()
            return self.generic_visit(node)
    candidate=ast.unparse(ast.fix_missing_locations(Rewrite().visit(tree)))+'\n';assert len(removed)==1
    aot=ast.parse((base/'f32_tf32_rna_aot.py').read_text())
    call=next(x for x in ast.walk(aot) if isinstance(x,ast.Call) and isinstance(x.func,ast.Name) and x.func.id=='linear_f32_activation')
    bm,bn,bk=[ast.literal_eval(call.args[i]) for i in (7,8,9)]
    candidate+='''\n@triton.jit
def round_static_weight(W, Y, N, BLOCK:tl.constexpr):
    offsets=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    x=tl.load(W+offsets,offsets<N,0)
    x=tl.inline_asm_elementwise('cvt.rna.tf32.f32 $0, $1;',constraints='=f,f',args=[x],dtype=tl.float32,is_pure=True,pack=1)
    tl.store(Y+offsets,x,offsets<N)
'''
    target=ROOT/f'.work/f32-weight-prepack/b{args.batch}/kernel.py';target.parent.mkdir(parents=True,exist_ok=True);target.write_text(candidate)
    def load(name,path):
        spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
    old=load('f32_prepack_control',source);new=load('f32_prepack_candidate',target)
    torch.manual_seed(9102);records=[]
    for t in (300,380,460,600,760,920):
        for n in (1152,1536,1920):
            m=args.batch*t;x=torch.randn(m,512,device='cuda')*.1;w=torch.randn(512,n,device='cuda')*.02;bias=torch.randn(n,device='cuda')*.01;packed=torch.empty_like(w)
            # One-time model-weight setup work; original W remains intact.
            new.round_static_weight[(triton.cdiv(w.numel(),256),)](w,packed,w.numel(),256)
            outputs={k:torch.full((m,n),float('nan'),device='cuda') for k in ('control','candidate')};graphs={};compiled={}
            def launch(kind):
                fn=old.linear_f32_activation if kind=='control' else new.linear_f32_activation_prepacked
                compiled[kind]=fn[(triton.cdiv(m,bm)*triton.cdiv(n,bn),)](x,w if kind=='control' else packed,bias,outputs[kind],m,512,n,bm,bn,bk,4.,0.07999999821186066,0.03500000014901161,'tf32',num_warps=4,num_stages=2,enable_fp_fusion=False)
            for kind in outputs:
                for _ in range(3):launch(kind)
                torch.cuda.synchronize();assert torch.isfinite(outputs[kind]).all().item()
                g=torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):launch(kind)
                graphs[kind]=g
            assert torch.equal(outputs['control'],outputs['candidate']),'Prepacking altered source TF32 math'
            samples={k:[] for k in outputs}
            for kind in ('control','candidate','candidate','control','candidate','control','control','candidate'):
                for _ in range(5):
                    a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record();graphs[kind].replay();b.record();b.synchronize();samples[kind].append(a.elapsed_time(b))
            med={k:statistics.median(v) for k,v in samples.items()}
            row={'batch':args.batch,'frames_at_stage':t,'M':m,'N':n,'median_ms':med,'gain_percent':100*(med['control']-med['candidate'])/med['control'],'samples_ms':samples,'all_elements_bitwise_equal':True,'resources':{k:{'registers':v.n_regs,'spills':v.n_spills,'shared_bytes':v.metadata.shared,'global_scratch_bytes':v.metadata.global_scratch_size} for k,v in compiled.items()}}
            records.append(row);print(json.dumps({k:v for k,v in row.items() if k!='samples_ms'}),flush=True)
            del graphs,compiled,outputs,x,w,bias,packed;torch.cuda.empty_cache()
    report={'status':'synthetic_f32_existing_weight_rounding_prepack_probe_complete_not_accepted','preflight':preflight,'original_kernel_sha256':sha(source),'candidate_kernel_sha256':sha(target),'records':records,'semantics':'Same TF32 RNA weights, inputs, bias, Float32 accumulation and original stableactivation. Original raw W retained; rounding moved to one-time GPU setup. No calibration or source weight/scale change.','limits':['Synthetic JIT fullwork probe only, not AOT or application E2E','Deployment would require exact derived-weight mapping, setup/prepack cost disclosure and full model validation']}
    (ROOT/f'reports/sm89/zipvoice/a1007/b{args.batch}/history'/args.report).write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
