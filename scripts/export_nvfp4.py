"""Export the current B64 graph family with native FP4-ready ModelOpt operators."""
import argparse,os,json,hashlib
from pathlib import Path
def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,default=7);p.add_argument('--component',choices=['target','draft','context','prefill','latent','latent_suffix','cfm','vocoder'],required=True)
    p.add_argument('--batch',type=int,default=64);p.add_argument('--config',required=True);p.add_argument('--calibration',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
    import torch
    from inspark_infer.runtime.config import load
    from inspark_infer.runtime.device import GPULease
    from inspark_infer.runtime.engine import Engine
    from inspark_infer.quantization.nvfp4 import install
    from inspark_infer.build.nvfp4_graph import export_nvfp4
    from inspark_infer.build.nvfp4_models import TargetVerification,ContextKV,Prefix
    from inspark_infer.build.unified_draft_export import DraftBackbone,input_names
    from inspark_infer.build.unified_prefix_export import from_engine
    from inspark_infer.build.unified_acoustic_export import FourStepCFM,TailWaveNetDiT
    from trt113_provenance import capture_provenance,capture_onnx_artifact,file_record
    torch.set_num_threads(8);recipe=json.loads(args.calibration.read_text());cfg=load(args.config);cfg['max_batch']=args.batch
    with GPULease(args.gpu):
        engine=Engine(cfg)
        try:
            with torch.inference_mode(),torch.cuda.stream(engine.model.stream):
                component='draft' if args.component in ['draft','context'] else 'target' if args.component in ['target','prefill','latent','latent_suffix'] else args.component
                provenance=capture_provenance(component,cfg,args.config,model=engine)
                roles=install(engine,recipe);b=args.batch
                if args.component=='target':
                    model=TargetVerification(engine.rt.engine.target);inputs=(torch.zeros(b,8,1280,device='cuda'),torch.ones(b,1,8,88,device='cuda',dtype=torch.bool),
                        *[torch.zeros(b,20,80,64,device='cuda',dtype=torch.bfloat16) for _ in range(48)])
                    names=['x','mask']+[v for i in range(24) for v in [f'k_cache_in_{i}',f'v_cache_in_{i}']]
                    outputs=['logits','selected','final']+[v for i in range(24) for v in [f'k_append_{i}',f'v_append_{i}']];kind='full_verification';frames=8
                elif args.component=='draft':
                    model=DraftBackbone(engine.rt.engine.draft);inputs=(torch.zeros(b,7,1280,device='cuda'),torch.ones(b,1,7,87,device='cuda',dtype=torch.bool),
                        *[torch.zeros(b,20,80,64,device='cuda') for _ in range(6)])
                    names=input_names();outputs=['hidden','base'];kind='backbone';frames=7
                elif args.component=='context':
                    model=ContextKV(engine.rt.engine.draft);inputs=(torch.zeros(b*8,1280,device='cuda'),);names=['projected_context'];outputs=['keys','values'];kind='context_kv';frames=8
                elif args.component in ['prefill','latent']:
                    model=Prefix(engine,args.component);frames=48 if args.component=='prefill' else 80
                    inputs=(torch.zeros(b,frames,1280,device='cuda'),torch.ones(b,frames,device='cuda',dtype=torch.int64));names=['x','keep']
                    outputs=['last_logits','packed_kv','selected','final'] if args.component=='prefill' else ['latent'];kind=args.component
                elif args.component=='latent_suffix':
                    model=Prefix(engine,'latent_suffix');frames=40;kind='latent_cached_suffix'
                    inputs=(torch.zeros(b,40,1280,device='cuda'),torch.ones(b,48,device='cuda',dtype=torch.int64),torch.zeros(24,2,b,20,48,64,device='cuda'))
                    names=['x','prefix_keep','past'];outputs=['latent_suffix']
                elif args.component=='cfm':
                    model=FourStepCFM(TailWaveNetDiT(engine.student.model,258));frames=310;kind='full_solver'
                    inputs=(torch.zeros(b,80,310,device='cuda'),torch.zeros(b,80,310,device='cuda'),torch.full((b,),310,device='cuda',dtype=torch.int64),
                        torch.zeros(b,192,device='cuda'),torch.zeros(b,310,512,device='cuda'),(torch.arange(310,device='cuda')[None,None]<258).expand(b,1,-1).contiguous())
                    names=['x','prompt','lengths','style','mu','mask'];outputs=['output']
                else:
                    model=engine.tts.bigvgan;inputs=(torch.zeros(b,80,52,device='cuda'),);names=['mel'];outputs=['pcm'];frames=52;kind='vocoder'
                # Official export uses inferred input/output names; rename after packing.
                export_nvfp4(model.eval(),inputs,args.output)
                import onnx
                graph=onnx.load(args.output)
                # Initializers are also graph inputs; only real data inputs are renamed.
                initializers={v.name for v in graph.graph.initializer};real=[v.name for v in graph.graph.input if v.name not in initializers]
                rename=dict(zip(real,names));rename.update(zip([v.name for v in graph.graph.output],outputs))
                if len(real)!=len(names):raise ValueError((real,names))
                for node in graph.graph.node:
                    for field in [node.input,node.output]:
                        for i,name in enumerate(field):field[i]=rename.get(name,name)
                for item in [*graph.graph.input,*graph.graph.output,*graph.graph.value_info]:item.name=rename.get(item.name,item.name)
                onnx.save_model(graph,args.output,save_as_external_data=True,all_tensors_to_one_file=True,location=args.output.name+'.data',size_threshold=1024)
                artifact=capture_onnx_artifact(args.output)
                manifest=[{k:v for k,v in r.items() if k!='module'} for r in roles if r['component']==component]
                result={'component':component,'kind':kind,'batch':b,'frames':frames,'kv_limit':80,'prompt_frames':258 if args.component=='cfm' else None,
                    'onnx':str(args.output.resolve()),'onnx_sha256':artifact['sha256'],'onnx_artifact':artifact,'provenance':provenance,'plugins':[],
                    'export_settings':{'precision':'nvfp4_dynamic_w4a4_fp32_interfaces','cfm_intervals':[[0,.25],[.25,.5],[.5,.75],[.75,1]]},
                    'quantization_recipe':{'scheme':'nvfp4','calibration':file_record(args.calibration,'calibration_artifact'),'role_manifest':{'roles':manifest},
                        'role_specs_sha256':hashlib.sha256(json.dumps({k:v for k,v in recipe['role_specs'].items() if k.startswith(component+'.')},sort_keys=True).encode()).hexdigest()}}
                args.output.with_suffix('.export.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'component':args.component,'onnx':str(args.output)}),flush=True)
        finally:engine.close()
if __name__=='__main__':main()
