"""Freeze actual static engine bindings after a real B64 first-chunk wave.

Capture is diagnostic: Graph-bound input references are remembered while
preparing deployment, then copied only after a complete real head. No timed
measurement or production route is altered.
"""
from pathlib import Path
import argparse
import torch

from benchmarks.benchmark_unified_first_chunk import prepare_references
from benchmarks.unified_first_chunk import load_manifest,run_wave,wave_cases
from inspark_infer.ops.tensorrt.unified_ar import StaticEngine
from inspark_infer.runtime.config import load
from inspark_infer.runtime.deployment import load as load_plan
from inspark_infer.runtime.device import GPULease,select_gpu
from inspark_infer.runtime.engine import Engine


def main():
    p=argparse.ArgumentParser();p.add_argument('--deployment',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    original=StaticEngine.__call__
    def remember(self,inputs):
        self.audit_binding_references=dict(inputs)
        return original(self,inputs)
    StaticEngine.__call__=remember
    with GPULease(7):
        select_gpu(7);config=load('artifacts/current_release/runtime.yaml');config['max_batch']=64;config['precision_batches']=[64]
        engine=Engine(config)
        try:
            manifest=load_manifest('artifacts/sm120_0924/unified/cases.json')
            prepare_references(engine,manifest);engine.prepare_deployment(load_plan(args.deployment))
            run_wave(engine,wave_cases(manifest['splits']['evaluation'],64,0),'framework-freeze',admission_mode='batch')
            objects=dict(target=engine.unified_first_chunk.provider.target_engine,draft=engine.unified_first_chunk.provider.draft_engine,**engine.unified_prefix.backends)
            result={'schema':1,'kind':'real_static_engine_bindings','deployment':str(args.deployment.resolve()),'manifest_sha256':manifest['manifest_sha256'],'batch':64,'inputs':{},'outputs':{}}
            with torch.cuda.stream(engine.model.stream),torch.inference_mode():
                for kind,obj in objects.items():
                    inputs=obj.audit_binding_references if kind in ('target','draft') else obj.inputs
                    result['inputs'][kind]={k:v.detach().cpu().clone() for k,v in inputs.items()}
                    result['outputs'][kind]={k:v.detach().cpu().clone() for k,v in obj.outputs.items()}
            args.out.parent.mkdir(parents=True,exist_ok=True);torch.save(result,args.out)
        finally:engine.close();StaticEngine.__call__=original


if __name__=='__main__':main()
