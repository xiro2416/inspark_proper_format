"""Create local candidate deployments; engine and model identities remain explicit."""
import argparse
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BUNDLE=ROOT/'local_assets/download/engines/rtx6000d_sm120/draft900_cfm800'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batches',type=int,nargs='+',default=[1,2,4,8,16])
    parser.add_argument('--cfm-directory',default='cfm',choices=['cfm','cfm-estimator'])
    parser.add_argument('--vocoder-directory',default='vocoder-gemm',choices=['vocoder','vocoder-conv2d','vocoder-gemm'])
    args=parser.parse_args()
    original=json.loads((ROOT/'configs/current/int8_smoothquant_b1.json').read_text())
    directory=ROOT/'configs/hardware/sm89/indextts';directory.mkdir(parents=True,exist_ok=True)
    from inspark_infer.runtime.unified_deployment import validate
    for batch in args.batches:
        if batch not in (1,2,4,8,16,32,64,128):raise ValueError('Unsupported local batch')
        assets=ROOT/'artifacts/sm89/int8_smoothquant'/f'b{batch}'
        plan=dict(original,batch=batch,status='local_sm89_candidate_not_yet_validated',
                  hardware=dict(gpu_name='NVIDIA GeForce RTX 4090',sm=89),
                  calibration=str(BUNDLE/'artifacts/current_release/calibration/int8_smoothquant.json'),
                  official_sources=str(BUNDLE/'artifacts/official_trtllm_dspark'),
                  batch_conditions=batch>1,latent_cached_prefix=batch>1,
                  component_calibrations={c:str(BUNDLE/'artifacts/current_release/calibration/int8_smoothquant_target_vocoder.json') for c in ('target','vocoder')})
        for c in ('target','draft','cfm','vocoder','prefill','latent'):
            path=assets/(args.cfm_directory if c=='cfm' else args.vocoder_directory if c=='vocoder' else c)/'model.plan.json'
            if not path.is_file():raise RuntimeError('Component not built: '+str(path))
            data=json.loads(path.read_text())
            if data.get('sm')!=89 or data.get('batch')!=batch:
                raise ValueError('Engine hardware/batch mismatch: '+str(path))
            plan[c+'_plan']=str(path)
        for backend in ('framework_dspark_adapter','native_dspark_worker_trt_compute'):
            label='framework' if backend.startswith('framework') else 'native'
            for graphs in (False,True):
                value=dict(plan,runtime_backend=backend,graphs=graphs)
                validate(value)
                name=f'int8_b{batch}_{label}_{"graph" if graphs else "direct"}.json'
                (directory/name).write_text(json.dumps(value,indent=2)+'\n')
        print('Materialized candidate configurations for batch',batch,flush=True)


if __name__=='__main__':main()
