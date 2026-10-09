"""Export the exact ModelOpt NVFP4 storage format on CPU, without GPU math."""
import argparse,json,hashlib
from pathlib import Path
import torch
from safetensors.torch import load_file,save_file
from inspark_infer.build.nvfp4_graph import MatrixWeightOp
from modelopt.torch.quantization.qtensor import NVFP4QTensor

def main():
    p=argparse.ArgumentParser();p.add_argument('--weights',type=Path,required=True);p.add_argument('--calibration',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    from export_weights import fold,key_for
    torch.set_num_threads(8);recipe=json.loads(args.calibration.read_text());args.out.mkdir(parents=True,exist_ok=True)
    summaries={}
    for component in ['target','draft','cfm','vocoder']:
        source=load_file(str(args.weights/'unquantized'/f'{component}.safetensors'));state=fold(source);roles={}
        for role,spec in recipe['role_specs'].items():
            if not role.startswith(component+'.'):continue
            key=key_for(role,component);w=state[key].float()
            if component=='target':w=w.t().contiguous()
            if spec['precision']=='bf16':state[key]=w.bfloat16();roles[role]=dict(spec,state_key=key,transpose_linear_from_checkpoint=component=='target',storage_dtype='bfloat16');continue
            if spec['precision']=='fp8':
                scale=torch.tensor(spec['weight_scale'],dtype=torch.float32)
                broadcast=[1]*w.ndim;broadcast[spec['weight_axis']]=scale.numel()
                state[key]=(w/scale.reshape(broadcast)).clamp(-448,448).to(torch.float8_e4m3fn).contiguous()
                state[key+'.weight_scale']=scale
                state[key+'.input_scale']=torch.tensor(spec['input_scale'],dtype=torch.float32)
                roles[role]=dict(spec,state_key=key,original_shape=list(w.shape),stored_layout=('original convolution layout' if w.ndim==3 else 'canonical GEMM out,in'),transpose_linear_from_checkpoint=component=='target',storage_dtype='float8_e4m3fn',weight_norm_folded=True)
                continue
            transpose=w.ndim==3 and spec['weight_input_axis']==0
            if transpose:w=w.transpose(0,1).flip(-1).contiguous()
            shape=list(w.shape)
            if w.ndim==3 and spec.get("conv_layout")=="tap_channel":w=w.permute(0,2,1).contiguous()
            matrix=w.flatten(1).half();k=matrix.shape[-1];pad=(-k)%16
            matrix=torch.nn.functional.pad(matrix,(0,pad))
            qt,block_scale,global_scale=NVFP4QTensor.quantize(matrix,16)
            state[key]=qt._quantized_data.contiguous()
            state[key+'.block_scale']=block_scale.to(torch.float8_e4m3fn).contiguous()
            state[key+'.global_scale']=global_scale.float()
            state[key+'.activation_global_scale']=torch.tensor(spec['activation_amax']).half().float()/(6*448)
            roles[role]=dict(spec,state_key=key,original_shape=shape,canonical_gemm_shape=list(matrix.shape),padding_k=pad,
                transpose_flip_from_checkpoint=transpose,transpose_linear_from_checkpoint=component=='target',packed_nibble_order='low=evenK;high=oddK',
                weight_norm_folded=True,storage_dtype='uint8:two_E2M1_values_per_byte')
        path=args.out/f'{component}.safetensors';save_file({k:v.contiguous().clone() for k,v in state.items()},str(path))
        restored=load_file(str(path))
        for key,value in state.items():assert torch.equal(value.reshape(-1).view(torch.uint8),restored[key].reshape(-1).view(torch.uint8)),key
        path.with_suffix('.json').write_text(json.dumps({'component':component,'scheme':recipe['scheme'],'roles':roles,'outside_role_parameters':'unchanged original dtype and value'},indent=2)+'\n')
        summaries[component]={'bytes':path.stat().st_size,'roles':len(roles)};print(component,summaries[component],flush=True)
    (args.out/'quantization_config.json').write_text(json.dumps(recipe,indent=2)+'\n')
    (args.out/'export_summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
if __name__=='__main__':main()
