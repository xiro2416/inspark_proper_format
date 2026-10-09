"""Probe the costly untuned INT8 regions identified by this target's profiler."""
import argparse,json,re
from pathlib import Path
import torch
from deployment.b32.implicit_int8_probe import ImplicitConv
from deployment.b32.probe_fir import graph_measure
from deployment.multibatch.schedule_plugin import TILES


def main():
    from inspark_infer.runtime.device import GPULease,select_gpu
    p=argparse.ArgumentParser();p.add_argument('--batch',type=int,required=True);p.add_argument('--profile',type=Path,required=True);p.add_argument('--export',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    profile=json.loads(a.profile.read_text());export=json.loads(a.export.read_text());rewrites=export['custom_implicit_int8_conv']['rewrites']
    grouped={}
    for layer in profile['layers']:
        match=re.fullmatch(r'b32_implicit_conv_(\d+)',layer['name'])
        if not match:continue
        i=int(match[1])
        if 16<=i<=21:continue  # already covered by the separate source-region probe
        r=rewrites[i];shape=layer['inputs'][0]['Dimensions'];o=layer['inputs'][1]['Dimensions'][0]
        key=tuple(shape+[o,r['kernel'],r['stride'],r['pad'],r['dilation'],r['expand'],r['output_shape'][2]])
        group=grouped.setdefault(key,dict(shape=shape,out_channels=o,geometry=r,indices=[],profile_ms=0.))
        group['indices'].append(i);group['profile_ms']+=layer['mean_ms']
    groups=sorted(grouped.values(),key=lambda r:-r['profile_ms'])[:6]
    report=dict(batch=a.batch,scope='Synthetic target-shape complete quantize+implicit-convolution Graph; chosen by actual retained layer costs. Exact outputs across schedules; not application E2E.',groups=[])
    with GPULease(1),torch.inference_mode():
        select_gpu(1);torch.manual_seed(7241);stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        for group in groups:
            b,c,f=group['shape'];o=group['out_channels'];r=group['geometry'];k=r['kernel']
            assert b==a.batch
            x=torch.randn(b,c,f,device='cuda');w=torch.randint(-8,9,(o,c,k),device='cuda',dtype=torch.int8)
            act=torch.tensor(.03125,device='cuda');ws=torch.full((o,),.0078125,device='cuda');bias=torch.randn(o,device='cuda')*.01;smooth=torch.ones(c,device='cuda')
            rows=[];expected=None
            for bm,bn,bk in TILES:
                op=ImplicitConv(w,act,ws,bias,smooth,stride=r['stride'],padding=r['pad'],dilation=r['dilation'],expand=r['expand'],bm=bm,bn=bn,bk=bk)
                base=((f-1)*r['expand']+1+2*r['pad']-r['dilation']*(k-1)-1)//r['stride']+1
                op.output_padding=(r['output_shape'][2]-base)*r['stride']
                measured,y=graph_measure(lambda:op(x),stream)
                assert list(y.shape)==r['output_shape']
                if expected is None:expected=y
                if not torch.equal(y,expected):raise RuntimeError('Schedule changed discrete/scaled output')
                ptx=op.compiled.asm['ptx'];assert 'mma.sync' in ptx and '.s8.s8.s32' in ptx
                rows.append(dict(tile=[bm,bn,bk],p50_ms=measured['p50_ms'],bit_identical=True,actual_signed_int8_mma=True,registers=op.compiled.n_regs,shared_memory=op.compiled.metadata.shared))
            group['rows']=rows;report['groups'].append(group)
            a.out.write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps(dict(indices=group['indices'],shape=group['shape'],kernel=k,baseline_ms=rows[0]['p50_ms'],best=min(rows,key=lambda r:r['p50_ms']))),flush=True)


if __name__=='__main__':main()
