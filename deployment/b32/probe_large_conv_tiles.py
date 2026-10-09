"""Target the measured expensive 192-channel/K11 implicit convolution region."""
import argparse
import json
from pathlib import Path
import torch
from deployment.b32.implicit_int8_probe import ImplicitConv
from deployment.b32.probe_fir import graph_measure

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--batch",type=int,default=32);parser.add_argument("--out",type=Path,default=ROOT/"deployment/b32/history/large-conv-tile-probe.json");args=parser.parse_args()
    from inspark_infer.runtime.device import GPULease,select_gpu
    rows=[]
    with GPULease(1),torch.inference_mode():
        select_gpu(1);torch.manual_seed(9132)
        x=torch.randn(args.batch,192,1664,device='cuda');w=torch.randint(-8,9,(192,192,11),device='cuda',dtype=torch.int8)
        act=torch.tensor(.03125,device='cuda');ws=torch.full((192,),.0078125,device='cuda');bias=torch.randn(192,device='cuda')*.01;smooth=torch.ones(192,device='cuda')
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        for dilation in (1,3,5):
            expected=None
            for bm,bn,bk in [(32,32,64),(32,64,64),(32,128,64),(64,64,64),(64,128,64),(64,64,128),(128,64,64)]:
                op=ImplicitConv(w,act,ws,bias,smooth,padding=5*dilation,dilation=dilation,bm=bm,bn=bn,bk=bk)
                measured,y=graph_measure(lambda:op(x),stream)
                if expected is None:expected=y
                if not torch.equal(y,expected):raise RuntimeError('Tile changed exact INT32 accumulation/scaling output')
                rows.append(dict(dilation=dilation,tile=[bm,bn,bk],p50_ms=measured['p50_ms'],registers=op.compiled.n_regs,shared_memory=op.compiled.metadata.shared,bit_identical=True))
                print(json.dumps(rows[-1]),flush=True)
                (args.out).write_text(json.dumps(dict(shape=[args.batch,192,1664],kernel=11,scope='synthetic same-shape complete quantize+implicit-convolution Graph; only tile varies; not E2E',rows=rows),indent=2)+'\n')


if __name__=='__main__':main()
