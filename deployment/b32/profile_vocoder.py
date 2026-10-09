"""Frozen-input layer diagnostics and a separate unprofiled component Graph timing."""
import argparse,json
from pathlib import Path
from collections import defaultdict
import statistics

ROOT=Path(__file__).resolve().parents[2]


def main():
    import torch
    from inspark_infer.runtime.device import GPULease,select_gpu
    from inspark_infer.ops.tensorrt.unified_ar import StaticEngine
    history=ROOT/'deployment/b32/history'
    parser=argparse.ArgumentParser();parser.add_argument('--component',choices=['vocoder','cfm'],default='vocoder');parser.add_argument('--plan',type=Path);parser.add_argument('--out',type=Path);parser.add_argument("--history-dir",type=Path,default=history);parser.add_argument("--batch",type=int,default=32);parser.add_argument("--capture",type=Path);args=parser.parse_args();history=args.history_dir;batch=args.batch
    frozen=torch.load(args.capture or history/'capture-acoustics-b32.pt',map_location='cpu',weights_only=True)
    selected=json.loads((history/'current-best-selected.json').read_text())
    plan=args.plan or Path(selected[args.component+'_plan'])
    with GPULease(1):
        select_gpu(1)
        engine=StaticEngine(plan,batch)
        wave=frozen['waves'][0]
        if args.component=='vocoder':inputs={'mel':wave['vocoder_inputs'][0].cuda().contiguous()}
        else:
            x,prompt,lengths,style,mu,mask=wave['cfm_inputs']
            inputs={name:v.cuda().contiguous() for name,v in [('x',x),('prompt',prompt),('lengths',lengths),('style',style),('mu',mu)]}
            inputs['times']=torch.tensor([[0.,.25]],device='cuda').expand(batch,2).contiguous()
        for _ in range(3):engine(inputs)
        torch.cuda.synchronize()
        class Recorder(engine.trt.IProfiler):
            def __init__(self):
                super().__init__();self.rows=defaultdict(list)
            def report_layer_time(self,name,ms):self.rows[name].append(float(ms))
        recorder=Recorder();engine.context.profiler=recorder
        for _ in range(3):
            engine(inputs);torch.cuda.synchronize()
        inspection=json.loads(plan.with_name('model.inspector.json').read_text())['Layers']
        inventory={layer['Name']:layer for layer in inspection}
        rows=[]
        for name,times in recorder.rows.items():
            layer=inventory.get(name,{})
            rows.append(dict(name=name,mean_ms=statistics.mean(times),n=len(times),
                layer_type=layer.get('LayerType'),tactic=layer.get('TacticName'),
                inputs=layer.get('Inputs'),constants=layer.get('Constants'),outputs=layer.get('Outputs')))
        rows.sort(key=lambda row:-row['mean_ms'])
        # A fresh context removes per-layer profiling from component timings.
        del engine
        engine=StaticEngine(plan,batch)
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):engine(inputs)
            torch.cuda.synchronize()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):engine(inputs)
            times=[]
            for _ in range(20):
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record(stream);graph.replay();end.record(stream);end.synchronize()
                times.append(start.elapsed_time(end))
            output=next(iter(engine.outputs.values()));assert torch.isfinite(output).all() and output.abs().max()>0
        report=dict(batch=batch,physical_gpu=1,plan=str(plan),engine_sha256=engine.engine_sha256,
            scope='layer diagnostics use IProfiler; separate unprofiled frozen-mel component Graph; not application E2E',
            layers=rows,component_graph_ms=dict(p50=statistics.median(times),mean=statistics.mean(times),samples=times))
        (args.out or history/f'profile-{args.component}-layers-b32.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(dict(layers=len(rows),component_graph_ms=report['component_graph_ms']['p50'],
            top_layers=[dict(name=row['name'],ms=row['mean_ms'],kind=row['layer_type']) for row in rows[:12]])),flush=True)


if __name__=='__main__':main()
