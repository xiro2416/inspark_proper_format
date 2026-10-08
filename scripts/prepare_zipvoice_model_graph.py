"""Create isolated full-model CUDA Graph candidates from accepted application sources."""
import hashlib
import json
from pathlib import Path
import py_compile

ROOT=Path(__file__).resolve().parents[1]
ROUTES=ROOT/'src/inspark_infer/runtime/zipvoice/routes'


def main():
    baseline=json.loads((ROOT/'reports/sm89/zipvoice/a1007/migration-baseline.json').read_text())
    assert baseline['status']=='all_seven_migrations_accepted_optimization_pending'
    generated={}
    for source_name in ('a1007','a1007_delivery'):
        source=ROUTES/f'{source_name}.py';text=source.read_text()
        text=text.replace('A_1007 inherited INT8 application route; migration acceptance pending.',
            'Isolated A_1007 whole-model graph candidate; optimization validation pending.')
        start=text.index("        encoder = engines['unique'] if shared else engines['text']")
        reset=text.index('        def reset_state():',start)
        encoder=text[start:reset]
        post_start=text.index('        mel = (state_buffer[:, prompt_frames:',reset)
        post_end=text.index('        if waves.shape !=',post_start) if source_name=='a1007_delivery' else text.index('        stream.synchronize()\n        if waves.shape !=',post_start)
        post=text[post_start:post_end]
        body='        def model_body():\n'+''.join('    '+line+'\n' for line in encoder.splitlines() if line.strip())
        body+='            steps(state_buffer)\n'+''.join('    '+line+'\n' for line in post.splitlines())
        body+='            return mel, waves\n'
        replacement=body+'''
        def reset_state():
            # Fresh caller-owned noise stays outside capture on every request.
            state_buffer.normal_(generator=torch.Generator(device='cuda').manual_seed(seed))
        reset_state()
        if graph is None:
            direct_mel, direct_waves = model_body()
            stream.synchronize()
            expected = state_buffer.cpu()
            expected_waves = direct_waves.cpu()
            reset_state()
            model_body()
            stream.synchronize()
            reset_state()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                graph_mel, graph_waves = model_body()
            reset_state()
            graph.replay()
            stream.synchronize()
            captured = state_buffer.cpu()
            captured_waves = graph_waves.cpu()
            if not torch.isfinite(expected).all() or not torch.isfinite(captured).all() or not torch.isfinite(expected_waves).all() or not torch.isfinite(captured_waves).all():
                raise RuntimeError('Nonfinite direct or captured whole-model result')
            report['graph_bitwise_guard'] = bool(torch.equal(captured, expected))
            report['graph_wave_bitwise_guard'] = bool(torch.equal(captured_waves, expected_waves))
            if not report['graph_bitwise_guard'] or not report['graph_wave_bitwise_guard']:
                raise RuntimeError('Whole-model graph changed direct state or waveform')
            report['graph_state_relative_l2'] = float((captured - expected).norm() / expected.norm().clamp_min(1e-20))
            report['graph_state_max_abs'] = float((captured - expected).abs().max())
            report['capture_domain'] = 'text,8FM+Euler,mel,Vocos,CENTER_ISTFT,RMS; fresh H2D/noise and ordered PCM outside'
            del expected, captured, expected_waves, captured_waves, direct_mel, direct_waves
            print(json.dumps({'event': 'whole_model_graph_validation_complete', 'finite': True, 'total_frames': total_frames}), flush=True)
        reset_state()
        graph.replay()
        mel, waves = graph_mel, graph_waves
'''
        # Model outputs are local to the first invocation unless retained explicitly.
        text=text.replace('    graph = None\n','    graph = None\n    graph_mel = graph_waves = None\n',1)
        start=text.index("        encoder = engines['unique'] if shared else engines['text']")
        post_start=text.index('        mel = (state_buffer[:, prompt_frames:',start)
        post_end=text.index('        if waves.shape !=',post_start) if source_name=='a1007_delivery' else text.index('        stream.synchronize()\n        if waves.shape !=',post_start)
        text=text[:start]+replacement+text[post_end:]
        if source_name=='a1007':
            text=text.replace('        mel, waves = graph_mel, graph_waves\n','        mel, waves = graph_mel, graph_waves\n        stream.synchronize()\n')
        text=text.replace('        nonlocal graph\n','        nonlocal graph, graph_mel, graph_waves\n',1)
        target=ROUTES/f'{source_name}_graph.py';target.write_text(text);py_compile.compile(str(target),doraise=True)
        generated[source_name+'_graph']={'source':str(source.relative_to(ROOT)),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'candidate':str(target.relative_to(ROOT)),'candidate_sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
    report=ROOT/'reports/sm89/zipvoice/a1007/model-graph-preparation.json'
    report.write_text(json.dumps({'status':'isolated_candidate_source_prepared_not_validated','gate':'all7 migration accepted','sources':generated,'preserved':'All engines/plugins/weights/scales/mathematics; original application modules untouched','change':'Capture text+8FM/Euler+Vocos+ISTFT/RMS as one shape-bound graph; fresh input staging and noise reset remain outside'},indent=2)+'\n')
    print(json.dumps({'status':'prepared_not_validated','modules':list(generated)}))


if __name__=='__main__':main()
