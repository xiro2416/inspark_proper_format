#!/usr/bin/env python3
"""Collect real first-chunk channel statistics once for both quantization arms."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gpu', type=int, default=7, choices=(7,))
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--config', default='artifacts/current_release/runtime.yaml')
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--requests', type=int, default=128)
    ap.add_argument('--components',nargs='+',choices=('target','draft','cfm','vocoder'),default=['draft','cfm'])
    args = ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    import torch
    from inspark_infer.runtime.device import GPULease
    from inspark_infer.runtime.config import load
    from inspark_infer.runtime.engine import Engine
    from inspark_infer.quantization.unified import ChannelObserver, iter_roles, role_specs, save_artifact

    manifest = json.loads(args.manifest.read_text())
    cases = manifest.get('calibration', manifest.get('splits', {}).get('calibration'))
    if cases is None:
        cases = [c for c in manifest.get('cases', []) if c.get('split') == 'calibration']
    cases = cases[:args.requests]
    if len(cases) != args.requests:
        raise ValueError('Manifest has insufficient calibration requests')
    config = load(args.config)
    config['max_batch'] = args.batch
    records = []
    cases = [dict(c, id=c.get('id', c.get('case_id'))) for c in cases]
    reference_map = {r['voice_id']: r['path'] for r in manifest.get('references', [])}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with GPULease(args.gpu):
        engine = Engine(config)
        handles = []
        try:
            voice_paths = {}
            for case in cases:
                source = case.get('reference_audio') or case.get('voice_path') or case.get('audio')
                if source is None:
                    voice = case.get('voice_id') or case.get('voice')
                    source = reference_map.get(voice) or manifest.get('voices', {})[voice]
                    if isinstance(source, dict):
                        source = source.get('path') or source.get('reference_audio')
                voice_paths[case['id']] = str(source)
            voices = {p: f'calibration-voice-{i}' for i, p in enumerate(dict.fromkeys(voice_paths.values()))}
            for path, voice in voices.items():
                engine.prepare_reference(voice, path)
            roles = list(iter_roles(engine,components=tuple(args.components)))
            observers = {r.path: ChannelObserver(r) for r in roles if r.precision != 'bf16'}
            for role in roles:
                if role.path in observers:
                    handles.append(role.module.register_forward_pre_hook(observers[role.path]))
            engine.head_batch_barrier = True
            for start in range(0, len(cases), args.batch):
                group = cases[start:start + args.batch]
                pending = {c['id'] for c in group}
                began = time.perf_counter()
                for case in group:
                    ident = case['id']
                    engine.create_session(ident, voices[voice_paths[ident]], case['seed'], case.get('emotion'))
                    engine.push_text(ident, case['text'])
                    engine.finish_input(ident)
                steps = 0
                while pending:
                    def receive(event):
                        pending.discard(event['request_id'])
                    engine.run_ready(on_chunk=receive)
                    steps += 1
                    if steps > 256:
                        raise RuntimeError(f'First chunk stalled: {sorted(pending)}')
                    for case in group:
                        if engine.sessions[case['id']]['error']:
                            raise RuntimeError(engine.sessions[case['id']]['error'])
                for case in group:
                    state = engine.sessions[case['id']]
                    records.append(dict(id=case['id'], rounds=state.get('rounds'),
                                        kv_head_lengths=state.get('kv_head_lengths'),
                                        first_segment=state['parts'][0],
                                        accepted=state.get('accepted'),
                                        first_chunk_samples=state['chunks'][0]['sample_end']))
                    engine.cancel(case['id'])
                print(json.dumps(dict(completed=len(records), total=len(cases),
                                      wave_seconds=time.perf_counter()-began)), flush=True)
            for handle in handles:
                handle.remove()
            handles.clear()
            torch.cuda.synchronize()
            metadata = dict(schema=1, manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                            calibration_requests=len(cases), calibration_reference='original_float_weights_no_graph',
                            alpha=1.0, hardware=torch.cuda.get_device_name(), torch=torch.__version__,
                            cfm_intervals=[[0,.25],[.25,.5],[.5,.75],[.75,1]],
                            observations=records)
            for scheme in ('fp8', 'int8_smoothquant'):
                specs = role_specs(roles, observers, scheme)
                output = args.output_dir / f'{scheme}.json'
                sha = save_artifact(output, dict(metadata, scheme=scheme, role_specs=specs))
                print(json.dumps(dict(artifact=str(output), sha256=sha, roles=len(specs))), flush=True)
            torch.save({name: observer.amax.detach().cpu() for name, observer in observers.items()},
                       args.output_dir / 'channel_amax.pt')
        finally:
            for handle in handles:
                handle.remove()
            engine.close()


if __name__ == '__main__':
    main()
