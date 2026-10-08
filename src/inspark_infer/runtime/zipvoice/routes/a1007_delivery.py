"""A_1007 adaptation of source B32 compact H2D and first-PCM/D2H overlap."""
import argparse
import importlib
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
ROOT = Path(os.environ['INSPARK_REPO_ROOT'])

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def finite_all(tensor, torch, chunk_elements=1 << 20):
    """Scan every element while bounding validation-only device temporaries."""
    flat = tensor.view(-1)
    for start in range(0, flat.numel(), chunk_elements):
        if not bool(torch.isfinite(flat[start:start + chunk_elements]).all()):
            return False
    return True

from inspark_infer.ops.tensorrt.zipvoice.engine import Engine

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch', type=int, choices=(1,2,4,8,16,32,64), required=True)
    p.add_argument('--pcm-chunk', type=int, default=None)
    p.add_argument('--pcm-workers', type=int, default=4)
    p.add_argument('--minimum-seconds', type=float, default=0)
    p.add_argument('--assets', type=Path, default=ROOT / 'downloads')
    p.add_argument('--engine-manifest', type=Path, help='Required verified target-GPU engine inventory')
    p.add_argument('--include-input-transfer', action='store_true', help='Measure CPU condition shaping and H2D on every request, with stable graph buffer addresses')
    p.add_argument('--nvtx-range', action='store_true', help='Expose measured requests as zvoice_measure for profiler capture')
    p.add_argument('--inspect-context', action='store_true', help='Save context-bound engine inspection after initialization, outside measurement')
    p.add_argument('--shared-context-workspace', action='store_true', help='Share context scratch memory across serial text/FM/Vocos execution on the same stream')
    p.add_argument('--arena-istft', action='store_true', help='Full-batch cuFFT/CENTER ISTFT in dead serial Vocos scratch')
    p.add_argument('--inputs', type=Path, required=True, help='Caller-owned safetensors from prepare_inputs.py')
    p.add_argument('--gpu', type=int, default=1)
    p.add_argument('--disable-text-reuse', action='store_true', help='Use the unchanged full-batch text engine even for identical inputs')
    p.add_argument('--warmup', type=int, default=1)
    p.add_argument('--repetitions', type=int, default=1)
    p.add_argument('--seed', type=int, default=9100)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--save-indices', type=int, nargs='+', default=None, help='Save selected WAVs after measured inference')
    p.add_argument('--dump-selected-state', action='store_true', help='Save diagnostic state/conditions for selected rows after measurement, outside E2E')
    p.add_argument('--functional-only', action='store_true', help='Validate complete execution without accepting timing/power under unresolved GPU co-load')
    args = p.parse_args()
    B = args.batch
    if args.save_indices is None: args.save_indices=list(range(B))
    if args.pcm_workers < 1 or (args.pcm_chunk is not None and args.pcm_chunk < 1): p.error('Positive PCM workers/chunk required')
    if args.engine_manifest is None:
        p.error('INT8 inference requires its explicit verified target inventory')
    if args.arena_istft and (not args.shared_context_workspace):
        p.error('Arena ISTFT requires serial shared context workspace')
    if args.gpu < 0 or args.warmup < 1 or args.repetitions < 1:
        p.error('One GPU, at least one warmup and one repetition required')
    if any((i < 0 or i >= B for i in args.save_indices)):
        p.error('Save indices must be within the selected batch')
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in (None, str(args.gpu)):
        p.error('CUDA_VISIBLE_DEVICES must select exactly --gpu')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    import torch
    import tensorrt as trt
    target_manifest = json.loads(args.engine_manifest.read_text())
    from inspark_infer.build.zipvoice import plugin_package
    package = plugin_package(B,target_manifest.get('plugin_package'))
    for name in ('dw_int8_plugin', 'normal_tf32_plugin' if B != 64 else 'online_branch_wide_plugin', 'online_nonlinear_rna_plugin', 'f32_tf32_rna_plugin', 'int8_nonlinear_value_plugin', 'i8_residual_plugin'):
        importlib.import_module(f'{package}.{name}')
    import numpy as np
    import soundfile as sf
    from safetensors.torch import load_file
    from inspark_infer.runtime.zipvoice.telemetry import PowerSampler
    from inspark_infer.runtime.zipvoice.request_telemetry import summarize_request_windows
    from inspark_infer.models.zipvoice.pcm import pcm_quantized_exact, ordered_pcm_chunks
    target_manifest = json.loads(args.engine_manifest.read_text()) if args.engine_manifest else None
    actual_uuid = str(torch.cuda.get_device_properties(0).uuid)
    if not actual_uuid.startswith('GPU-'):
        actual_uuid = 'GPU-' + actual_uuid
    if not target_manifest or actual_uuid != target_manifest.get('gpu_uuid'):
        raise SystemExit('Deployment CUDA UUID does not match the selected physical GPU')
    expected_sm = tuple(target_manifest['compute_capability']) if target_manifest else (12, 0)
    expected_trt = target_manifest['tensorrt'] if target_manifest else '11.3.0.99'
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability() != expected_sm:
        raise SystemExit('Engine inventory does not match the selected GPU architecture')
    if trt.__version__ != expected_trt:
        raise SystemExit('Engine inventory does not match TensorRT; rebuild and validate')
    workload = target_manifest['workload'] if target_manifest else {'batch': B, 'total_frames': 681, 'prompt_frames': 375, 'target_frames': 306, 'joint_tokens': 69, 'padded_tokens': 70, 'steps': 8, 't_shift': 0.5, 'guidance': 1.0, 'feat_scale': 0.1}
    fixed = {'batch': B, 'prompt_frames': 375, 'steps': 8, 't_shift': 0.5, 'guidance': 1.0, 'feat_scale': 0.1}
    if any((workload.get(k) != v for k, v in fixed.items())) or any((type(workload.get(k)) is not int or workload[k] <= 0 for k in ('total_frames', 'target_frames', 'joint_tokens', 'padded_tokens'))) or workload['total_frames'] != workload['prompt_frames'] + workload['target_frames'] or (workload['padded_tokens'] != workload['joint_tokens'] + 1):
        raise SystemExit('Unsupported workload or inconsistent frame/token lengths')
    if target_manifest and target_manifest['physical_gpu'] != args.gpu:
        raise SystemExit('Engine inventory does not match the selected physical GPU')
    total_frames, target_frames = (workload['total_frames'], workload['target_frames'])
    prompt_frames = workload['prompt_frames']
    joint_tokens, padded_tokens = (workload['joint_tokens'], workload['padded_tokens'])
    wave_samples = (target_frames - 1) * 256
    pcm_chunk = args.pcm_chunk or (6 if B == 32 else 16)
    torch.set_num_threads(8)
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (ROOT / '.gpu-inference.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    names = {key:key for key in ('fm','text','unique','vocos')}
    engines = {}
    for key, name in names.items():
        if target_manifest:
            entry = target_manifest['engines'][key]
            path = Path(entry['path'])
            if not path.is_absolute():
                path = args.engine_manifest.resolve().parent / path
            expected_hash = entry['sha256']
            for plugin_path, source_hash in entry.get('plugin_sources', {}).items():
                source_path = Path(plugin_path).resolve()
                if not source_path.is_relative_to(ROOT) or digest(source_path) != source_hash:
                    raise SystemExit('Experimental plugin source does not match engine inventory')
        if digest(path) != expected_hash:
            raise SystemExit(f'Engine integrity failure: {name}')
        if key == 'fm':
            shapes = {n: (B, total_frames, 100) for n in ('x', 'text_condition', 'speech_condition')}
            shapes.update(t=(B, 1, 1), guidance_scale=(B, 1, 1), padding_mask=(B, total_frames))
        elif key == 'vocos':
            shapes = {'mel': (B, 100, target_frames)}
        else:
            batch = 1 if key == 'unique' else B
            shapes = {'token_ids': (batch, padded_tokens), 'token_lens': (batch,), 'features_lens': (batch,), 'frame_positions': (1, total_frames)}
        engines[key] = Engine(path, trt, torch, args.shared_context_workspace, shapes)
    context_memory = {key: engine.context_memory for key, engine in engines.items()}
    if args.inspect_context:
        for key, engine in engines.items():
            inspector = engine.engine.create_engine_inspector()
            inspector.execution_context = engine.context
            (args.output / f'{key}-context-inspector.json').write_text(inspector.get_engine_information(trt.LayerInformationFormat.JSON))
    workspace = None
    if args.shared_context_workspace:
        if any((engine.engine.num_aux_streams for engine in engines.values())):
            raise RuntimeError('Shared context workspace requires zero auxiliary streams')
        workspace = torch.empty(max((engine.output_arena_size() for engine in engines.values())), dtype=torch.uint8, device='cuda')
        if workspace.data_ptr() % 256:
            raise RuntimeError('Context workspace is not 256-byte aligned')
        for engine in engines.values():
            engine.context.set_device_memory(workspace.data_ptr(), workspace.numel())
            engine.bind_output_arena(workspace)
    data = load_file(str(args.inputs))
    if set(data) != {'prompt_mel', 'token_ids', 'prompt_rms'}:
        raise SystemExit('Input keys must be prompt_mel, token_ids, prompt_rms')
    rows = data['token_ids'].shape[0]
    if rows not in (1, B) or data['token_ids'].shape != (rows, padded_tokens) or data['token_ids'].dtype != torch.int64:
        raise SystemExit(f'Expected INT64 token_ids [1|B,{padded_tokens}]')
    if data['prompt_mel'].shape != (rows, prompt_frames, 100) or data['prompt_mel'].dtype != torch.float32 or data['prompt_rms'].shape != (rows,):
        raise SystemExit(f'Expected FLOAT32 prompt_mel [rows,{prompt_frames},100], prompt_rms [rows]')
    if not torch.isfinite(data['prompt_mel']).all() or not torch.isfinite(data['prompt_rms']).all() or (data['prompt_rms'] <= 0).any():
        raise SystemExit('Invalid prompt values')
    shared = rows == 1 or (torch.equal(data['token_ids'], data['token_ids'][:1].expand_as(data['token_ids'])) and torch.equal(data['prompt_mel'], data['prompt_mel'][:1].expand_as(data['prompt_mel'])))
    shared = shared and not args.disable_text_reuse
    tokens = data['token_ids'].expand(B, -1).contiguous().cuda()
    speech = torch.nn.functional.pad(data['prompt_mel'].expand(B, -1, -1) * 0.1, (0, 0, 0, target_frames)).contiguous().cuda()
    rms = data['prompt_rms'].float().expand(B).cuda()[:, None]
    input_host = {key: torch.empty_like(value, device='cpu', pin_memory=True) for key, value in data.items()}
    input_device = {key: torch.empty_like(value, device='cuda') for key, value in data.items()}
    grid = torch.linspace(0.0, 1.0, 9).cuda()
    grid = 0.5 * grid / (1 + (0.5 - 1) * grid)
    times = [grid[k].expand(B).reshape(B, 1, 1).contiguous() for k in range(8)]
    guidance = torch.ones(B, 1, 1, device='cuda')
    state_buffer = torch.empty(B, total_frames, 100, device='cuda')
    text_buffer = torch.empty_like(state_buffer)
    mask_buffer = torch.empty(B, total_frames, dtype=torch.bool, device='cuda')
    window = torch.hann_window(1024).cuda()
    arena_istft = None
    if args.arena_istft:
        from inspark_infer.ops.cuda.zipvoice.arena_istft import ArenaISTFT
        arena_istft = ArenaISTFT(workspace, engines['vocos'].context_memory, B, target_frames, window, direct_cufft=True)
    graph = None
    stream = torch.cuda.Stream()
    first_host = torch.empty(wave_samples, device='cpu', pin_memory=True)
    first_ready = torch.cuda.Event()
    host = torch.empty((B - 1), wave_samples, device='cpu', pin_memory=True)
    lens1 = torch.full((1,), joint_tokens, dtype=torch.int64, device='cuda')
    feats1 = torch.full((1,), total_frames, dtype=torch.int64, device='cuda')
    frame_positions = torch.arange(total_frames, dtype=torch.int64, device='cuda')[None, :]
    lens = lens1.expand(B).contiguous()
    feats = feats1.expand(B).contiguous()
    retained = {}
    import subprocess
    power_limits_before = subprocess.check_output(['nvidia-smi','-i',str(args.gpu),'--query-gpu=power.limit,enforced.power.limit','--format=csv,noheader,nounits'],text=True).strip()
    if power_limits_before != '400.00, 400.00':
        raise RuntimeError(f'400W cap required: {power_limits_before}')
    report = {'status': 'running', 'shape': [B, total_frames, 100], 'workload': workload, 'input_sha256': digest(args.inputs), 'runner_sha256': digest(Path(__file__)), 'seed': args.seed, 'warmup': args.warmup, 'repetitions': args.repetitions, 'gpu_name': torch.cuda.get_device_name(), 'sm': expected_sm[0] * 10 + expected_sm[1], 'text_reuse': shared, 'engine_manifest_sha256': digest(args.engine_manifest) if args.engine_manifest else None, 'context_memory_bytes': context_memory, 'shared_context_workspace': args.shared_context_workspace, 'allocated_context_memory_bytes': workspace.numel() if workspace is not None else sum(context_memory.values()), 'serial_output_arena': args.shared_context_workspace, 'arena_istft': args.arena_istft, 'arena_istft_storage_bytes': arena_istft.storage_bytes if arena_istft else 0, 'cufft_workspace_bytes': arena_istft.fft.workspace_bytes if arena_istft else None, 'arena_istft_source_sha256': digest(ROOT / 'src/inspark_infer/ops/cuda/zipvoice/arena_istft.py') if arena_istft else None, 'input_transfer_included': args.include_input_transfer, 'clock': 'prepared input to ordered PCM; excludes file/tokenization/front-end/init/warmup/capture/WAV writes; NOT original G2P-inclusive benchmark clock'}
    report['text_reuse_disabled'] = args.disable_text_reuse
    report['pcm_chunk'] = pcm_chunk
    report['pcm_workers'] = args.pcm_workers
    report['position_mapping_audit_source_sha256'] = target_manifest['origin_mapping_source_sha256']
    report['power_limits_before_w'] = power_limits_before
    report['power_telemetry_note'] = 'NVML nvmlDeviceGetPowerUsage sampled maximum; not configured limit or instantaneous spike measurement'
    if args.include_input_transfer:
        report['clock'] = 'prepared CPU conditions including request shaping/H2D to ordered PCM; excludes file/tokenization/front-end/persistent buffer allocation/init/warmup/capture/WAV writes'

    def steps(state):
        for k in range(8):
            velocity = engines['fm']({'t': times[k], 'x': state, 'text_condition': text_buffer, 'speech_condition': speech, 'padding_mask': mask_buffer, 'guidance_scale': guidance}, stream)['velocity']
            state.add_(velocity.float() * (grid[k + 1] - grid[k]))

    @torch.inference_mode()
    def run(seed, pool):
        nonlocal graph
        start_ns = time.perf_counter_ns()
        start = start_ns / 1e9
        if args.include_input_transfer:
            for key, value in data.items():
                input_host[key].copy_(value)
                input_device[key].copy_(input_host[key], non_blocking=True)
            tokens.copy_(input_device['token_ids'].expand(B, -1))
            speech.zero_()
            speech[:, :prompt_frames].copy_(input_device['prompt_mel'].expand(B, -1, -1) * 0.1)
            rms.copy_(input_device['prompt_rms'].expand(B)[:, None])
        encoder = engines['unique'] if shared else engines['text']
        encoder_inputs = {'token_ids': tokens[:1] if shared else tokens, 'token_lens': lens1 if shared else lens, 'features_lens': feats1 if shared else feats}
        if 'frame_positions' in encoder.inputs:
            encoder_inputs['frame_positions'] = frame_positions
        out = encoder(encoder_inputs, stream)
        text_buffer.copy_(out['text_condition'].expand_as(text_buffer))
        mask_buffer.copy_(out['padding_mask'].expand_as(mask_buffer))

        def reset_state():
            state_buffer.normal_(generator=torch.Generator(device='cuda').manual_seed(seed))
        reset_state()
        if graph is None:
            steps(state_buffer)
            stream.synchronize()
            expected = state_buffer.cpu()
            reset_state()
            steps(state_buffer)
            stream.synchronize()
            reset_state()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                steps(state_buffer)
            reset_state()
            graph.replay()
            stream.synchronize()
            captured = state_buffer.cpu()
            if not torch.isfinite(expected).all() or not torch.isfinite(captured).all():
                raise RuntimeError('Nonfinite direct or captured ODE state')
            report['graph_bitwise_guard'] = bool(torch.equal(captured, expected))
            report['graph_state_relative_l2'] = float((captured - expected).norm() / expected.norm().clamp_min(1e-20))
            report['graph_state_max_abs'] = float((captured - expected).abs().max())
            del expected, captured
            print(json.dumps({'event': 'initial_graph_validation_complete', 'finite': True, 'total_frames': total_frames}), flush=True)
        reset_state()
        graph.replay()
        mel = (state_buffer[:, prompt_frames:, :].permute(0, 2, 1) / 0.1).contiguous()
        coeff = engines['vocos']({'mel': mel}, stream)['spectral_coefficients']
        if arena_istft:
            waves = arena_istft(coeff)
            arena_istft.scale_rms_(rms)
        else:
            magnitude, phase = coeff.chunk(2, dim=1)
            magnitude = magnitude.exp().clamp(max=100.0)
            spectrum = torch.complex(magnitude * phase.cos(), magnitude * phase.sin())
            waves = torch.istft(spectrum, n_fft=1024, hop_length=256, win_length=1024, window=window, center=True).clamp(-1, 1)
        if waves.shape != (B, wave_samples):
            raise RuntimeError('Unexpected native CENTER ISTFT waveform shape')
        first_wave = waves[0] if arena_istft else waves[0] * rms[0] / 0.1 if rms[0].item() < 0.1 else waves[0]
        first_host.copy_(first_wave.float(), non_blocking=True)
        first_ready.record(stream)
        scaled = waves[1:] if arena_istft else torch.where(rms[1:] < 0.1, waves[1:] * rms[1:] / 0.1, waves[1:])
        host.copy_(scaled, non_blocking=True)
        first_ready.synchronize()
        first = pcm_quantized_exact(first_host.numpy(), cache_fades=True)
        diagnostic_pcm = [first.copy()] if args.functional_only else None
        first_s = time.perf_counter() - start
        if 0 in args.save_indices:
            retained[0] = first.copy()
        stream.synchronize()
        count = 1
        for index, pcm in ordered_pcm_chunks(host.numpy(), pool, pcm_chunk, True, start_index=1):
            count += 1
            if diagnostic_pcm is not None:
                diagnostic_pcm.append(pcm.copy())
            if index in args.save_indices:
                retained[index] = pcm.copy()
        done_ns = time.perf_counter_ns()
        done_s = (done_ns - start_ns) / 1e9
        if count != B or not finite_all(state_buffer, torch) or (not finite_all(mel, torch)) or (not finite_all(waves, torch)):
            raise RuntimeError('Invalid inference result')
        if args.functional_only:
            import numpy as np
            expected_first = first_wave.float().cpu().numpy()
            expected_rest = scaled.float().cpu().numpy()
            assert np.array_equal(first_host.numpy(), expected_first)
            assert np.array_equal(host.numpy(), expected_rest)
            expected_pcm = [pcm_quantized_exact(expected_first, cache_fades=True)] + [pcm_quantized_exact(w, cache_fades=True) for w in expected_rest]
            assert len(diagnostic_pcm) == len(expected_pcm) == B
            assert all(np.array_equal(a, b) for a, b in zip(diagnostic_pcm, expected_pcm))
            report['overlap_all_batch_transfers_and_ordered_pcm_exact'] = True
            expected_speech = torch.nn.functional.pad(data['prompt_mel'].expand(B, -1, -1) * 0.1, (0, 0, 0, target_frames)).contiguous()
            assert torch.equal(speech.cpu(), expected_speech)
            assert torch.equal(tokens.cpu(), data['token_ids'].expand(B, -1))
            assert torch.equal(rms.cpu(), data['prompt_rms'].expand(B)[:, None])
            report['compact_input_all_batch_tokens_speech_rms_bitwise_exact'] = True
        return {'first_pcm_s': first_s, 'all_pcm_s': done_s, 'pcm_items': count, 'request_start_ns': start_ns, 'request_done_ns': done_ns}
    stream.wait_stream(torch.cuda.current_stream())
    with ThreadPoolExecutor(max_workers=args.pcm_workers) as pool, torch.cuda.stream(stream):
        for i in range(args.warmup):
            run(8100 + i, pool)
            print(json.dumps({'event': 'warmup_complete', 'index': i, 'pcm_items': B}), flush=True)
        gc.freeze()
        sampler = PowerSampler(args.gpu).start()
        try:
            with torch.cuda.nvtx.range('zvoice_measure') if args.nvtx_range else nullcontext():
                results = []
                measured_start = time.perf_counter()
                while len(results) < args.repetitions or time.perf_counter() - measured_start < args.minimum_seconds:
                    results.append(run(args.seed + len(results), pool))
                args.repetitions = len(results)
                report['repetitions'] = args.repetitions
        finally:
            telemetry = sampler.stop()
    telemetry['request_windows'] = summarize_request_windows(sampler.samples, results)
    if args.dump_selected_state:
        from safetensors.torch import save_file
        chosen = args.save_indices
        with torch.cuda.stream(stream):
            generator = torch.Generator(device='cuda').manual_seed(args.seed + args.repetitions - 1)
            initial = torch.empty_like(state_buffer).normal_(generator=generator)
            selected = {
                'initial_state': initial[chosen].cpu(), 'final_state': state_buffer[chosen].cpu(),
                'text_condition': text_buffer[chosen].cpu(), 'speech_condition': speech[chosen].cpu(),
                'padding_mask': mask_buffer[chosen].cpu(), 'time_grid': grid.cpu(),
            }
        save_file(selected, str(args.output / 'selected-state.safetensors'))
        report['diagnostic_selected_state_rows'] = chosen
        report['diagnostic_noise_batch'] = B
        report['diagnostic_noise_seed'] = args.seed + args.repetitions - 1
    for index, pcm in retained.items():
        sf.write(args.output / f'{index:04d}.wav', pcm, 24000, subtype='PCM_16')
    report['power_limits_after_w'] = subprocess.check_output(['nvidia-smi','-i',str(args.gpu),'--query-gpu=power.limit,enforced.power.limit','--format=csv,noheader,nounits'],text=True).strip()
    report.update(status='complete_functional_only' if args.functional_only else 'complete', results=results, power_telemetry=telemetry, performance_acceptance=False)
    if args.functional_only:
        report['gpu_isolation'] = 'Functional reference checks enabled; see preflight for device occupancy'
        report['power_attribution'] = 'Whole-board samples, not solely this request'
    (args.output / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
if __name__ == '__main__':
    main()
