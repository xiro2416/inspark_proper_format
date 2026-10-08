"""Evaluate CER, UTMOS and SIM-o using the user's existing local model assets."""
import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys
import unicodedata
from unittest.mock import patch


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def normalize(text):
    return ''.join(c for c in unicodedata.normalize('NFKC', text).lower() if c.isalnum())


def edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[-1] + 1,
                               previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True,
                        help='JSON list of {path, target_text, reference_wav}')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda:0'], default='cuda:0')
    args = parser.parse_args()
    rows=json.loads(args.inputs.read_text())
    if not isinstance(rows,list) or not rows:raise ValueError('Expected nonempty audio inventory')
    for row in rows:
        for field in ('path','reference_wav'):
            path=Path(row[field]).resolve()
            if not path.is_relative_to('/workspace') or not path.is_file():
                raise ValueError('Missing or non-workspace audio: '+str(path))
        for field,hash_field in (('path','wav_sha256'),('reference_wav','reference_sha256')):
            if hash_field in row and digest(Path(row[field]))!=row[hash_field]:
                raise ValueError('Audio hash mismatch: '+row[field])
    gpu_lock = None
    if args.device == 'cuda:0':
        import fcntl, subprocess
        assert os.environ.get('CUDA_VISIBLE_DEVICES') == '1'
        gpu_lock = (Path(__file__).resolve().parents[1]/'.gpu-inference.lock').open('a')
        fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        hardware = subprocess.check_output(['nvidia-smi', '-i', '1', '--query-gpu=uuid,power.limit,enforced.power.limit', '--format=csv,noheader,nounits'], text=True).strip()
        assert hardware == 'GPU-7dab7d6b-ac8c-7ccc-6410-916d3b7689b3, 400.00, 400.00'
    # These existing local checkpoints contain model configuration objects.
    os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio
    from funasr import AutoModel
    torch.set_num_threads(4)
    if args.device == 'cuda:0':
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '1' or torch.cuda.device_count() != 1:
            raise RuntimeError('Use exactly physical GPU 1')
        assert str(torch.cuda.get_device_properties(0).uuid)=='7dab7d6b-ac8c-7ccc-6410-916d3b7689b3'
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
    from inspark_infer.models.zipvoice.evaluation.utmos import UTMOS22Strong
    from inspark_infer.models.zipvoice.evaluation.ecapa_tdnn_wavlm import ECAPA_TDNN_WAVLM
    assets = (Path(__file__).resolve().parents[1]/'models/evaluation')
    mos_path = assets / 'mos/utmos22_strong_step7459_v1.pt'
    sim_path = assets / 'speaker_similarity/wavlm_large_finetune.pth'
    ssl_dir = assets / 'speaker_similarity/wavlm_large'
    mos = UTMOS22Strong()
    mos.load_state_dict(torch.load(mos_path, map_location='cpu', weights_only=False), strict=True)
    mos = mos.eval().to(args.device)
    original_hub_load = torch.hub.load
    def local_ssl_loader(repo, entry, *positional, **kwargs):
        # The existing ECAPA wrapper supplies dirname(ssl_dir), but the actual
        # hubconf.py and WavLM checkpoint both live inside ssl_dir.
        assert Path(repo) == ssl_dir.parent and entry == 'wavlm_local'
        assert kwargs.get('source') == 'local'
        assert Path(kwargs['ckpt']) == ssl_dir / 'wavlm_large.pt'
        return original_hub_load(str(ssl_dir), entry, *positional, **kwargs)
    with patch('torch.hub.load', side_effect=local_ssl_loader):
        sim = ECAPA_TDNN_WAVLM(feat_dim=1024, channels=512, emb_dim=256,
                              sr=16000, ssl_model_path=str(ssl_dir))
    sim_state = torch.load(sim_path, map_location='cpu', weights_only=False)['model']
    load = sim.load_state_dict(sim_state, strict=False)
    if load.missing_keys:
        raise RuntimeError(f'SIM-o missing model keys: {load.missing_keys}')
    sim = sim.eval().to(args.device)
    asr = AutoModel(model=str(assets / 'wer/paraformer-zh'), device=args.device,
                    disable_update=True, ncpu=4, disable_pbar=True)
    def load_wave(path):
        wave, sr = sf.read(path, dtype='float32', always_2d=True)
        if not np.isfinite(wave).all() or not len(wave):
            raise RuntimeError(f'Invalid waveform: {path}')
        wave = torch.from_numpy(wave.mean(axis=1)).unsqueeze(0)
        if sr != 16000:
            wave = torchaudio.functional.resample(wave, sr, 16000)
        return wave.squeeze(0)
    references = {}
    results = []
    for row in rows:
        wave = load_wave(row['path'])
        hypothesis = asr.generate(input=wave.numpy(), hotword='', use_itn=True,
                                  disable_pbar=True)[0]['text']
        truth, predicted = normalize(row['target_text']), normalize(hypothesis)
        if not truth:
            raise ValueError('Empty target text')
        device_wave = wave.to(args.device)
        with torch.inference_mode():
            utmos = float(mos(device_wave.unsqueeze(0), 16000).item())
            reference = row['reference_wav']
            if reference not in references:
                references[reference] = sim([load_wave(reference).to(args.device)])
            embedding = sim([device_wave])
            similarity = float(torch.nn.functional.cosine_similarity(
                embedding, references[reference], dim=-1).item())
        errors = edit_distance(truth, predicted)
        result = {**row, 'transcription': hypothesis, 'normalized_target': truth,
            'normalized_hypothesis': predicted, 'character_errors': errors,
            'reference_characters': len(truth), 'cer': errors / len(truth),
            'utmos': utmos, 'sim_o': similarity}
        assert np.isfinite([utmos, similarity]).all()
        results.append(result)
        print(json.dumps({'path': row['path'], 'cer': result['cer'],
                          'utmos': utmos, 'sim_o': similarity}, ensure_ascii=False), flush=True)
    total_chars = sum(r['reference_characters'] for r in results)
    report = {'status': 'complete', 'count': len(results), 'device': args.device,
        'input_inventory_sha256':digest(args.inputs),
        'cer_definition': 'NFKC/lowercase alphanumeric Chinese character edit distance; punctuation/spaces removed',
        'model_assets': {str(p): digest(p) for p in
            [mos_path, sim_path, ssl_dir / 'wavlm_large.pt', assets / 'wer/paraformer-zh/model.pt']},
        'sim_loader_mapping_fix': 'Use actual wavlm_large/hubconf.py directory, instead of its parent; checkpoint and architecture unchanged',
        'sim_unexpected_checkpoint_keys': load.unexpected_keys,
        'packages': {name: importlib.metadata.version(name) for name in
                     ['torch', 'torchaudio', 'funasr', 's3prl']},
        'cer': sum(r['character_errors'] for r in results) / total_chars,
        'utmos_mean': float(np.mean([r['utmos'] for r in results])),
        'sim_o_mean': float(np.mean([r['sim_o'] for r in results])), 'results': results}
    if args.device == 'cuda:0':
        report['physical_gpu'] = 1
        report['gpu_hardware_before'] = hardware
        report['evaluator_source_sha256'] = digest(Path(__file__))
    report['by_reference']={}
    report['evaluator_source_sha256']=digest(Path(__file__))
    report['by_route']={}
    for route in sorted({r.get('route','unspecified') for r in results}):
        group=[r for r in results if r.get('route','unspecified')==route]
        report['by_route'][route]={'count':len(group),
            'cer':sum(r['character_errors'] for r in group)/sum(r['reference_characters'] for r in group),
            'character_errors':sum(r['character_errors'] for r in group),
            'reference_characters':sum(r['reference_characters'] for r in group),
            'utmos_mean':float(np.mean([r['utmos'] for r in group])),
            'sim_o_mean':float(np.mean([r['sim_o'] for r in group]))}
    for reference in sorted({r['reference_wav'] for r in results}):
        group=[r for r in results if r['reference_wav']==reference]
        report['by_reference'][reference]={'count':len(group),
            'cer':sum(r['character_errors'] for r in group)/sum(r['reference_characters'] for r in group),
            'utmos_mean':float(np.mean([r['utmos'] for r in group])),
            'sim_o_mean':float(np.mean([r['sim_o'] for r in group]))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
