"""Prepare disjoint bilingual natural-duration FP8 calibration and evaluation.

User audio is read-only. References are continuous four-second VAD windows;
their transcriptions are recomputed, never copied from a longer recording.
"""
import argparse
import gzip
import json
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from inspark_infer.runtime.zipvoice_fp8.common import profile_manifest, sha, write


def language(text):
    return 'zh' if any('\u4e00' <= c <= '\u9fff' for c in text) else 'en'


def corpus(base, split):
    rows = []
    with gzip.open(base / 'manifests' / f'{split}.jsonl.gz', 'rt') as f:
        for line in f:
            d = json.loads(line)
            if not d['id'].startswith('real_'):
                continue
            text = d['supervisions'][0]['text'].strip()
            audio = base / 'audio' / f"{d['id']}.wav"
            if text and '<' not in text and '>' not in text and audio.is_file():
                rows.append(dict(id=d['id'], text=text, language=language(text),
                                 audio=str(audio), duration=d['duration'], split=split))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path('/workspace/A_cleanup/zipvoice/data'))
    p.add_argument('--seed', type=int, default=9100)
    args = p.parse_args()
    if os.getenv('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Reference preparation is CPU-only')
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio
    from safetensors.torch import load_file
    from silero_vad import get_speech_timestamps, load_silero_vad
    from funasr import AutoModel
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    from inspark_infer.models.zipvoice.frontend import prepare
    from inspark_infer.models.zipvoice.tokenizer import EmiliaTokenizer

    torch.set_num_threads(8)
    folder = ROOT / 'outputs/fp8/data'
    folder.mkdir(parents=True, exist_ok=True)
    models = ROOT / 'models/evaluation/wer'
    vad = load_silero_vad()
    asr_zh = AutoModel(model=str(models / 'paraformer-zh'), device='cpu',
                       disable_update=True, ncpu=8, disable_pbar=True)
    processor = WhisperProcessor.from_pretrained(models / 'whisper-large-v3', local_files_only=True)
    asr_en = WhisperForConditionalGeneration.from_pretrained(
        models / 'whisper-large-v3', local_files_only=True).eval()
    tokenizer = EmiliaTokenizer(ROOT / 'models/zipvoice/config/tokens.txt')
    manifest = profile_manifest(1)
    all_rows = {s: corpus(args.data, s) for s in ('train', 'test')}
    rng = random.Random(args.seed)
    references = []
    legal = {'train': [], 'test': []}
    excluded = {'train': 0, 'test': 0}
    for split in ('train', 'test'):
        for lang in ('zh', 'en'):
            candidates = [x for x in all_rows[split] if x['language'] == lang and x['duration'] >= 4]
            rng.shuffle(candidates)
            count = 4 if split == 'train' else len(candidates)
            targets = [x for x in all_rows[split] if x['language'] == lang]
            rng.shuffle(targets)
            targets = targets[:1024] if split == 'train' else targets
            target_ids = tokenizer.texts_to_token_ids([x['text'] for x in targets])
            for reference_number,row in enumerate(candidates[:count]):
                # Keep the initial three held-out references per language;
                # inspect further intact references only until natural760 exists.
                if split=='test' and reference_number>=3 and any(c['total_frames']==760 for c in legal['test']):
                    break
                ref_path = folder / 'references' / f"{split}-{row['id']}-vad4s.wav"
                record_path = ref_path.with_suffix('.json')
                if record_path.is_file() and ref_path.is_file():
                    record = json.loads(record_path.read_text())
                    if sha(row['audio']) != record['source_sha256'] or sha(ref_path) != record['reference_sha256']:
                        raise RuntimeError('Reference provenance changed')
                else:
                    wave, sr = sf.read(row['audio'], dtype='float32', always_2d=True)
                    mono = torch.from_numpy(wave.mean(axis=1))
                    w16 = torchaudio.functional.resample(mono, sr, 16000)
                    w24 = torchaudio.functional.resample(mono, sr, 24000)
                    spans = get_speech_timestamps(w16, vad, sampling_rate=16000, speech_pad_ms=30)
                    if not spans or len(w16) < 64000:
                        raise RuntimeError('No continuous 4s reference: ' + row['id'])
                    maximum = len(w16)-64000
                    starts = {0, maximum}
                    for span in spans:
                        starts.update((max(0, min(maximum, span['start'])),
                                       max(0, min(maximum, span['end']-64000))))
                    def score(start):
                        coverage = sum(max(0, min(start+64000,s['end'])-max(start,s['start'])) for s in spans)
                        return coverage, -min(abs(start+64000-s['end']) for s in spans), -start
                    start = max(sorted(starts), key=score)
                    clip = w24[min(round(start*1.5),len(w24)-96000):][:96000].contiguous()
                    assert len(clip) == 96000
                    ref_path.parent.mkdir(parents=True, exist_ok=True)
                    sf.write(ref_path, clip.numpy(), 24000, subtype='FLOAT')
                    clip16 = torchaudio.functional.resample(clip, 24000, 16000)
                    if lang == 'zh':
                        text = asr_zh.generate(input=clip16.numpy(), use_itn=True,
                            disable_pbar=True)[0]['text'].replace(' ', '').strip()
                    else:
                        inputs = processor(clip16.numpy(), sampling_rate=16000, return_tensors='pt',return_attention_mask=True)
                        with torch.inference_mode():
                            predicted = asr_en.generate(inputs.input_features,attention_mask=inputs.attention_mask, language='en', task='transcribe')
                        text = processor.batch_decode(predicted, skip_special_tokens=True)[0].strip()
                    if not text:
                        raise RuntimeError('Empty 4s transcription')
                    record = dict(source_id=row['id'], source_audio=row['audio'],
                                  source_sha256=sha(row['audio']), reference_wav=str(ref_path),
                                  reference_sha256=sha(ref_path), reference_text=text,
                                  language=lang, split=split, start_seconds=start/16000,
                                  samples=96000, sample_rate=24000, vad_spans=spans)
                    write(record_path, record)
                references.append(record)
                prompt_ids = tokenizer.texts_to_token_ids([record['reference_text']])[0]
                if not prompt_ids:
                    continue
                for target, ids in zip(targets, target_ids):
                    if target['id'] == row['id']:
                        continue
                    frames = int(np.ceil(np.float32(375)*np.float32(len(ids))/np.float32(len(prompt_ids))))
                    total, length = 375+frames, len(ids)+len(prompt_ids)+1
                    if not 600 <= total <= 920 or not 52 <= length <= 141:
                        excluded[split] += 1
                        continue
                    case = dict(**record, target_id=target['id'], text=target['text'],
                                total_frames=total, padded_tokens=length)
                    legal[split].append(case)
                print(json.dumps({'event':'reference_complete','split':split,'language':lang,
                    'id':row['id'],'legal':len(legal[split])}), flush=True)
    calibration = []
    for lang in ('zh', 'en'):
        pool = [x for x in legal['train'] if x['language'] == lang]
        if len(pool) < 64:
            raise RuntimeError('Insufficient bilingual calibration conditions')
        pool.sort(key=lambda x:x['total_frames'])
        # Evenly cover the legal short/medium/long natural-duration distribution.
        calibration.extend(pool[round(i*(len(pool)-1)/63)] for i in range(64))
    quality = []
    quality_references=[]
    for lang in ('zh','en'):
        quality_references.extend([x for x in references if x['split']=='test' and x['language']==lang][:3])
    for ref in quality_references:
        pool = [x for x in legal['test'] if x['source_id']==ref['source_id']]
        if not pool:
            continue
        for case in (min(pool,key=lambda x:x['total_frames']),
                     min(pool,key=lambda x:abs(x['total_frames']-760)),
                     max(pool,key=lambda x:x['total_frames'])):
            if case not in quality:
                quality.append(case)
    exact = [x for x in legal['test'] if x['total_frames']==760]
    exact.sort(key=lambda x:abs(x['padded_tokens']-78))
    if exact and exact[0] not in quality:
        quality.append(exact[0])
    def materialize(cases, name):
        for i, case in enumerate(cases):
            condition = folder / name / f'{i:03d}.safetensors'
            w = prepare(ROOT/'models/zipvoice', manifest, case['reference_wav'],
                        case['reference_text'], case['text'], condition)
            case.update(condition=str(condition), condition_sha256=sha(condition), workload=w)
    materialize(calibration, 'calibration')
    materialize(quality, 'quality')
    train_ids = {x['source_id'] for x in calibration} | {x['target_id'] for x in calibration}
    test_ids = {x['source_id'] for x in quality} | {x['target_id'] for x in quality}
    assert not train_ids & test_ids
    hashes = {sha(x['audio']) for x in all_rows['train'] if x['id'] in train_ids}
    test_hashes = {sha(x['audio']) for x in all_rows['test'] if x['id'] in test_ids}
    assert not hashes & test_hashes
    result = dict(status='prepared', seed=args.seed, calibration=calibration, quality=quality,
                  primary_760=next((x for x in quality if exact and x==exact[0]), None),
                  legal_counts={k:len(v) for k,v in legal.items()}, excluded_counts=excluded,
                  reference_records=references, split_disjoint=True, audio_hashes_disjoint=True,
                  source_manifests={s:sha(args.data/'manifests'/f'{s}.jsonl.gz') for s in all_rows})
    write(folder/'manifest.json', result)
    print(json.dumps({'event':'data_ready','calibration':len(calibration),'quality':len(quality),
                      'natural_exact760':bool(result['primary_760'])}), flush=True)


if __name__ == '__main__':
    main()
