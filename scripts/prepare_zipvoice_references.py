"""Select VAD-informed four-second reference windows and transcribe locally.

Run in the evaluation environment with CUDA_VISIBLE_DEVICES empty.
VAD and ASR operate on CPU; original user audio/text files remain untouched.
"""
import hashlib
import json
import os
from pathlib import Path
import random
import re

import numpy as np
import soundfile as sf
import torch
import torchaudio
from funasr import AutoModel
from silero_vad import get_speech_timestamps, load_silero_vad

ROOT = Path(__file__).resolve().parents[1] / 'outputs/zipvoice-validation'
ROOT.mkdir(parents=True, exist_ok=True)
REFERENCES = Path('/workspace/index-tts/data/audio/babckup')
TARGETS = Path('/workspace/index-tts/data/A_0810_pureneutral.txt')
ASR = Path(__file__).resolve().parents[1] / 'models/evaluation/paraformer-zh'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('This preparation pass requires CPU-only visibility')
    torch.set_num_threads(2)
    output = ROOT / 'cases'
    output.mkdir(exist_ok=True)
    vad = load_silero_vad()
    asr = AutoModel(model=str(ASR), device='cpu', disable_update=True,
                    ncpu=2, disable_pbar=True)
    records = []
    for source in sorted(REFERENCES.glob('*.wav')):
        wave, sr = sf.read(source, dtype='float32', always_2d=True)
        mono = torch.from_numpy(wave.mean(axis=1)).unsqueeze(0)
        mono16 = torchaudio.functional.resample(mono, sr, 16000).squeeze(0)
        mono24 = torchaudio.functional.resample(mono, sr, 24000).squeeze(0)
        spans = get_speech_timestamps(mono16, vad, sampling_rate=16000,
                                     threshold=.5, speech_pad_ms=30)
        if not spans or len(mono16) < 64000:
            raise RuntimeError(f'No usable four-second VAD window: {source}')
        max_start = len(mono16) - 64000
        starts = {0, max_start}
        for span in spans:
            starts.add(max(0, min(max_start, span['start'])))
            starts.add(max(0, min(max_start, span['end'] - 64000)))
        def coverage(start):
            return sum(max(0, min(start + 64000, s['end']) - max(start, s['start']))
                       for s in spans)
        # Retain original time order/pauses; do not concatenate speech regions.
        def window_score(start):
            # For equally voiced windows prefer an ending at a VAD boundary,
            # so the ASR prompt is less likely to end midway through a word.
            end_distance = min(abs(start + 64000 - s['end']) for s in spans)
            return coverage(start), -end_distance, -start
        start16 = max(sorted(starts), key=window_score)
        start24 = min(round(start16 * 1.5), len(mono24) - 96000)
        clip = mono24[start24:start24 + 96000].contiguous()
        assert len(clip) == 96000
        path = output / (source.stem + '-vad4s.wav')
        sf.write(path, clip.numpy(), 24000, subtype='FLOAT')
        clip16 = torchaudio.functional.resample(clip.unsqueeze(0), 24000, 16000).squeeze(0)
        result = asr.generate(input=clip16.numpy(), hotword='', use_itn=True,
                              disable_pbar=True)
        text = result[0]['text'].replace(' ', '').strip()
        if not text:
            raise RuntimeError(f'Empty ASR transcript: {source}')
        record = {'source': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                  'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                  'reference_text': text, 'sample_rate': 24000, 'samples': 96000,
                  'duration_seconds': 4, 'start_seconds': start24 / 24000,
                  'vad_speech_coverage': coverage(start16) / 64000,
                  'vad_spans_16k': spans, 'asr_model': str(ASR)}
        records.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
    (output / 'reference-vad4s.json').write_text(json.dumps(records, ensure_ascii=False, indent=2) + '\n')
    targets = []
    for number, line in enumerate(TARGETS.read_text().splitlines(), 1):
        match = re.search(r'文本："(.*?)"', line)
        if match:
            targets.append({'line': number, 'text': match.group(1)})
    sample = random.Random(9100).sample(targets, min(70, len(targets)))
    (output / 'sample70.json').write_text(json.dumps({'source': str(TARGETS),
        'source_sha256': hashlib.sha256(TARGETS.read_bytes()).hexdigest(),
        'seed': 9100, 'count': len(sample), 'samples': sample}, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
