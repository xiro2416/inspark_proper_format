"""Input-boundary VAD and bounded, immutable voice-only feature cache."""
from __future__ import annotations
import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from inspark_infer.runtime.config import atomic_json
from inspark_infer.runtime.assets import sha256
FIELDS = ('cache_spk_cond', 'cache_s2mel_style', 'cache_s2mel_prompt', 'cache_spk_audio_prompt', 'cache_mel', 'cache_emo_cond', 'cache_emo_audio_prompt')

def vad_crop(source, cache_dir, seconds=3):
    import torch
    import torchaudio
    import soundfile as sf
    import numpy as np
    import silero_vad
    from silero_vad import load_silero_vad, get_speech_timestamps
    source = Path(source).resolve()
    if seconds not in (1, 2, 3, 5):
        raise ValueError('VAD duration must be 1,2,3,5 seconds')
    model_file = Path(silero_vad.__file__).parent / 'data/silero_vad.jit'
    settings = dict(source_sha256=sha256(source), model_sha256=sha256(model_file), seconds=seconds, threshold=0.5, pad_ms=30, min_speech_ms=250, min_silence_ms=100, policy='ordered_voiced_concat_prefix_v1')
    key = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    directory = Path(cache_dir) / 'reference' / key
    directory.mkdir(parents=True, exist_ok=True)
    output, meta = (directory / 'prompt.wav', directory / 'metadata.json')
    if output.is_file() and meta.is_file():
        data = json.loads(meta.read_text())
        if data['settings'] == settings and data['output_sha256'] == sha256(output):
            return (output, dict(data, cache_hit=True))
    prior_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        audio, sr = sf.read(source, dtype='float32', always_2d=True)
        mono = audio.mean(1)
        if not len(mono) or not np.isfinite(mono).all():
            raise ValueError('Empty or non-finite reference audio')
        waveform = torchaudio.functional.resample(torch.from_numpy(mono), sr, 16000)
        spans = get_speech_timestamps(waveform, load_silero_vad(), sampling_rate=16000, threshold=0.5, min_speech_duration_ms=250, min_silence_duration_ms=100, speech_pad_ms=30)
        if not spans:
            raise ValueError('Reference has no VAD-detected speech')
        intervals = [(max(0, round(t['start'] * sr / 16000)), min(len(mono), round(t['end'] * sr / 16000))) for t in spans]
        voiced = np.concatenate([mono[a:b] for a, b in intervals])
        cropped = voiced[:seconds * sr]
        from tempfile import NamedTemporaryFile
        with NamedTemporaryFile(dir=directory, suffix='.wav', delete=False) as temp:
            temporary = Path(temp.name)
        sf.write(temporary, cropped, sr, subtype='FLOAT')
        temporary.replace(output)
        data = dict(settings=settings, original_seconds=len(mono) / sr, available_seconds=len(voiced) / sr, actual_seconds=len(cropped) / sr, shortfall=len(cropped) < seconds * sr, intervals_seconds=[[a / sr, b / sr] for a, b in intervals], output_sha256=sha256(output))
        atomic_json(meta, data)
        return (output, dict(data, cache_hit=False))
    finally:
        torch.set_num_threads(prior_threads)

class VoiceBank:

    def __init__(self, tts, limit=16):
        import torch
        self.tts, self.limit = (tts, int(limit))
        if self.limit < 1:
            raise ValueError('max_cached_voices must be positive')
        self.entries = OrderedDict()
        self.hits = self.misses = 0
        original = tts.gpt.get_conditioning

        def conditioning(x, lengths=None):
            from inspark_infer.runtime.state import CURRENT_REQUEST
            state = CURRENT_REQUEST.get()
            variants = state.values.get('voice.conditioning_variants') if state else None
            source = state.values.get('voice.cache_spk_cond') if state else None
            shape_matches = source is not None and tuple(x.shape) == (source.shape[0], source.shape[2], source.shape[1])
            if variants is not None and shape_matches and (x.data_ptr() == source.data_ptr()) and (x.dtype == source.dtype):
                signature = (tuple(lengths.detach().cpu().tolist()) if lengths is not None else None, tuple(x.stride()), x.dtype, torch.is_autocast_enabled('cuda'), torch.get_autocast_dtype('cuda'))
                if signature in variants:
                    self.hits += 1
                    return variants[signature]
                self.misses += 1
                value = original(x, lengths).detach()
                if len(variants) < 8:
                    variants[signature] = value
                return value
            self.misses += 1
            return original(x, lengths)
        tts.gpt.get_conditioning = conditioning
        self.original_conditioning = original

    def build(self, key, path, metadata):
        import torch
        import torchaudio
        tts = self.tts
        path = str(Path(path).resolve())
        with torch.inference_mode():
            audio, sr = tts._load_and_cut_audio(path, 15, False)
            audio_22 = torchaudio.transforms.Resample(sr, 22050)(audio)
            audio_16 = torchaudio.transforms.Resample(sr, 16000)(audio)

            def embeddings(wav):
                features = tts.extract_features(wav, sampling_rate=16000, return_tensors='pt')
                return tts.get_emb(features['input_features'].to(tts.device), features['attention_mask'].to(tts.device))
            spk = embeddings(audio_16)
            _, codes = tts.semantic_codec.quantize(spk)
            mel = tts.mel_fn(audio_22.to(spk.device).float())
            lengths = torch.LongTensor([mel.size(2)]).to(mel.device)
            feat = torchaudio.compliance.kaldi.fbank(audio_16.to(mel.device), num_mel_bins=80, dither=0, sample_frequency=16000)
            feat = feat - feat.mean(0, keepdim=True)
            style = tts.campplus_model(feat.unsqueeze(0))
            prompt = tts.s2mel.models['length_regulator'](codes, ylens=lengths, n_quantizers=3, f0=None)[0]
            emo_wav, _ = tts._load_and_cut_audio(path, 15, False, sr=16000)
            emo = embeddings(emo_wav)
            conditioning = self.original_conditioning(spk.transpose(1, 2), torch.tensor([spk.shape[1]], device=spk.device))
        fields = dict(zip(FIELDS, (spk, style, prompt, path, mel, emo, path)))
        values = {'voice.' + name: value for name, value in fields.items()}
        values['voice.conditioning'] = conditioning
        values['voice.conditioning_variants'] = {}
        self.entries[key] = dict(path=path, metadata=metadata, values=values)
        self.entries.move_to_end(key)
        while len(self.entries) > self.limit:
            self.entries.popitem(last=False)
        return self.entries[key]

    def get(self, key):
        entry = self.entries[key]
        self.entries.move_to_end(key)
        return entry

