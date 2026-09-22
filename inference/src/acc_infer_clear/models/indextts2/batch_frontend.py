"""Parallel CPU text preparation and batched GPU condition/prefix assembly.

Only reference-derived tensors are cached across requests. Text tokens, emotion
weights, mixed emotions and prepared prefixes exist for one incoming batch only.
Learned text/mel positions and original per-row special-token filtering remain.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import time

@dataclass
class PreparedSegment:
    text_ids: object
    prefix: object
    mask: object
    speech_latent: object

@dataclass
class PreparedRequest:
    text: str
    reference_path: str
    emotion: list
    tokens: list
    segments: list
    all_ids: list
    prepared_segments: list
    emovec: object
    max_segment_tokens: int

class BatchFrontend:

    def __init__(self, model, workers=8):
        self.model = model
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='tts-text')
        self.calls = 0
        self.last_stats = {}
        self.cached = {}

    def close(self):
        self.pool.shutdown(wait=True)

    def prime_voices(self):
        import torch
        from acc_infer_clear.models.indextts2.upstream.infer_v2 import find_most_similar_cosine
        m = self.model
        g = m.tts.gpt
        if g.spk_cond_mode == 'campplus':
            raise ValueError('Batch frontend currently requires conformer/perceiver voice conditions')
        m._acquire('priming_batch_frontend_voices')
        try:
            with torch.inference_mode(), torch.cuda.stream(m.stream):
                self.cached={key:value for key,value in self.cached.items() if key in m.bank.entries}
                for key, entry in m.bank.entries.items():
                    # VAD paths are content-addressed; reference models stay unquantized.
                    if key in self.cached and self.cached[key]['path']==entry['path']:continue
                    v = entry['values']
                    spk = v['voice.cache_spk_cond']
                    emo = v['voice.cache_emo_cond']
                    style = v['voice.cache_s2mel_style']
                    sl = torch.tensor([spk.shape[-1]], device=spk.device)
                    el = torch.tensor([emo.shape[-1]], device=emo.device)
                    speech = m.bank.original_conditioning(spk.transpose(1, 2), sl)
                    base = g.get_emovec(spk, sl)
                    emotion = g.get_emovec(emo, el)
                    indices = [find_most_similar_cosine(style, x) for x in m.tts.spk_matrix]
                    basis = torch.cat([x[i].unsqueeze(0) for x, i in zip(m.tts.emo_matrix, indices)], 0)
                    assert speech.shape == (1, 32, 1280) and basis.shape == (8, 1280)
                    self.cached[key] = dict(path=entry['path'], speech=speech.detach(), base=base.detach(), emotion=emotion.detach(), basis=basis.detach())
                m.stream.synchronize()
        finally:
            m.status = 'ready'
            m._release()

    def _text(self, request):
        tokenizer = self.model.tts.tokenizer
        text = request['text']
        tokens = tokenizer.tokenize(text)
        maximum = self.model.cfg['data']['max_text_tokens_per_segment']
        segments = tokenizer.split_segments(tokens, maximum, quick_streaming_tokens=0)
        if not segments:
            raise ValueError('Empty tokenized request')
        return dict(tokens=tokens, segments=segments, all_ids=tokenizer.convert_tokens_to_ids(tokens), ids=[tokenizer.convert_tokens_to_ids(s) for s in segments])

    def prepare(self, requests):
        import torch
        import math
        for r in requests:
            if not isinstance(r['text'], str) or not r['text'].strip():
                raise ValueError('Empty text')
            if len(r['emotion']) != 8 or any((not math.isfinite(float(v)) or not 0 <= float(v) <= 1 for v in r['emotion'])):
                raise ValueError('Invalid emotion vector')
            if self.cached[r['voice_id']]['path'] != self.model.bank.get(r['voice_id'])['path']:
                raise ValueError('Reference changed; re-prime voice-only frontend cache')
        start = time.perf_counter()
        texts = list(self.pool.map(self._text, requests))
        cpu_done = time.perf_counter()
        g = self.model.tts.gpt
        device = next(g.parameters()).device
        voices = [self.cached[r['voice_id']] for r in requests]
        weights = torch.tensor([r['emotion'] for r in requests], device=device, dtype=torch.float32)
        if weights.shape != (len(requests), 8):
            raise ValueError('Expected8 emotion weights per request')
        bases = torch.cat([v['base'] for v in voices], 0)
        reference = torch.cat([v['emotion'] for v in voices], 0)
        basis = torch.stack([v['basis'] for v in voices])
        merged_ref = bases + 1.0 * (reference - bases)
        mixed = (weights[:, :, None] * basis).sum(1) + (1 - weights.sum(1))[:, None] * merged_ref
        speech = torch.cat([v['speech'] for v in voices], 0)
        speed_half = g.speed_emb(torch.ones(len(requests), device=device, dtype=torch.long))
        speed_zero = g.speed_emb(torch.zeros(len(requests), device=device, dtype=torch.long))
        conditions = torch.cat((speech + mixed[:, None, :], speed_half[:, None, :], speed_zero[:, None, :]), 1)
        flat = [(i, j, ids) for i, t in enumerate(texts) for j, ids in enumerate(t['ids'])]
        clean = [[g.start_text_token, *[x for x in ids if x not in (g.start_text_token, g.stop_text_token)], g.stop_text_token] for _, _, ids in flat]
        raw_lengths = [len(ids) for _, _, ids in flat]
        text_lengths = [len(ids) for ids in clean]
        left_pad = [raw - (length - 2) for raw, length in zip(raw_lengths, text_lengths)]
        text_max = max(text_lengths)
        prefix_lengths = [34 + n + 3 for n in raw_lengths]
        prefix_max = max(prefix_lengths)
        ids = torch.tensor([x + [g.start_text_token] * (text_max - len(x)) for x in clean], device=device, dtype=torch.long)
        raw_max = max(raw_lengths)
        raw_ids = torch.tensor([x + [g.stop_text_token] * (raw_max - len(x)) for _, _, x in flat], device=device, dtype=torch.int32)
        positions = torch.arange(text_max, device=device)
        text_emb = g.text_embedding(ids) + g.text_pos_embedding.emb(positions)[None]
        owner = torch.tensor([i for i, _, _ in flat], device=device, dtype=torch.long)
        cond = conditions.index_select(0, owner)
        lp = torch.tensor(left_pad, device=device)[:, None]
        tl = torch.tensor(text_lengths, device=device)[:, None]
        pl = torch.tensor(prefix_lengths, device=device)[:, None]
        pos = torch.arange(prefix_max, device=device)[None].expand(len(flat), -1)
        ci = pos - lp
        ti = pos - lp - 34
        cg = cond.gather(1, ci.clamp(0, 33)[:, :, None].expand(-1, -1, 1280))
        tg = text_emb.gather(1, ti.clamp(0, text_max - 1)[:, :, None].expand(-1, -1, 1280))
        prefix = torch.where(((ci >= 0) & (ci < 34))[:, :, None], cg, 0.0)
        prefix = torch.where(((ti >= 0) & (ti < tl))[:, :, None], tg, prefix)
        bos = g.mel_embedding(torch.tensor([g.start_mel_token], device=device)) + g.mel_pos_embedding.emb(torch.tensor([0], device=device))
        prefix = torch.where((pos == pl - 1)[:, :, None], bos[None], prefix)
        mask = ((pos >= lp) & (pos < pl)).long()
        result = {}
        for i, (r, t, v) in enumerate(zip(requests, texts, voices)):
            result[r['id']] = PreparedRequest(r['text'], v['path'], list(r['emotion']), t['tokens'], t['segments'], t['all_ids'], [], mixed[i:i + 1], self.model.cfg['data']['max_text_tokens_per_segment'])
        for row, (i, segment, _) in enumerate(flat):
            n = prefix_lengths[row]
            result[requests[i]['id']].prepared_segments.append(PreparedSegment(raw_ids[row:row + 1, :raw_lengths[row]], prefix[row:row + 1, :n], mask[row:row + 1, :n], speech[i:i + 1]))
        self.calls += 1
        self.last_stats = dict(cpu_text_ms=(cpu_done - start) * 1000, gpu_host_submit_ms=(time.perf_counter() - cpu_done) * 1000, requests=len(requests), segments=len(flat), prefix_lengths=prefix_lengths, packed_shape=list(prefix.shape))
        return result

    def validate(self, requests, prepared):
        import torch
        from acc_infer_clear.models.indextts2.upstream.infer_v2 import find_most_similar_cosine
        from acc_infer_clear.runtime.state import request_scope
        errors = []
        m = self.model
        g = m.tts.gpt
        with torch.inference_mode():
            for r in requests:
                p = prepared[r['id']]
                entry = m.bank.get(r['voice_id'])
                v = entry['values']
                serial = self._text(r)
                assert serial['tokens'] == p.tokens and serial['segments'] == p.segments
                with request_scope(r['id'], v):
                    spk = v['voice.cache_spk_cond']
                    emo = v['voice.cache_emo_cond']
                    style = v['voice.cache_s2mel_style']
                    sl = torch.tensor([spk.shape[-1]], device=spk.device)
                    el = torch.tensor([emo.shape[-1]], device=emo.device)
                    w = torch.tensor(r['emotion'], device=spk.device)
                    indices = [find_most_similar_cosine(style, x) for x in m.tts.spk_matrix]
                    basis = torch.cat([x[i].unsqueeze(0) for x, i in zip(m.tts.emo_matrix, indices)], 0)
                    e = (w[:, None] * basis).sum(0)[None] + (1 - w.sum()) * g.merge_emovec(spk, emo, sl, el, alpha=1.0)
                    torch.testing.assert_close(p.emovec, e, atol=1e-06, rtol=1e-06)
                    for segment in p.prepared_segments:
                        ids, mask, speech = m.engine.target._prepare_inputs(spk, segment.text_ids, emo_speech_condition=emo, cond_lengths=sl, emo_cond_lengths=el, emo_vec=e)
                        expected = m.engine.target._prefill_embeddings(ids)
                        torch.testing.assert_close(segment.prefix, expected, atol=1e-05, rtol=1e-05)
                        torch.testing.assert_close(segment.mask, mask, atol=0, rtol=0)
                        errors.append(dict(request=r['id'], prefix_max_abs=float((expected - segment.prefix).abs().max()), emotion_max_abs=float((p.emovec - e).abs().max())))
        return errors
