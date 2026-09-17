from dataclasses import dataclass,field
import torch
import torch.nn.functional as F
from .target import crop_legacy_cache
from .logic import accepted_prefix

@dataclass(eq=False)
class BatchRow:
    request: dict
    prepared: object
    generator: object
    state: dict = field(default_factory=dict)
    codes: list = field(default_factory=list)
    chunks: list = field(default_factory=list)
    accepted: list = field(default_factory=list)
    kv: object = None
    cache: object = None
    mask: object = None
    prefix_length: int = 0
    mel_length: int = 0
    past_length: int = 0
    done: bool = False
    acoustic: object = None
    acoustic_state: object = None
    acoustic_pending_since: float = 0.0
    acoustic_done: bool = False
    finished: float = 0.0
    segment: int = 0

class _CapturedLatent(Exception):

    def __init__(self, args, kwargs):
        self.job = (args, kwargs)

class _LatentPreparation:
    """Run original input preparation without patching the shared GPT module."""

    def __init__(self, gpt):
        self.gpt = gpt

    def __getattr__(self, name):
        return getattr(self.gpt, name)

    def get_logits(self, *args, **kwargs):
        raise _CapturedLatent(args, kwargs)

class ARCore:

    def _step(self, rows, max_tokens):
        draft = self.engine.draft
        k = draft.block_size
        snapshots = None
        jobs = [dict(cache=r.cache, anchor_token=r.codes[-1], first_position=r.past_length + 1 - r.mel_length, temperature=0.8) for r in rows]
        with self.span('draft', rows):
            proposed = self.proposal(jobs, rows)
        if snapshots is not None:
            for row, expected in zip(rows, snapshots):
                torch.testing.assert_close(torch.cat([t.reshape(-1) for t in row.codes]), expected, atol=0, rtol=0)
        with self.span('verify', rows):
            tokens = torch.cat([torch.cat((r.codes[-1].reshape(1, 1), out[0]), 1) for r, out in zip(rows, proposed)])
            positions = torch.tensor([j['first_position'] for j in jobs], device=self.device)[:, None] + torch.arange(k + 1, device=self.device)[None]
            tm = self.engine.target.model
            embeds = tm.embeddings(tokens) + tm.text_pos_embedding.emb(positions)
            masks = [F.pad(r.mask, (0, k + 1), value=1) for r in rows]
            verified = self.target([(embeds[i:i + 1], r.kv, masks[i], None) for i, r in enumerate(rows)])
        fallback_before = self.residual.fallbacks
        with self.span('accept_commit', rows):
            accepted = self.accept([(v[0][0, :k], p[1][0], p[0][0]) for v, p in zip(verified, proposed)], rows)
            residual_rows, residual_jobs, decisions = ([], [], [])
            eos = int(self.engine.target.gpt.stop_mel_token)
            for row, p, v, a in zip(rows, proposed, verified, accepted):
                count = min(k, max_tokens - len(row.codes))
                n, end = accepted_prefix(a[2].rows, count, eos)
                row.codes.extend((p[0][:, j] for j in range(n)))
                row.generator.set_state(row.state['cuda_rng'])
                correction = not end and len(row.codes) < max_tokens
                decisions.append((n, end, correction))
                if correction and n < count:
                    residual_rows.append(row)
                    residual_jobs.append(((a[0][n], p[1][0, n], self.engine.dense_groups), dict(max_thinning_attempts=self.engine.max_thinning_attempts)))
                elif correction:
                    row.codes.append(self.sample(row, v[0][:, count]))
            if residual_rows:
                residuals = self.residual(residual_jobs, residual_rows)
                for row, out in zip(residual_rows, residuals):
                    row.codes.append(out[0].reshape(1))
                    row.generator.set_state(row.state['cuda_rng'])
            last = torch.cat([r.codes[-1] for r in rows]).cpu().tolist()
            context_jobs = []
            for i, (row, v, (n, end, correction), token) in enumerate(zip(rows, verified, decisions, last)):
                committed = n + 1
                row.accepted.append(n)
                row.past_length += committed
                row.kv = crop_legacy_cache(v[1], row.past_length)
                row.mask = masks[i][:, :row.past_length]
                context_jobs.append(((row.cache, v[2][:, :committed], v[3][:, :committed]), dict(committed_tokens=tokens[i:i + 1, :committed])))
                row.done = end or token == eos or len(row.codes) >= max_tokens
            self.context(context_jobs)

    def _latents(self, ops):
        jobs = []
        gpt = self.tts.gpt
        for op in ops:
            try:
                type(gpt).forward(_LatentPreparation(gpt), *op.args, **op.kwargs)
            except _CapturedLatent as captured:
                jobs.append(captured.job)
            else:
                raise RuntimeError('GPT latent preparation did not reach get_logits')
        return [x[1][:, :-2] for x in self.latent(jobs)]

