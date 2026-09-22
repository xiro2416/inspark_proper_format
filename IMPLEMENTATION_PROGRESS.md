# Implementation goal and evidence

Started: 2026-09-22 15:32:34 UTC (23:32:34 Asia/Shanghai).
Six-hour target: 2026-09-22 21:32:34 UTC. Completion takes priority over this target.

## Stage 1 — independent SM89 B1/B4 TensorRT 11.3

- [ ] Preserve original local B8 source snapshot.
- [ ] Parameterize CFM/Vocoder runtime, builders and validators; verify engine I/O.
- [ ] Build independent B1/B4 acoustic engines, retaining Target/Draft and B8.
- [ ] Add trustworthy graph replay routing evidence and cache/shape safety checks.
- [ ] Verify direct execution, graph replay and real first chunks on GPU 6.
- [ ] Publish original snapshot and B1/B4 implementation to inspark_proper_format.

Coverage matches existing B8: Target verify8/K128, Draft backbone7/K128,
CFM prompt258/F310/two steps, Vocoder F52 with in-engine plugins. Prefill,
latent, proposal RNN, sampling and scheduling remain outside these engines;
tails and unsupported shapes use explicit fallback. Stage 1 completion is an
intermediate milestone, not completion of the overall goal.

## Stage 2 — architecture, correctness and full audit

- [ ] Migrate to inference/ and models/ops/quantization/runtime/api/guardrails.
- [ ] Preserve CLI, model semantics, resource identities and packaging.
- [ ] Establish true same-weight FP32/BF16 eager and torch.compile baselines.
- [ ] Fix request RNG/cache/buffer isolation and test cancellation/reuse/fallback.
- [ ] Run per-stage numerical and 256-case quality audits with honest statuses.
- [ ] Measure operators and end-to-end paths on GPU 6.
- [ ] Run concurrency 1/4/8/16, ten minutes each; record and skip 16 on OOM.
- [ ] Publish SM89-specific reports and README; verify final remote commit.

FP32 tolerances: atol=1e-5, rtol=1e-4. BF16 arithmetic tolerances:
atol=1e-2, rtol=1e-2. Reduction-order differences within tolerance are allowed.
Discrete metadata must match under identical replay inputs. Missing metrics,
NaN/Inf, shape mismatches and unsupported claims cannot pass. Quality gates:
UTMOS relative decrease <=3%, CER absolute increase <=0.02 (not WER).

Use only physical GPU 6, with GPU tasks sequential. Preserve existing external
GPU processes. All task files/caches/locks stay within /workspace. Never publish
credentials, official weights, engine/ONNX binaries, environments or audio.

## Initial observations

Local and original remote base: 7802054. New target repository is empty.
11 CPU contract checks passed during planning; no real-model GPU audit yet.
Existing labels sometimes call custom-op paths eager; shared batch RNG and
Draft identity-slot/K128 checks need attention. Existing real-audio Vocoder A/B
was reproduced on CPU (16 pairs, mean cosine 0.999989578, mean SNR 58.3034 dB),
but this is not a complete same-model eager audit. Random Vocoder validation
has known numerical failures, retained as evidence.
