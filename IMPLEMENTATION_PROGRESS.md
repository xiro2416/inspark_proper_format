# Implementation goal and evidence

Started: 2026-09-22 15:32:34 UTC (23:32:34 Asia/Shanghai).
Six-hour target: 2026-09-22 21:32:34 UTC. Completion takes priority over this target.

## Stage 1 — independent SM89 B1/B4 TensorRT 11.3

- [x] Preserve original local B8 source snapshot: commit `813eb3d`.
- [x] Parameterize CFM/Vocoder runtime, builders and validators; verify engine I/O.
- [x] Build independent B1/B4 acoustic engines, retaining Target/Draft and B8.
- [x] Add trustworthy graph replay routing evidence and cache/shape safety checks.
- [x] Execute direct/graph numerical audits and verify real first-chunk routing on GPU 6 (numerical failures retained).
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

## Execution log

- Original B8 snapshot committed locally; no GitHub push before B1/B4 completion.
- GPU 6 has 21,854 MiB pre-existing allocations and was idle at preflight.
  Explicit shared-device mode preserves these processes and records this limit.
- B1 CFM (1x80x310) and native-convolution Vocoder (1x80x52) built with
  TensorRT 11.3.0.99, strongly typed, TF32 disabled. B4 builds are in progress.
- Added ONNX 1.23.0 and pytest 8.4.2 from the Tsinghua mirror; dependency changes
  include protobuf 7.36.2 and ml-dtypes 0.6.0, to be captured in reproducible setup.
- First CPU regression run: 17 passed. Cache-boundary tests added subsequently.
- 2026-09-22 15:49 UTC: 45 CPU tests passed; four GPU-only tests correctly skipped.
- B1 real first-chunk audit passed two warmups and five measured waves, with
  Target/Draft native counts matching all execution counts, CFM/Vocoder frozen
  TRT graph routes, and no runtime fallback. First-chunk P50 42.564 ms,
  P95 52.785 ms. This does not establish numerical parity, complete EOS or soak.
- Guarded compact K128 context scatter and device-loop boundaries; the canonical
  long-context pool remains intact. A GPU boundary regression test is queued.
- Legacy Target/Draft engine metadata does not attest model checkpoint identity.
  Builders are being extended to record actual checkpoint and converted-constant
  hashes; the Target builder must explicitly disable TF32 before new audits.
- Request-local RNG repair requires an explicit legacy-seed compatibility choice;
  asked the user asynchronously. No new RNG semantics have been enabled yet.
- 2026-09-22 16:08 UTC: 73 CPU tests passed; all four GPU boundary tests passed
  separately on GPU6. B1/B4 source-attested Target and Draft engines were rebuilt
  with TF32 disabled; original engine files are retained.
- B4 initial real first-chunk audit also passed: five measured four-request waves,
  all four TRT routes, no fallback; all-ready P50 74.236 ms. Final routing checks
  for rebuilt AR engines and retained B8 are queued.
- Diagnostic numerical results are NOT an overall pass: B1 CFM passes; B4 CFM
  has 19 native-vs-BF16 and 3 BF16-vs-FP32 out-of-tolerance elements. Random-mel
  Vocoder B1/B4 fails; these inputs are not a perceptual quality distribution.
  Source-attested AR B1/B4 also fails strict numerical comparisons, while the
  measured cache preservation and native-direct/graph invariants pass. Keep the
  candidate experimental; publish failures without relaxing tolerances.
- Final stage-one routing with rebuilt AR engines passed B1/B4; the retained B8
  regression also passed. Five-wave all-ready P50: 41.554/74.262/115.784 ms for
  B1/B4/B8. One B8 preflight-busy attempt is retained; the next idle check and run
  succeeded without touching other GPU processes.
- Stage-one code committed as `d646331`; full reports are under
  `reports/sm89/trt113_b1_b4_stage1/`. Overall numerical parity is FAILED, not
  implied by successful routing. Stage two remains mandatory.
