# Implementation goal and evidence

Started: 2026-09-22 15:32:34 UTC (23:32:34 Asia/Shanghai).
Six-hour target: 2026-09-22 21:32:34 UTC. Completion takes priority over this target.

## Stage 1 — independent SM89 B1/B4 TensorRT 11.3

- [x] Preserve original local B8 source snapshot: commit `813eb3d`.
- [x] Parameterize CFM/Vocoder runtime, builders and validators; verify engine I/O.
- [x] Build independent B1/B4 acoustic engines, retaining Target/Draft and B8.
- [x] Add trustworthy graph replay routing evidence and cache/shape safety checks.
- [x] Execute direct/graph numerical audits and verify real first-chunk routing on GPU 6 (numerical failures retained).
- [x] Publish original snapshot and B1/B4 implementation to inspark_proper_format (verified remote `34e0443`).

Coverage matches existing B8: Target verify8/K128, Draft backbone7/K128,
CFM prompt258/F310/two steps, Vocoder F52 with in-engine plugins. Prefill,
latent, proposal RNN, sampling and scheduling remain outside these engines;
tails and unsupported shapes use explicit fallback. Stage 1 completion is an
intermediate milestone, not completion of the overall goal.

## Stage 2 — architecture, correctness and full audit

- [x] Migrate to inference/ and models/ops/quantization/runtime/api/guardrails.
- [x] Preserve CLI, model semantics, resource identities and packaging.
- [x] Establish true same-weight FP32/BF16 eager and torch.compile baselines (failed numerical trials retained).
- [x] Fix request RNG/cache/buffer isolation and test cancellation/reuse/fallback (long soak remains separate).
- [x] Run per-stage numerical and 256-case quality audits with honest statuses (numerical failed; relative quality gates passed).
- [x] Measure operators and end-to-end paths on GPU 6 (experimental results; numerical failures retained).
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
- First publish verified remote main `34e0443`; implementation then moved to
  `inference/`, retaining only thin public compatibility exports. Wheel resource
  checks include CUDA source/attribution and exclude model/engine artifacts.
- CPU lifecycle tests now cover pool alias double-release, failed prefill
  rollback, cancel/close best-effort cleanup and unchanged per-request RNG.
  Safe TRT configs disable shared device/batched RNG instead of silently
  redefining request seed semantics. Historical fast profiles remain explicit.
- Pure FP32/BF16 eager full-EOS B1 smoke succeeded. Three-wave short-sentence
  first-PCM medians: 289.153/252.073 ms; complete-EOS medians: 641.390/741.541 ms.
  These are preliminary shared-device baselines, not matched-token speedups.
- Initial Inductor B1 trial ran full EOS, but Target compilation failed at a
  Transformers deprecation logger; Draft/CFM numerical gates failed and Vocoder
  passed. Retained the report; a local cache-format adapter and explicit eager
  precision-cast preservation are under re-audit, without tolerance changes.
- First real B1 full-EOS acoustic capture proved TRT head routes and bitwise
  direct/graph output equality. True-input head CFM/Vocoder comparisons still
  fail strict BF16/FP32 audits. BF16 eager tail replay matches the deployed tail;
  BF16-vs-FP32 arithmetic differences are reported separately.
- B1/B4 acoustic engines were rebuilt with actual checkpoint/ONNX/engine
  provenance, preserving original files. New source evidence does not upgrade
  old engine provenance or numerical status.
- Pure B4 FP32/BF16 eager and Draft/Vocoder-only compile baselines completed.
  The restricted B1 compile trial passed its sampled numerical gates and kept
  the BF16 eager token sequence for the measured input. B4 did not pass; both
  component failures remain public. This is not general compile certification.
- Request-local GPU RNG checks, 54 controlled same-input acceptance/prefix
  cases and four compact-cache boundary tests passed. Host-scheduled Target
  native dispatch now synchronizes the bounded canonical cache; legacy device
  entry also refreshes the selected batch mirror without changing RNG draws.
  CPU regressions cover batch switches, cancellation/reuse and native fallback.
- B1/B4 ten-second functional soaks passed with actual native Target calls,
  complete EOS, long KV and drained ownership checks. These short runs use the
  original runtime config and do not qualify as ten-minute stability evidence;
  the formal run selects the TF32-disabled reference runtime explicitly.
- Source-attested B1/B4 actual trajectories and the retained legacy B8 trajectory
  completed four independent FP32/BF16 AR/acoustic replays each. All overall
  numerical gates failed; component-level passes and exact BF16 tail controls
  are retained. B8 engine checkpoint provenance remains unverified.
- Request-isolated B1/B4 timing (two warmups, five complete-EOS waves): first-PCM
  medians 54.991/106.151 ms, full-EOS medians 330.840/839.198 ms. Different token
  trajectories prevent treating these as matched-work eager speedup ratios.
- Latest CPU regression: 211 passed, four GPU-only tests skipped there and
  passed separately. The 256-case B1 quality corpus is complete; B4, legacy B8,
  BF16 eager generation and independent CPU quality scoring are in progress.
- All five 256-case complete-EOS generation arms finished: FP32/BF16 eager B1,
  request-isolated TRT B1/B4 and original shared-RNG TRT B8. Paired B1/B4/B8
  quality gates passed; the BF16 eager control is still being evaluated. These
  automatic quality results do not waive the failed floating-point audits.
- 2026-09-22 17:48 UTC: started formal TF32-disabled 600-second concurrency
  tiers 1/4/8/16, with maximum model microbatch 8, on shared physical GPU6.
  CPU-only quality evaluation overlaps the initial tier and is recorded.
- Final packaging review found an unused historical audiotools/protobuf
  dependency conflict. Removed that unused dependency and explicitly declared
  the model utilities' existing Matplotlib/SciPy imports. The combined primary
  inference/ONNX build dependencies resolved as 100 packages on the Tsinghua
  mirror; no running environment was installed into or modified. The wheel
  builds offline, includes CUDA resources/notices and excludes model artifacts.
- 2026-09-22 17:54 UTC: all four paired quality evaluations completed with
  exactly 256 full-EOS pairs and no CUDA initialization in the evaluator.
  Relative quality gates passed for isolated TRT B1/B4, original TRT B8 and
  BF16 eager B1 against FP32 eager. Baseline UTMOS 1.643671675 / CER 144/4093;
  these are relative automatic gates, not high subjective quality or numerical
  equivalence. The original B8 engine provenance remains unverified.
- 2026-09-22 18:10 UTC: the first formal soak passed B1 (489 EOS, 48 cancelled)
  but failed B4's absolute RSS budget: +287.71 MiB versus the fixed 256 MiB
  limit, after 866 EOS and 86 cancellations. CUDA live growth was zero; the
  actual late RSS slope passed at 1.106 MiB/min. Original failure evidence is
  preserved. A diagnostic label incorrectly coupled absolute and trend checks;
  it is now separated without changing the overall gate or any threshold.
  Full CPU regression after that diagnostic-only change: 212 passed, four GPU
  tests skipped. Independent 8/16 tiers are running; a clearly separate warmed
  B4 control is planned to test the observed early-growth/late-plateau behavior.
- 2026-09-22 18:35 UTC: independent 8- and 16-concurrency ten-minute tiers
  passed without OOM: 1,164/1,237 EOS completions and 116/123 cancellations,
  respectively. Both use maximum B8 execution and the request-isolated legacy
  B8 engine profile; they do not certify the original shared-RNG profile or
  legacy engine checkpoint provenance. B4's separate 128-wave warmup control
  has started, with unchanged 600-second measurement and memory thresholds.
