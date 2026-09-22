# SM89 stage-two evidence

This directory archives completed RTX 4090 / SM89 observations. It is separate
from historical SM120 results and from the
[stage-one build and preliminary audits](../trt113_b1_b4_stage1/README.md).
The files are byte-for-byte copies of existing local JSON/JSONL reports and
original test, build, dependency-resolution and failure logs, not regenerated
results: [COPY_MANIFEST.json](COPY_MANIFEST.json) records their source paths,
SHA256 hashes and sizes. Original absolute paths, source identities, errors and
provenance fields are retained without alteration.

The archive paths have distinct scopes:

- `baselines/`: same-model B1/B4 eager/compile execution, sampled compile-versus-eager comparisons, and separately sampled TRT timings.
- `legacy_real_acoustic_b1/`: one completed real trajectory captured with the old acoustic engines, plus frozen-input FP32/BF16 replays.
- `builds/{cfm,vocoder}/`: new B1/B4 export, engine-build and plan metadata, kept separate from the legacy capture.
- `real_safe_b{1,4}_audited/`: source-attested real AR/acoustic captures and their independent FP32/BF16 replays, including failures.
- `real_legacy_b8/`: original shared-device-RNG B8 capture/replays, with unverified legacy engine origins.
- `runtime/`: controlled-input CUDA RNG/acceptance checks, functional smoke, formal soak attempts including failures, regression logs and the original GPU-busy rejection.
- `quality/`: paired complete-EOS quality reports and the original per-arm generation summaries/JSONL; generated audio is not published.
- `environment/`: dependency-resolution inputs/output and packaging/CLI checks; no fresh-environment GPU certification is implied.

No tensors, audio, ONNX graphs, engines or checkpoints are included. References
to local `.pt` files in the JSON identify withheld replay evidence by hash; this
public report archive is not a standalone executable replay bundle.

## Eager and compile baselines

Each run used one warmup and three measured complete-EOS waves, the listed batch,
text `他正在整理文件。`, and a shared GPU with 21,854 MiB external residency.
The timing includes host admission, copies, allocation, launches and first-PCM /
complete-EOS work; model/reference preparation is excluded. See each JSON's
`summary`, `operator_audit`, `routes` and `compile` counters.

| Run | All first PCM P50, ms | Full EOS P50, ms | Numerical conclusion |
| --- | ---: | ---: | --- |
| [B1 FP32 eager](baselines/eager_fp32_b1.json) | 289.15 | 641.39 | Execution passed; no cross-precision numerical gate was run (`null`) |
| [B1 BF16 eager](baselines/eager_bf16_b1.json) | 252.07 | 741.54 | Execution passed; no cross-precision numerical gate was run (`null`) |
| [B1 initial compile](baselines/compile_bf16_b1.json) | 213.99 | 833.41 | Failed; Target fell back after a logging-related fullgraph error; Draft/CFM comparisons failed |
| [B1 compile with precision casts](baselines/compile_bf16_b1_casts.json) | 249.78 | 719.43 | All four compiled; all four sampled numerical gates failed |
| [B1 compile without pattern rewrite](baselines/compile_bf16_b1_no_rewrite.json) | 293.54 | 823.38 | Draft/Vocoder sampled gates passed; Target/CFM failed; overall failed |
| [B1 Draft/Vocoder-only compile](baselines/compile_bf16_b1_subset.json) | 185.94 | 574.14 | Sampled Draft/Vocoder gate passed for this B1 input only; Target/CFM remained eager |
| [B4 FP32 eager](baselines/eager_fp32_b4.json) | 413.69 | 1,026.89 | Execution passed; no cross-precision numerical gate was run (`null`) |
| [B4 BF16 eager](baselines/eager_bf16_b4.json) | 542.86 | 1,485.64 | Execution passed; no cross-precision numerical gate was run (`null`) |
| [B4 Draft/Vocoder-only compile](baselines/compile_bf16_draft_vocoder_b4.json) | 514.42 | 1,454.72 | Draft and Vocoder both failed; overall failed |

For the B1 subset trial, all three measured token-sequence SHA256 values match
the BF16 eager B1 run (`dc12c6a8bda4d8615ec2303e01a17e1dfa05e18230669fdd526a433069943221`). That observation is limited to this
text, reference, seed and batch; it is not bitwise waveform equivalence or a
corpus-quality guarantee. Draft records 4 compiled / 96 eager calls and Vocoder
4 / 4, so this is still mixed execution. The B4 subset trial fails with 532 Draft
and 1,070 Vocoder mismatched elements. Independently switchable components do
not imply that a B1 pass generalizes to B4. The B4 full-EOS difference
(1,454.72 vs 1,485.64 ms) is not evidence of stable acceleration, especially with
failed numerical gates and only three measured shared-GPU waves.

These are experimental observations, not a speedup ranking or acceptance of a
numerically invalid backend. AR trajectories and generated lengths differ across
the runs. Consequently the real component inputs also differ, and mismatch
counts must not be compared across trials as error-rate improvements on the
same dataset. Within each reported operator comparison, compiled and eager
receive the same boundary inputs. The unchanged BF16 gate is
`abs(actual-reference) <= 0.01 + 0.01 * abs(reference)`.

Only one signature per component was warmed in these trials. In the no-rewrite
run, Target/Draft each record 4 compiled calls and 120 eager calls; CFM records
8/8, Vocoder 4/4. These are mixed compiled/eager end-to-end routes, not wholly
compiled inference. Compilation/audit overhead is reported separately. The
[compiler investigation](../../../inference/docs/compile_numerics.md) documents
the BF16 bias-rounding counterexample and why successful compilation alone does
not establish parity.

## Request-isolated TensorRT timing observations

These two runs used **two warmups and five measured waves**, not the one-plus-three
sampling of the eager/compile table above. They used `runtime_reference.yaml`
(TF32 disabled), the same short text, complete-EOS execution, and the same recorded
21,854 MiB external GPU residency. Preparation is excluded; the report's
`timing_scope` defines the measured host admission/copy/allocation/launch and
execution work.

| Run | All first PCM P50, ms | Full EOS P50, ms | Timing-run conclusion |
| --- | ---: | ---: | --- |
| [B1 safe TRT](baselines/trt113_safe_b1.json) | 54.99 | 330.84 | Execution passed; timing report does not run a numerical gate (`null`) |
| [B4 safe TRT](baselines/trt113_safe_b4.json) | 106.15 | 839.20 | Execution passed; timing report does not run a numerical gate (`null`) |

The independent same-input replays below fail their overall numerical gates.
These timings therefore **do not qualify this backend for numerical acceptance**,
nor establish a controlled speedup over differently sampled eager runs or
different AR trajectories. The `safe` name denotes request-isolation settings,
not numerical or model-quality certification.

## Source-attested acoustic build metadata

These files describe newly exported and built engines; they do not upgrade the
identity or accuracy status of the old engines below. The builders retain actual
checkpoint hashes, source/software/device identity, ONNX and external-data
binding, engine hashes and IO shapes. `provenance_status=recorded_not_audited`
means the evidence was recorded, not that eager parity or model quality passed.
Both CFM plans use 310 frames (258 prompt frames); Vocoder plans use 52 frames,
native regular convolution and strongly typed execution. Build metadata records
TensorRT 11.3, SM89 and `tf32=false`.

| Component / batch | Export metadata | Build metadata | Runtime plan |
| --- | --- | --- | --- |
| CFM B1 | [export](builds/cfm/cfm_solver_b1.export.json) | [build](builds/cfm/cfm_solver_b1.json) | [plan](builds/cfm/plan_b1_audited_source.json) |
| CFM B4 | [export](builds/cfm/cfm_solver_b4.export.json) | [build](builds/cfm/cfm_solver_b4.json) | [plan](builds/cfm/plan_b4_audited_source.json) |
| Vocoder B1 | [export](builds/vocoder/vocoder_b1_nativeconv.export.json) | [build](builds/vocoder/vocoder_b1_nativeconv_strong.json) | [plan](builds/vocoder/plan_b1_nativeconv_strong_audited_source.json) |
| Vocoder B4 | [export](builds/vocoder/vocoder_b4_nativeconv.export.json) | [build](builds/vocoder/vocoder_b4_nativeconv_strong.json) | [plan](builds/vocoder/plan_b4_nativeconv_strong_audited_source.json) |

The plans are archived unchanged, including original local engine paths; they
are evidence, not portable engine bundles. Each new build needs its own real-input
replay and quality gate. There are no binary ONNX or engine files in this archive.

## Source-attested B1/B4 real replay

The [B1 capture](real_safe_b1_audited/capture.json) completed one real request;
the [B4 capture](real_safe_b4_audited/capture.json) completed four, all through EOS.
Each records two actual native Target calls and two native Draft calls at its
declared batch, plus native CFM/Vocoder first-head calls. This is observed
execution coverage, not merely the presence of installed engines. B1 records
6 acoustic calls; B4 records 18, with fallback tails audited separately.

Both batches verify the actual engine/checkpoint identity for all four native
components. Recorded direct/graph outputs are exact; the AR cache-state checks
also pass. These findings establish artifact identity and graph/cache behavior,
not equality with eager arithmetic. Each batch has **four failed overall replay
reports**:

| Batch | AR against FP32 | AR against BF16 | Acoustic against FP32 | Acoustic against BF16 |
| --- | --- | --- | --- | --- |
| B1 | [Failed](real_safe_b1_audited/ar_reference_fp32/report.json) | [Failed](real_safe_b1_audited/ar_reference_bf16/report.json) | [Failed](real_safe_b1_audited/reference_fp32/report.json) | [Failed](real_safe_b1_audited/reference_bf16/report.json) |
| B4 | [Failed](real_safe_b4_audited/ar_reference_fp32/report.json) | [Failed](real_safe_b4_audited/ar_reference_bf16/report.json) | [Failed](real_safe_b4_audited/reference_fp32/report.json) | [Failed](real_safe_b4_audited/reference_bf16/report.json) |

Both sampled calls of Target and Draft fail native-versus-reference numerical
comparisons in each precision and batch. Acoustic failures are more specific:

| Batch / reference | CFM native head mismatches | Vocoder native head mismatches |
| --- | ---: | ---: |
| B1 / FP32 | 0 (passed) | 2,741 (failed) |
| B1 / BF16 | 0 (passed) | 677 (failed) |
| B4 / FP32 | 15 (failed) | 6,184 (failed) |
| B4 / BF16 | 6 (failed) | 2,198 (failed) |

Counts above use `deployed_vs_reference.comparisons.all` (the full CFM tensor
or Vocoder pre-clamp waveform), without double-counting generated-region metrics.
The four B1 and sixteen B4 fallback tail calls match their BF16 eager reference
exactly (`max_abs=0`); they do not match FP32 arithmetic. The additional
`bf16_reference_vs_fp32_reference` comparisons are cross-precision errors and
must not be relabeled as native-versus-BF16 failures. Overall report failure
does not mean every component, region or comparison failed.

All candidate comparisons retain the declared BF16 `atol=rtol=0.01` policy even
when the reference/output storage is FP32. These source-attested captures are
distinct from the legacy B1 capture below: new B1 CFM head comparisons pass for
these inputs, and the old legacy mismatch counts must not be transplanted here.
Bounded AR output/KV and acoustic-boundary coverage is not an audit of every
intermediate model tensor, all signatures, all AR steps, or perceptual quality.
Capture/replay is instrumented and makes no performance claim. Keep failed
reports and their frozen-input hashes; faster execution cannot waive these gates.

## Legacy B1 real acoustic capture

The [capture manifest](legacy_real_acoustic_b1/capture.json) contains six acoustic
calls from one complete-EOS request: CFM and Vocoder for the first head through
TensorRT graphs, then four eager-labeled tail calls. Recorded first-head graph
outputs equal direct engine outputs. This verifies a route, not eager parity.

Both frozen-input replay reports fail their overall gate:

| Comparison for the TensorRT head | CFM mismatched elements | Vocoder mismatched elements |
| --- | ---: | ---: |
| [Against FP32 reference](legacy_real_acoustic_b1/reference_fp32/report.json) | 5 | 2,374 |
| [Against BF16 reference](legacy_real_acoustic_b1/reference_bf16/report.json) | 3 | 879 |

These counts use the full returned CFM tensor and the Vocoder pre-clamp waveform,
without double-counting the CFM generated-region submetric. Candidate arithmetic
is BF16 even when output storage or reference arithmetic is FP32, so these
candidate comparisons use the same `0.01/0.01` policy. The four eager-labeled tail
calls match their BF16 replay outputs exactly in this capture; BF16-versus-FP32
arithmetic comparisons remain a separate failing gate in the replay reports.

**The old acoustic engine source/checkpoint identity remains unverified.**
`same_loader_checkpoint_identity=true` establishes that the current capture and
reference loaders used matching checkpoint files; it does not retroactively
establish which files produced a historical engine. New export metadata or
later source-attested engine builds must never be backfilled into these old
reports. New engines require new, separately named captures and comparisons.

The capture does not audit AR internal tensors, latent/conditioning internals,
CFM intermediate steps or crossfade arithmetic. It makes no performance or
perceptual-quality claim. Hashes and model provenance are evidence of what was
observed, not a substitute for numerical and model-quality gates.

## Legacy B8 real replay: not the request-isolated profile

The [B8 capture](real_legacy_b8/capture.json) uses the original
`sm89_bf16_trt113_full_b8.json` shared-device-RNG/device-round profile, **not**
`sm89_trt113_safe_b8.json`. Capture/reference arithmetic uses
`runtime_reference.yaml` with `target_tf32=false` and `rnn_tf32=false`.
All eight requests completed EOS. There are two observed native Target calls,
two native Draft calls, and 34 acoustic calls: native CFM/Vocoder B8 first heads
followed by 32 fallback calls. All four components have real native execution
coverage; recorded direct/graph equality and AR cache-state checks pass.

**All four engine origins remain `legacy_unverified`.** Actual engine bytes are
hashed, but historical source/checkpoint binding is not verified. B1/B4's new
source-attested engines cannot establish B8 identity. Matching current eager
loader checkpoint hashes, graph/direct equality, or a known engine-file hash
must not be promoted into historical weight-provenance proof.

All four completed reports have a failed overall gate:
[AR/FP32](real_legacy_b8/ar_reference_fp32/report.json),
[AR/BF16](real_legacy_b8/ar_reference_bf16/report.json),
[acoustic/FP32](real_legacy_b8/reference_fp32/report.json), and
[acoustic/BF16](real_legacy_b8/reference_bf16/report.json). Selected component
metrics below are native-versus-reference comparisons on the same frozen inputs,
using unchanged BF16 `atol=rtol=0.01`:

| Component / measured output | Against FP32: mismatches | Against BF16: mismatches |
| --- | ---: | ---: |
| Target logits, sum of two recorded calls | 31,415 | 40,724 |
| Draft base logits, sum of two recorded calls | 47,195 | 96,642 |
| CFM full first-head tensor | 39 | 17 |
| Vocoder pre-clamp first-head waveform | 4,829 | 3,889 |

Both native calls of Target and Draft fail their component comparison gates;
the table selects logits rather than summing unlike outputs/KV. It does not mean
every output fails: the second Draft hidden-output comparison against FP32
passes. Both acoustic heads fail for each reference precision. All 32 fallback
tail calls instead match BF16 eager exactly (`max_abs=0`); their FP32 comparisons
fail. Additional `bf16_reference_vs_fp32_reference` failures are separately
identified cross-precision errors, not failures of those BF16 tail routes.
This bounded audit neither certifies shared-RNG isolation nor measures full
corpus quality/performance, and its mismatch counts are not directly comparable
to B1/B4 trajectories with different inputs.

## Controlled runtime checks and short smoke

- [CUDA request-RNG isolation](runtime/request_rng_isolation_gpu.json) passed the
  controlled identical-probability-input scenarios, including cancellation and
  recreation; its shared-RNG negative control detected the intended violation.
  It does not assert identical real-model trajectories across batch sizes.
- [CUDA acceptance/prefix metadata](runtime/acceptance_metadata_gpu.json) passed
  54 controlled-input cases over batches 1/4/8 using the actual hash-bound ASG
  checkpoint and EOS token 8193. Exact discrete metadata and floating gates are
  separate from full-model numerical/quality validation.
- [Short soak smoke](runtime/soak_smoke.json) requested only 10 seconds at each
  concurrency 1 and 4. It completed 10 and 15 EOS requests respectively, with
  one cancellation in the latter tier, and observed KV lengths up to 399/405.
  Its `status=smoke_passed` explicitly has `full_soak_pass=false` and
  `soak_qualified=false`. The memory trend windows were only 5.96/5.61 seconds,
  below the required 60 seconds; neither is a qualified memory-stability result.

This earlier smoke used the original `runtime.yaml` defaults (`target_tf32=true`),
not the TF32-disabled reference configuration for the subsequent 600-second
tests. Its original JSON predates the runtime-file/source preflight fields and
is preserved without backfilling them. Do not combine its timings or scope with
later reference-config soak evidence, or claim it covers 8/16/32/64 concurrency.

The original [B1 BF16 replay preflight rejection log](runtime/real_safe_b1_bf16_preflight_busy.log)
records `GPU6 is busy (21854MiB,23%)`: the cooperative lease rejected that attempt
before replay. After utilization returned idle, the retry completed and produced
the [B1 BF16 acoustic report](real_safe_b1_audited/reference_bf16/report.json),
whose numerical gate remains failed. A successful retry means completed
execution, not numerical acceptance. The log does not establish whether the
23% utilization came from another process or residual activity of prior work;
no such attribution is made. Both the rejection and later result are retained.

## Initial ten-minute soak: B1 passed, B4 RSS budget failed

The [initial formal run](runtime/soak_600_initial.json) used the TF32-disabled
reference runtime, one warmup wave per tier and the unchanged memory budgets.
The [original execution log](runtime/soak_600_initial.log) records its nonzero
exit after B4. Both completed tiers are retained; **this run failed overall**
and did not reach 8/16 concurrency.

| Concurrency / model batch | Measured seconds | Complete EOS / cancelled | Maximum observed KV | Peak drained CUDA growth, MiB | Peak drained RSS growth, MiB | Late RSS slope, MiB/min | Result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 / B1 | 600.77 | 489 / 48 | 607 | 0.00 | 138.21 | 1.377 | Passed |
| 4 / B4 | 601.34 | 866 / 86 | 596 | 0.00 | 287.71 | 1.106 | Failed: RSS growth >256 MiB |

Both tiers passed actual concurrency, complete EOS, long KV, microbatch limits,
per-request RNG boundary checks, error recovery and fully drained slot ownership.
B4's post-warmup RSS rose from 3,513.63 to 3,801.34 MiB, exceeding the fixed
256 MiB growth budget. About 93.2% of that growth occurred in the first 91 seconds;
the last 300 seconds added approximately 5.38 MiB. The late slope passed the
16 MiB/min limit. This supports investigating initialization, shape/text caches
or allocator high-water retention, **not** a proven root cause or a guarantee
that there is no slow leak. Parent-side report history cannot explain the metric:
RSS is measured in the model worker. No thresholds were relaxed.

The original top-level B4 `failed_checks` lists both `memory_absolute` and
`memory_trend`: its diagnostic combined the absolute and trend predicates.
The underlying `memory.metrics.*.trend_passed` fields correctly remain true.
Subsequent code separates those diagnostic labels while keeping the same
overall conjunction and thresholds. [Regression tests](runtime/soak_diagnostic_labels_tests.log)
cover both an over-budget early allocation with flat late RSS and an excessive
late slope with a below-budget peak. The original JSON was not rewritten.

Independent 8/16-concurrency runs and a separately labeled B4 fully warmed
control are in progress. A warmed control will not replace or erase this cold
growth failure; its preparation cost, warmup count and measured window must be
reported explicitly. These are bounded stability observations, not numerical,
quality, isolated-device performance or indefinite leak-free certification.

## Complete-EOS paired quality: 256 cases per arm

All five generation arms completed exactly 256 requests through EOS. The four
paired evaluations below each compare their complete PCM16 audio against the
same FP32 eager B1 corpus, text, request seeds and nine generation-time
reference-audio identities. No first-packet crop substitutes for full audio.
The original `generation_summary.json` and 256-row `generation.jsonl` for each
arm are archived under `quality/<arm>/`; audio remains local.

The fixed gates are mean UTMOS relative decrease <=3% and **corpus character
CER** absolute increase <=0.02 (two percentage points), not WER or mean
utterance CER. The baseline is UTMOS **1.643671675**, CER **144/4093 = 3.5182%**.
Negative UTMOS drop means an observed increase, not a reduction.

| Candidate / report | Mean UTMOS | Relative UTMOS drop | Character CER | CER increase, percentage points | Quality gate |
| --- | ---: | ---: | ---: | ---: | --- |
| [Request-isolated TRT B1](quality/quality_safe_b1.json) | 1.635844096 | +0.4762% | 159/4093 = 3.8847% | +0.3665 | Passed |
| [Request-isolated TRT B4](quality/quality_safe_b4.json) | 1.645488889 | -0.1106% | 167/4093 = 4.0801% | +0.5619 | Passed |
| [Original shared-RNG TRT B8](quality/quality_legacy_b8.json) | 1.644750395 | -0.0656% | 162/4093 = 3.9580% | +0.4398 | Passed |
| [BF16 eager B1 control](quality/quality_eager_bf16_b1.json) | 1.664038122 | -1.2391% | 147/4093 = 3.5915% | +0.0733 | Passed |

These are **relative automatic-quality gates on this corpus**, not high absolute
quality, a listening test, or floating-point equivalence. The approximately 1.64
baseline UTMOS is explicitly retained; a small relative decline does not imply
high subjective quality. All TRT numerical failures above remain failures.
The legacy B8 result measures that observed artifact's quality; its engine
checkpoint provenance remains unverified, and it is not the isolated B8 profile.

Evaluation used one CPU-only process with four Torch CPU threads, offline local
UTMOS and Paraformer models; all reports record `cuda_initialized=false`.
They include evaluator source/model hashes, software versions, full-wave
resampling/normalization policy, per-case scores and character edit counts.
Generated wave hashes, sample counts, PCM16 encoding, EOS and exact case
coverage were checked. CER sums S+D+I over all 4,093 reference characters;
an independent Levenshtein calculation cross-checks the character counts.

Paired reference identity compares the original generation-time SHA256 and byte
records; the evaluator does not reopen the original reference audio. Matching
loader hashes do not extract or independently prove old engine constants.
The old FP32 generation JSONL has no per-row emotion field: its emotion is bound
by the corpus hash only, not retroactively marked as observed. The corpus's
embedded original-text-source hash is a declaration, not an independent source
dataset verification. External reference audio/evaluator prerequisites and
reproduction commands are in [SM89_AUDIT.md](../../../inference/docs/SM89_AUDIT.md).

## Packaging and dependency checks

The latest [CPU suite](runtime/stage2_cpu_tests_soak_labels.log) passed 212 tests;
the [earlier 211-test run](runtime/stage2_cpu_tests_final.log) is retained. Four
GPU-only tests were skipped there and [passed separately on GPU6](runtime/gpu_boundary_tests_v2.log).
The [CLI help check](environment/cli_help_final.log) and
[offline wheel build](environment/wheel_final_build.log) completed. The checked
wheel contains 215 members, including two CUDA source files and attribution;
it contains no model weights, engines, ONNX, audio, tests or benchmark scripts.

The unused historical `descript-audiotools==0.7.2` dependency required
`protobuf<3.20`, which conflicts with the current ONNX exporter. There are no
runtime imports of audiotools in this source tree. The package now directly
declares the existing model utilities' Matplotlib 3.11.2 and SciPy 1.17.1 imports.
The [combined inference/export dependency input](environment/trt113_main_build_requirements.in)
and [resolver output](environment/trt113_main_clean_resolved.txt) record a
successful 100-package Python 3.11/Linux x86_64 resolution using the Tsinghua
mirror. This is a clean dependency-solver check, **not** a fresh environment
installation or GPU execution. The running environments were left unchanged;
unused historical installed packages were not silently removed.

## Subsequent evidence

Remaining 600-second concurrency tiers and the B4 warmed control will be indexed
separately when completed. Nothing here claims those outstanding gates passed.
Reproduction commands and external asset
requirements are in [SM89_AUDIT.md](../../../inference/docs/SM89_AUDIT.md). Preserve the
directories above and their original JSON; append later runs under distinct
names, retain failures and verify copied-file hashes before publication.
