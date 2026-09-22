# SM89 stage-two evidence

This directory archives completed RTX 4090 / SM89 observations. It is separate
from historical SM120 results and from the
[stage-one build and preliminary audits](../trt113_b1_b4_stage1/README.md).
The files are byte-for-byte copies of existing local JSON, not regenerated
results: [COPY_MANIFEST.json](COPY_MANIFEST.json) records their source paths,
SHA256 hashes and sizes. Original absolute paths, source identities, errors and
provenance fields are retained without alteration.

The archive paths have distinct scopes:

- `baselines/`: same-model B1/B4 eager/compile execution and sampled compile-versus-eager comparisons.
- `legacy_real_acoustic_b1/`: one completed real trajectory captured with the old acoustic engines, plus frozen-input FP32/BF16 replays.
- `builds/{cfm,vocoder}/`: new B1/B4 export, engine-build and plan metadata, kept separate from the legacy capture.

No tensors, audio, ONNX graphs, engines or checkpoints are included. References
to local `.pt` files in the JSON identify withheld replay evidence by hash; this
public JSON-only archive is not a standalone executable replay bundle.

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

## Subsequent evidence

New B1/B4 source-attested acoustic captures, paired quality evaluation and
concurrency/long-sequence/soak evidence will be indexed separately when completed.
Nothing in this initial archive claims those gates passed. Preserve the
directories above and their original JSON; append later runs under distinct
names, retain failures and verify copied-file hashes before publication.
