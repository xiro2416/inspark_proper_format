# B4 mixed-precision migration

Migration baseline only. Physical GPU4; concurrent migration on distinct devices, so this short run is not the coordinated final performance report. Same source model/calibration, no recalibration or new math.

## Exact static engine coverage

| Component | FP4 GEMM layers | FP8 Conv implementation layers | FP32 FIR | Build / tiling / actual aux | Engine SHA256 |
|---|---:|---:|---:|---|---|
| target | 72 | 0 | 0 | 5 / full / 0 | ed57f8c131215eb19e07c49c14d2d9d0692d26865511592bf294c804384f31fa |
| draft | 12 | 0 | 0 | 5 / full / 0 | e46e432d57a58d1848cf41c71e86dbeb355cb0c75899c362788a67be21c14c6c |
| context | 4 | 0 | 0 | 5 / full / 0 | 4f4a2615f49c25e88a286734ee33373a87e15be1338a9a96926ffc19bc625a9b |
| prefill | 72 | 0 | 0 | 5 / full / 0 | 7e0585899f634b1d0bd4abee5a77833e53e2020811a841df9743c7616b5a04c1 |
| latent | 72 | 0 | 0 | 5 / full / 0 | aaed7d73e8b67d27ffaa238a5932fd8548e5fef8b5fc6062442469522518e542 |
| latent_suffix | 72 | 0 | 0 | 5 / full / 0 | 69f45e67f2d9c967028ac78a063fe217de5ed278f7cffe0faead91270c495db4 |
| cfm | 168 | 64 | 0 | 5 / full / 0 | 28527b43b9ee49b38daf234026fe3b79b059d11550a30c024afed696fe4d854a |
| vocoder | 0 | 72 | 109 | 5 / full / 0 | b8a613612b0bab1cbc7003282b2ff05c3d8485dbf0800bba5829bdbe724cdb0a |

Native layer counts differ from logical roles. Recipe remains135NVFP4/92FP8/90BF16. CFM full4step/F310/P258 single engine; Vocode directFP8Conv/Mel52 completeengine plus109 existing FIR plugins, FP32 ABI. Target official Attention with original BF16 prescaledQ/K/V/mask and explicitBF16output rounding. Native DSparkWorker/RNN bridge+ordinaryTRT, not completeTRTLLM Executor.

## Inherited mechanisms

| Source mechanism | B4 decision |
|---|---|
| KV80/head-major identity slots/GPU PCG/accepted-only commit | Retained, new B4 buffers/capture |
| 7-step RNN, initialstate0 eachround, compiled rewrite | Retained |
| Two-round parent Graph, head CUDA Graph | Recaptured B4 |
| Prefill reuse/latent suffix/packing/GC ownership | Retained; no repeated text compute reuse |
| CPU8 and conditionstreams2 | Inherited baseline; reconsider in optimization |
| Source CFM FULL/L256MiB | Retained |
| Source Vocode MODERATE | Target strongbaseline FULL; actual engine inspected |
| B128 CFM/Vocode B64x2 and AR128→64→8 | Inapplicable to B4 |
| Source aux2 no gain, source ARFP8 slower | Source-conditioned conclusions; smallbatch future focusedrechecks only |

## Inherited precision exception

Four low-precision logical roles vocoder.stages.2/3/4/5.ups.0 retain FP8 activation/weight QDQ, but corresponding ConvTranspose actually runs Float IO with FP32 deconv tactics in B4 and source B64. This is preserved source coverage, not a newly lost target path;76 FP8 roles must not be described as76 native FP8 compute operators. Exact spec/scales, ONNX stride/pad/kernel/group geometry and both inspectors are in deconv_precision_evidence.json. Profile their exposed cost in optimization before attempting a mathematically equivalent zero-insert Conv graph.

## Validation and reproduction

Minimal real B4 wave plus3warm/10waves, zero fallback required. Acoustic frozenhead and AR frozenround audits use actual deployed history/samplechain; finite PCM/reference and4 unique PCM required. Floating errors reporting-only, not fixedL2 acceptance. 13CPU recipe/geometry/Attention tests passed. No speechquality certification or exhaustive optimumclaim.

`/workspace/index-tts/.venv/bin/python artifacts/mixed_b4/migrate.py` builds and validates serially onGPU4 with allcache/envsettings inside script. Outputs `artifacts/mixed_b4/deployment.json`, runtime.yaml, native_coverage.json, baseline_short.json, acoustic_audit.json, ar_audit.json, source_identity.json; detailed history logs.

## Handoff

Stop GPU work at migration gate. Remaining optimization leads: smallNVFP4GEMM quantization/launch/layout,1/2/4roundGraph tradeoff, CPUworker/IPC overhead, conditionstream/prefill overlap, existingFIR tile/resource policy. Parent coordinates exclusive baseline/final5warm30wave+power15 windows before publication.
