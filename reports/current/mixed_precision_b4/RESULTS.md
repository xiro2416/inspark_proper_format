# B4 final result

RTX6000D / SM120 / physicalGPU4; independent exclusive window while B16/B32 CPU/GPU experiments paused. Same128 bilingual inputmanifest/four3secVADrefs. Each5warm/30waves (120requests); powerseparate15sec sustainedE2E. Statistics unit iswave, not120independentrequests;P99 isshorttest quantile. No textcompute reuse/dummyload/outlier deletion.

| Scheme | Afteradmission P50/P95/P99 ms | Includingadmission ms | Power P50/P95/P99 W | Sampledpeak GiB | Physical rounds | Acceptance |
|---|---|---|---|---:|---:|---:|
| migration | 65.15 / 72.19 / 73.13 | 65.75 / 72.72 / 73.67 | 184.15 / 284.53 / 300.24 | 12.33 | 320 | 38.57% |
| all_f4 | 60.94 / 69.12 / 73.47 | 61.63 / 70.96 / 74.10 | 181.24 / 275.10 / 282.59 | 12.37 | 304 | 38.57% |
| target_fp8 | 56.94 / 65.84 / 67.93 | 57.52 / 66.40 / 68.45 | 198.80 / 269.94 / 290.44 | 13.52 | 325 | 37.92% |

Scheduling gain 6.47%; TargetNVFP4→FP8 under same selectedscheduling gain 6.56%; combined 12.61%. Do notattribute allgain toprecision. All3zero AR/CFM/Vocode/prefixfallback. Everywave retained; no latencyoutlier aboveP95×1.1 in these30waveblocks.

Selected: Target72FP8/24BF16, Draft12NVFP4/6BF16, CFM51NVFP4/16FP8/20BF16, Voco76FP8QDQ/40BF16 roles; global63/164/90.4 VocoConvTranspose roles still actualFP32deconv inheritedfromsource; countroles separatelyfromnativekernelprecision. Target4newexactB4engines keep sameBF16officialAttention/Graph/L5FULL; actualnativecoverage/calibcomponentSHA in selected_native_coverage.json/selected_precision_identity.json.

Retained CPU4/burst1/late8, existingKVappendfusion/readonlyprefill+contextviews/overlap;CFMfull4stepF310P258singleengine/Vocodewholeengine+109FP32FIR. Actualaux0,TimingCacheon,tacticsauto,CFML2=56MiB,othersauto112MiB. Graphburstchange saves extra fullrounds320→304; changingTargetprecision changesprobabilities/acceptance androunds304→325, so singleenginegain does notdirectlypredictE2E.

Selected8engineIOfinite, actualAR/CFM/Vocode routeszero fallback,4uniquePCM/no saturation. LatestcheckpointTensoridentitygates and exactcomponentrecipe checks passed; ARconditionalq/samplechain/RNNzero-state andoperationsemantics audited. Floatingerrors reporting-only in selected_floating_audit.json; no fixedL2/MOS/CER claim.

Stop: confirmed worthwhile candidate retained; burst4/threads/conditionGraph noimprovement; exposedFloatdeconv~0.133ms/wave andremainingDraftwholecostfewms do notjustifyfurtherwork underuserstoprule. No globaloptimumclaim.

Reproduce benchmark run --gpu4 --batch4 --deployment /workspace/inspark_mixed_b4/artifacts/mixed_b4/selected_deployment.json --config /workspace/inspark_mixed_b4/artifacts/mixed_b4/runtime.yaml --manifest /workspace/A_0924/artifacts/sm120_0924/unified/cases.json --warmups5 --waves30 --power-seconds15 (flags separated normally, environment.py contains exactnativeenv). SourceformalJSON/logs under artifacts/mixed_b4, source andallF4backup retained. Parent stillpackages8engines/4componentcalibrations/TargetFP8batch-specificcanonicalweights and performs portable/freshdownload validation/publication. GPU4released; workerdidnotpublish.
