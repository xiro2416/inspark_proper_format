# B128 SM120 FP8 — migration and optimization validated

User requests migration first, optimization after acceptance. Source B64
attention route/PCM32 workers4; median840.861ms versus native1093.112ms,
whole-board mean468.7W/max497.2W; these are source conditions only.
GPU3 UUIDGPU-9709cc87-48d8-c210-a7c7-37ded3f1fdc0,SM120 RTX6000D,
85651MiB/existing600W; serial one-GPU lease. Hardware profile
/workspace/.codex/skills/gpu_parameters/rtx6000d-sm120.md (FP8dense287.57TF/s,
Triad1311.1GB/s context, not target performance).

Frozen original checkpoint/216 W8A8E4M3FN staticmax projections,4protected/
12eligible layers; original128 calibration cases/scales untouched. Text,
Vocos,depthwise/sensitiveops remain floating; Torch2.11/TRT11.3/Triton3.6.
B128 unsplit model batch,concurrency1,dynamicframes600/760/920,tokens52/78/141,
375promptframes,8Eulersteps/t_shift.5/guidance1/feat_scale.1. Primary prepared
CPU conditions shaping/H2D through all orderedPCM at natural760/78;
full text encoder, exclude frontend/init/warmup/capture/WAV writes.

Semantic original fullbatchFP32/same noise, real16 heldout bilingual cases;
CER/WER,UTMOS/SIM-o and numeric deltas report-only,no fixedL2 gate. Validate
fullbatch operations/row mapping/masks/timegrid,extreme profiles,mixedtext,
direct/captured state/WAV bitexact,original PCM trimming and ordering.

Independent zipvoice_fp8_b128 namespaces/scripts/cache/artifacts; source
published closures preserved. Transfer floating onlineattention16boundaries/
48AV outputs,IEEE nonlinearQK/RNAAV,TF32RNA normal/shared unroundedstats.
Build native target comparison and inherited attention baseline level5/
FULLtiling. Verify all216projections actuallyFP8 (some fusion), no fallback.
Then adapt/review fullmodel capture,serial workspace,eligibletextreuse/full
heterogeneous fallback,cuFFT ISTFT and sourcePCM32/4. Source geometry16/16,
normal32/64 regressed atB64; target recheck only after migration acceptance.
Source protectedFFN fusion mostlyregressed, selective760FF1 gross.05% atB16;
changed target work distribution may justify focused optimization probe.

Migration complete: native/inherited207actualFP8GEMMtactics cover216nodes.
Full128PCM,16real/6boundarymixed/directgraph/fulltext checks pass. Original
fullB128FP32 audits/noise/speech/mask/timeexact,48pairedqualityrows complete.
Matched20ABBA760 native2325.255→inherited1813.091ms,+22.026%. Short/long
ABBA also complete. SourcePCM32/4 retained; sustainedboardmean467.7W,
max482.9W at600W cap. Publicreceipt history/003-migration-acceptance.json.

Optimization aftermigration: actual128 Nsight node trace reviewed; native
GEMM/TF32 floating compute/onlineattention dominate, repeatedresidual/FP8
casts exposed. Geometry16/16 normal32/64 regresses4.455%; retainattention.
PCM screen6policies then20ABBA: candidate gain-.070%, retain32/4.
ResidualGEMM+FP32bias/residual+dualF32/FP8 output microprobe +35-40%; whole
36regions regressed1.507%, FFN1-only12regions regressed.938%, both rejected.
Prototype constant lookup/argumentABI fixed; successfulwhole128smoke before
comparison; no acceptedresults from invalid builds. GPU3 staleutil recovered
by exclusiveCUDAhealthcheck, noreset/no unrelatedtask changes.

ProtectedfloatingFFN source fusion rechecked at128 for760/380 widths1152/
1536/1920. Selective2full-lengthFF1 and2half-lengthFF3 gross.215%E2E;
whole4regions20ABBA gain.065% (1813.043→1811.866ms), pairedbootstrap95%
saved.468..2.188ms. Candidatefull128/16real/6boundaries/mixed/directfulltext
checks pass. Exactoriginalnoise/text/speech/mask/time confirms reusable
originalfullB128FP32 audio; candidate quality48pairs complete, CER/WER same
as inherited; UTMOS/SIM-o changes reported. Short/longABBA stillrunning.

Finalretained attentionprotected: sourceattention plus fouroriginalfloating
FFN regions; PCM32/4. Native2324.803→final1811.089ms,+22.097% same-round
20ABBA. Incrementaloptimization .065% is separatefrommigration22.026%.
Short615frames1400.150ms,primary7601811.114ms,long9172274.878ms;
primaryP951813.052ms,throughput70.66requests/s,NVMLpeak4871.1MiB,
>=30smean463.4W/P95478.1W/max485.4W,existing600W cap unchanged.
40relevantCPUregressions pass; original7andnew128 sourceclosures hash-match.
Finalselection outputs/fp8/b128/final-selection.json; publichistory002 and
RESULTS.json, docs/zipvoice-sm120-fp8-b128.md.

PrivateHF additive revisiona1b6f97ee4ef58221b4c1062516f66de51602dd1,
13files(bundle+integratedREADME); oldweights/scales/ONNX/sevenassets/oldLFS
rules preserved. Newindependent registry zipvoice_fp8_b128_registry.json;
oldseven pins unchanged. EmptyprivatecacheB128 seven primary/shape/mixed
cases pass allorderedPCM and exactWAVhashes. Publichistory007receipt.
ConcurrentIndexB128 main2942952b67b6729c60b0c11b29a2190e9b9cfc9c merged
without conflicts; Indexrelease preserved, noIndexGPUexperiments.

Completed: normalGitHubmain source commit9ca4691c545c8c71212056ca993ddf441a88359f, preserving
concurrentIndex128 release2942952b67b6729c60b0c11b29a2190e9b9cfc9c.
NewGitHubcheckout/newemptycacheB128 sevenprimary/shape/mixed cases pass
ordered128PCM/exactWAV hashes. Existingpinnedlibraries reused,7privateinput
files/zero projectsource files copied; sourceimport freshcheckout only.
Finalreceipt history/008-fresh-github-checkout.json. Taskcomplete; stopafter
acceptedretainedartifact/executionreview/publication. No globaloptimality
claim; relatedmechanisms probed, remaining selectivegross<.1%.
Allsource/history/caches/engines preserved; noforcepush/deletion.
