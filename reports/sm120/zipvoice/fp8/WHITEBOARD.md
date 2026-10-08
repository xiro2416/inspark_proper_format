# ZipVoice SM120 FP8 — seven targets published privately

Goal: /workspace/A_1007, B1/2/4/8/16/32/64 complete unsplit batch,
concurrency1. Median prepared CPU conditions (shaping/H2D included) through
all orderedPCM, natural760frames/78paddedtokens. Frontend/init/warmup/capture/
WAV writes excluded. Dynamic600/760/920 frames,52/78/141 tokens; reference375
from continuous4sVAD/accurateASR; original8Eulersteps,t_shift.5,guidance1,
feat_scale.1,masks/prompt isolation/request-owned fullbatch noise preserved.

Authorized physicalGPU3 only, UUIDGPU-9709cc87-48d8-c210-a7c7-37ded3f1fdc0,
RTX6000D SM120/85651MiB, existing600W; serialcooperativelease, no unrelated
jobs changed. Hardwareprofile /workspace/.codex/skills/gpu_parameters/
rtx6000d-sm120.md. RuntimeTorch2.11+cu130/TRT11.3.0.99/Triton3.6; workspace
environments/caches. CPUquality separate environment, no evaluationGPU.

Reference original eager FP32, TF32disabled, full targetbatch/exactnoise.
Initialsource510e7c2624fbd3b3693f4a2b690291948f6ceae8; weights HF source
f6da21d25400d1b3e0b9a70de333503b3374df76. ConcurrentIndexNVFP4 main
1e7c5e9f4c56501b2a42a010e8c52df9be0a2459 fast-forwarded/preserved; no Index
GPUwork and no SM89 closure changes.

Frozenfirst4FMfloating/last12eligible W8A8E4M3FN, per-tensorstaticmax;
216linear projections including60wrapped outputs repaired before migration.
Original156scales bitexactunchanged;prototype retained. Protectedlayers,
last12's24depthwiseconvs/sensitiveops/Text/Vocos/ISTFT stayfloating.
128bilingualrealcalibration64zh/64en, all8steps;16heldout10zh/6en disjoint
IDs/audiohashes. Natural760/78 without changingfulltext/duration. RecipeSHA
7d907606b7e0ac7903b6a713ae2ee54c7441b55968f37855f53e86b7650f9ccf.
Strict1538tensorrestore;216FP8weights/QDQ;207nativeFP8GEMMtactics account
for all216nodes (fusion). Level5/FULLtiling search, separate SM120 caches.

Sevennative migrations accepted:16realcases,extremeframe/token combinations,
heterogeneous rows,direct/captured state/WAVbitexact,fulltextcontrol,
fullbatchFP32noise/speech/mask/timeaudits and pairedCER/WER/UTMOS/SIM-o.
Metrics report-only,no fixedL2/perceptualequivalence/productioncertification.
Qualityhomogeneous batches use eligibletextreuse; general performance
explicitly disablesreuse. Rawtexts/transcripts/audio/states/profiles stay
ignored outputs/fp8; publicaggregates/hashes only.

Optimization: B16trace exposedrelativeposition/softmaxmaterialization.
Sourcefloatingonlineattention adapted16boundaries/48AV outputs; frozenFP8
scales unchanged. NonlinearIEEEQK/RNAAV, normalTF32RNA/shared unroundedrow
statistics preserved. Sourceattentionregresses B1/B2/B4. Geometrynonlinear
16/16 normal32/64 helpsB4/B8, regressesB16/32/64; native/originalattention
retained there. PCMsourcepolicy/alternatives screened then20ABBA confirmed.
ProtectedfloatingFFN fusion mostlyregresses;380FF1 also regresses; selective
760FF1 gross~.05%E2E beforeintegration. Invaliddefaultstream capture archived
and excluded; correctedCUDAgraph probes in history/013. Remaining native
GEMM/floatingcompute reviewed; no claim ofglobaloptimum or newprecision.

Final760 matchednative→retained mediansms/gain:
B1 28.005→27.984(sameroute);B2 40.991→41.000(sameroute);
B4 65.114→63.427/2.59%;B8 121.630→102.788/15.49%;
B16 234.996→192.178/18.22%;B32 494.763→380.437/23.11%;
B64 1093.112→840.861/23.08%.
Routes native1/2;geometry4/8;attention16/32/64. PCMchunk/workers:
16/4,16/4,1/4,16/4,32/4,16/8,32/4. Finalshort/primary/long, p95,
throughput,NVMLpeakmemory,>=30s usefulcontinuous power completed; peaks
266/322/367/423/439/469/497W retainedraw. Nativefirstbaseline and candidate
artifacts preserved. Final localselection outputs/fp8/final-selection.json;
public tables docs/zipvoice-sm120-fp8.md, RESULTS.json, perbatchhistory002.

PrivateHF additive revision45883af1d711ef065354e0069177ecb4e5898e27,
89assets verified. OldSM89 assethashes preserved; HFauto-added29LFS rules
only fornewpaths, oldrules intact; READMEintentionally integratesFP8.
Sevenhash-bound bundles include selectedapplication/plugin/sourceidentities.
Allsevenempty-cache downloads pass48shape/mixed cases and exactPCMhashes;
publicreceipt history/015-private-publication.json. PublicCLI native/geometry/
attention primarypaths also exact. RelevantCPUregressions34pass.

Reproduce: bash scripts/bootstrap_zipvoice_fp8.sh;
INSPARK_REPO_ROOT="$PWD" PYTHONPATH="$PWD/src" .venv-zipvoice-fp8/bin/python
-m inspark_infer.command zipvoice ensure --precision fp8 --gpu 3.
Registryconfigs/hardware/sm120/
zipvoice_fp8_registry.json pinsprivateHF revision; HF_TOKEN viaenvironment.

Remaining: normalGitHubmain commit/push, then newGitHubcheckout withnew
emptycache, independent sourceimport path and allsevenshape/mixed PCM checks.
Only afterthat mark taskcomplete; retainhistory/oldassets, no forcepush/delete.
