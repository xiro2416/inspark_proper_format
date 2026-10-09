# B4 optimization handoff

Selected artifacts validated; formal exclusive timing/power completed by parent authorization; see RESULTS.md. All GPU work used only physicalGPU4. Source/migrationbaseline preserved.

## Retained changes

- CPU text workers8→4; parentGraph2→1round (requestready checks eachround); lateverify12→8.
- ExistingtargetKVfused append, prefill readonly/context views+overlap enabled. No repeated-textcompute reuse.
- Target72 lowprecisionGEMMs use frozenFP8 ratherthanNVFP4;24BF16protected unchanged. NewexactB4Target/Prefill/Latent/Suffix4engines. Draft/Context remainNVFP4; CFM/Vocoder unchanged.
- Global63NVFP4/164FP8/90BF16, all317roles accounted. SourcecheckpointTensorhashes unchanged. OldallF4Targetpackedweights do not represent this scheme; batch-specificFP8Targetstorage must be canonical[out,in] with frozen scales.

## Short matched observations

| Comparison | P50 ms | Notes |
|---|---:|---|
| Originalmigration3warm10 | 67.45 | Parallelmigration, diagnostic |
| Schedulingcontrol20 | 67.36 /64.62 | Twoendpoints showhost drift |
| Retainedscheduling20 | 60.33 | All80 code+PCM hashesequal |
| Same-schedulingF4Target20 | 62.80 /60.78 | Adjacentcontrol endpoints |
| SelectedFP8Target20 | 59.01 | Zero fallback;222physicalrounds vs202 |

FP8Targetfrozen-inputGraph median1.348vsNVFP4 1.823ms. Thislocalgain doesnot equalE2Egain: acceptedhistories changed; acceptance37.52%vs38.76%, requesttailslonger in20waves. Choose finalE2EP50 inparentcoordinatedwindow. These arenot final5warm/30wave+power15 results.

## Actual implementation and fusion review

- All8engines L5/FULL/actualaux0/tacticsauto;CFML2=56MiB, others112MiBauto;TimingCacheenabled,stronglytyped,TF32off. Full4stepCFMF310/P258/Mel52Vocoder.
- Target4engines each72nativeFP8GEMM;Draft12F4,Context4F4,CFM168F4+64F8Conv,Voco72F8correlation+109existingFP32FIR+4Floatdeconvs.4deconvs have samefallbackmath as sourceB64 afterFP8QDQ; no newprecisionomission.
- Inspector/trace shows nativeQKV, officialBF16Attention,FP8GEMM+GELU,Myelinresidual/norm/castcombination;existingFIR/Snake/FIR andKVappendfusion retained. Notclaiming allGEMM+activation+norm globallyfused.
- Two-waveTorchprofiling isdiagnostic, spansoverlap/instrumentation changesGraphcost. SelectedFloatdeconvtotal0.265ms/2waves, FIR4.322ms/2waves. No customnewGPUmath, no fullTRTLLMExecutorclaim.

## Rejected and stopping decision

- burst4 slower; threads no clearadvantage; conditionGraph74actualhits/0misses butP5062.24vs60.33ms, notretained. Allruntimefallbacks zero; someconditionPCMrounding differences acceptablebutnoE2Ewin.
- NoFP32deconvrewrite: only~0.133ms/waveexposed, belowworthwhile standalonework; no deeperFIR/newpersistentdesign. Draftwholecost atB4 onlyfewms andno remainingconfirmed largegain; userstoprule honored. No globaloptimumclaim.
- FP8Targetexport successfullywrote/hashverifiedallassets butnativeinterpreterexit stalled; ownPID interrupted, noexternalprocesses touched. Laterexportwrapper callsnormalmain/finallycleanup+flush thenOSexit. Exactcauseunknown, ptracedenied.
- Accidentalstdlibqueue/profile namecollision inartifacthelper renamed; unintendedrepeatcheap-screens excluded fromgainproof. Confirm20wavefiles/baseline untouched.

## Validation and delivery

Selected8engine staticIOexecutedfinite; actual4voices/3secVAD andmixed128text manifest, allselectedtests zerofallback. AR frozenproposalconditionalq reproducesactualsamples, RNNstatezeroeachround;latestmodelidentitygates passed. Acoustic4uniquePCM/no saturation; CFM/Voco andARfloatingaudits reportonly. No fixedL2/CER/MOScertification.

Selected: `/workspace/inspark_mixed_b4/artifacts/mixed_b4/selected_deployment.json`; runtime `/workspace/inspark_mixed_b4/artifacts/mixed_b4/runtime.yaml`. F4schedulingbackup `/workspace/inspark_mixed_b4/artifacts/mixed_b4/optimization/combo_views_deployment.json`. Nativecoverage,calibrationmapping/SHA,selectedaudits,selectedtrace under artifacts/mixed_b4/optimization; report copies beside this file.

Parent remaining: formal5/30+power15 completed in RESULTS.md;packageall8plans/actualcalib4mapping/latestmodelweights, exportbatch-specificTargetFP8packedweights, portable/freshdownload3wavevalidation, additiveGit/HFpublication. WorkerGPU4released; no publication done.
