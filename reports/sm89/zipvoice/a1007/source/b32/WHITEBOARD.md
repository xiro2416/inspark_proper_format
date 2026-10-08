# COMPLETED: INT8 SmoothQuant B32 / GPU1

Goal fulfilled: full unsplit32/concurrency1, physicalGPU1 UUID GPU-7dab7d6b-ac8c-7ccc-6410-916d3b7689b3,configured/enforced400W. Originalweights/scales,alpha=.5,first4floating/last12INT8,authorizedTF32 unchanged.70complete sampletexts+5VAD4s/localASR refs, mapping priority/numericalorder differences allowed; CER/UTMOS/SIM-o sideevidence. SourceB24 128sealedartifacts unchanged59; publicreleaseunchanged,CPU6checks57pass.

Retained: current_best.json/delivery-manifest.json; engine builds/normal_tf32_fm_full/engine.plan SHA a14eeb0698912a43bb5b2df4f9e9ef6b850221cbdb96244ad706a09b35bc6e93; runner code/measure_delivery_gpu1.py;cases/delivery inventories. ReproductionREADME command verified5full32requests59. GPUlockfree. No requiredwork pending.

Coverage16/17:180originalI8 modules120native+12value+24residual+24DW;12F32FFN,48sequentialattention,298weightednodes;protectedfirst4I8count0. StrongtargetFULL/level5/aux0/maxsupportedtactics. Native141TF32/1FFMA versusretained137TF32/5FFMA, fourprotectedoutprojchooseFFMA; originalW/scales/dataflow verified, noquantizationrecipechange. Transfer8EulerGraph,B1text/fullmixed,sharedarena,cuFFTISTFT,freshcompactH2D,firstD2H/CPUoverlap,all32orderedPCM.

No addedoptimizationretained. SourcePCM strategy: rawwave_samples=(T-376)*256; remaining31chunk6 forT<=1391,chunk4 forT>=1392. Normal64x32retained. Differentlengthsreviewedseparately54:fiveFMprofiles;exactbuilderAOT15actualdownsampledomains,19uniformdispatchdomains;5NLlengths;110FFN/110protectedprojectionvariants;10actualPCMlengthprobes. KB16helps146/190in isolation butuniformruntimebranchslowerallshapes; FFNalternativeslose. PCMformal20perroute1026+.018%,1381+.487%,906-.267%,1017-.283%,1279median-.081%; no stableexpandednewband, allcandidatesrejected55. SourcefullmigratedB32engine/runnerarefinalbest; incremental0byidentity, no meaningfulnewGPUcomputeacceleration.

Finalformal20perrouteABBA/BAAB unprofiled all32PCM:
T582 native.549089s->retained.461937s,15.872%; mean/P95/rawmax333.993/398.267/398.267W.
T760 .776563->.624008s,19.645%;350.197/399.631/399.688W.
T1026 1.190231->.945584s,20.555%;366.251/396.860/397.985W.
T1398 1.818527->1.364014s,24.993%;374.814/399.791/401.438W.
T1750 2.523721->1.836158s,27.244%;381.188/398.869/402.866W.
ClockCPUpreparedshaping/H2D->all32orderedPCM,excludesfrontend/init/persistentallocation/warmup/capture/WAVwrites. Claims onlythese5T; NVMLwholeboardwindow samples notinstantaneous/process-only andnotclipped400W.

Correctness66: native70+retained70+mixed32=4512mainPCM; initialcandidate70+mixedfull32guards,finaloriginalpolicy31changedcasesfresh reruns/39samebody+branchreuse. Originalnoise/speech/mask/timegridselectedexact;Graph/directfullstateexact;all32condition/D2H/PCMvsoriginalsync referenceexact. Native/retainedstateL2max.221report-only, no strictgate. All420qualityWAVhashesverified67; final31rerunselectedWAVbyteexact so scoresreusedexplicitly. CERnative1.4803%/retained1.3758%;UTMOS2.53944/2.54357;SIM.69960/.69701. Closemetrics,noqualityimprovementclaim.

Stopping55: targetcost/representation/dependencyreview andrelatedprobescomplete; originalNL->normal1->normal2 dependenciespreserved, directparallelAVillegal. Rejectedextra policiespermedianobjective. Retainbestmigratedscheme, stopcostlysubpercentpursuit. NativeGEMMdominant;fourprotectedFFMAfullgraphcanonicalizationandseparatespecializedprofilesremainlowvalueunprovenleads. Notglobaloptimality/allpossibilitiesexhausted.
