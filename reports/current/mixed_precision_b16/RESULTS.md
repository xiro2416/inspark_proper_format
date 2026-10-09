# B16 mixed pipeline optimization

## Final exclusive measurement

SingleGPU5, othertaskagentspaused; each5warm/30wave/480requests;15scontinuousE2E powerseparate. Alloutliers retained.

| Scheme | postadmission P50/P95/P99 ms | admission P50/P95/P99 ms | power P50/P95/P99 W | sampledpeak GiB |
|---|---|---|---|---|
| Migrationbaseline |108.64/120.38/127.34|109.82/121.44/128.37|223.41/401.57/406.89|13.745|
| Optimizedselected |105.83/117.49/119.21|106.97/118.60/120.39|219.12/411.24/417.79|13.726|

P50gain2.810ms/2.587%,P95/P99also improve; selectedretained. Matching480rows code/PCM SHA+accepted/round/generated/EOS arebitwiseequal. AllAR/CFM/Vocoderfallbacks0. Acceptance38.7299%both; physicalTarget371→364,Draft390→368,statusreads195→364. LowerpowerP50isnotacceptancecriterion. No extraresamplingoroutlierremoval. Finalselected_final_deployment.json differsfromexactformalinput selected_deployment.json only bystatus; calibration,engines,shape,schedulingidentical. Formalinputdeploymenthashescheckedagainstrecordedbenchmarkhashes. Detailsformal_summary.json.

The following short/profiler records are supporting history; formalresultsabove supersede preliminary performance estimates.

## Selected and evidence limit

Migration baseline eightB16engines remain unchanged. Retained existing mechanisms: readonly prefillTarget/contextviews andcontextoverlap, fusedTargetKVwrite, one-roundparentCUDA Graph, lateverify-only threshold10, CPU8workers. Precision135NVFP4/92FP8/90BF16 unchanged. CFMfull4steps/vocoderwholeengine retain L5/FULL/aux0; CFM56MiBL2,othersauto. No newauthoredGPUmath orquantization.

| Matched shortrun | P50 / P95 ms |
|---|---|
| baseline A1 |108.86 /114.21|
| selected B1 |106.77 /115.68|
| selected B2 |108.54 /156.60|
| baseline A2 |109.64 /119.34|

3warm10waves perrun, ABBAorder, sameGPU5/inputs/seed/boundary, no profiler or15spower. Baseline averageP50109.25→selected107.66ms,1.59ms/1.46% schemegain. Parallelagents were active elsewhere: B2P95outlier retained; subsequent finalexclusive baseline/selected30wave and15spower completedabove. Do not attribute screening113.49→105.63(6.9%) to a causal validatedgain.

## Correctness and actualroute

All11scheduling/controlvariants (160requests each) matchbaseline case/seed, codeSHA, PCM SHA, acceptedpositions, logicalrounds, generatedcodes andEOS exactly; zeroAR/CFM/Vocoder fallback. Thus the changespreserve completeheadwork/request ownership and actualoutputs in this matchedworkload. Existingmigration floatingaudit remains applicable to unchangedengineweights/graphs; floating report in migration_float_audit.json, noL2gate orCER/MOS certification. Selected2realwaveprofile confirmsARnative workergraph,32GPUcoderows,zeroCPUcodes,CFM/Vocograph2hits,prefill/latent2hits,zero fallback.

Nativecoverage in actual_execution.json: originalmixed8engines preserved, Target72/Draft12/context4/prefixes72/CFM168F4 GEMM;CFM64F8Conv;Vocoder72F8correlation+109FP32FIR. Fourlowprecisiondeconvs executeFP32 FFMA as documentedsourcegap; notclaimednativeF8. CUDA Graph is now onephysicalspeculative round perreplay, nottwo; RNN7stepstate0andGPU PCG/commit unchanged.

## Focused optimization record

- prefixviews+overlap screening113.49→111.28ms,KVfusion111.28→108.73ms; retained ascompoundcandidate withmatchedABBA evidence.
- CPU16workers113.25ms, noimprovement overCPU8 candidate108.73; rejected.
- late10thenburst1 screening107.97→105.63ms, savedfinalproposals and alteredsubmission granularity. late12+burst1 cheapcontrol107.08ms; late10 retainedprovisionally, noisolatedcausalclaim.
- burst4=108.94ms, physically125Target/132Draftvsburst1 121/122, rejected.
- FP32deconv total0.394ms/wave inbaselinekerneltrace; small0.37%E2Eupperbound. Noheavy graphbuild/newkernel pursued solelyfor this.
- ExistingFIR6shapes×8block/warp configurations: original256/4 fastest inall6shapes, weighted7.882→7.882ms; outputsbitwiseequal forallschedules. NoFIRparameterchange.

Selected diagnosticlast-wave scope timings: text13.16,prefill13.01,AR56.37,latent5.05,CFM17.54,Vocoder18.14ms. Instrumented, overlapping/dependency-sensitive spans: do notsumtheseasproductionE2E orattributeCPUtimechange to oneoptimization. Trace artifacts baseline_profile.trace.json andselected_profile.trace.json.

## Handoff and stop

Deployment artifacts/mixed_b16/selected_final_deployment.json; runtime.yaml; same8enginepaths asvalidatedmigration. Commands/environment in migration_queue.py, optimize_queue.py, confirm_queue.py, profile_selected.py; allsource+artifacthash records preserved. GPU5released0MiB/0%. Finalselected artifacts/mixed_b16/selected_final_deployment.json. NoGit/HFpublication byworker.

Residuals: FP32deconvprecisiongap, nativequantization/reformat/norm cost,FIRcompute andhoststatus synchronization remain. TailB8 mayreduceonlylate1–3rounds and addsKV/RNGproposalstate transfer androutingcost; noB16tail/stream/serialsplitbenefit claimed. Remaining confirmedpaths arefewmilliseconds; endindependentsearch peruserstopcondition and handoff finalverifiedvariant toparent. Notglobal optimum.
