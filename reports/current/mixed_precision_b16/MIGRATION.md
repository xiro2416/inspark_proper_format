# B16 mixed pipeline migration

## Baseline

RTX6000D GPU5; same latest Draft900/CFM800 weights and original mixed calibration135 NVFP4 /92 FP8 /90 BF16. Eight preciseB16 engines rebuilt, all builder5/FULL/aux0/tacticsauto/timing enabled. CFM56MiB L2; otherL2auto. Exact engine hashes/static IO and allmodel-source hashes verified.

- TargetQ8/DraftQ7/KV80, native NVIDIA DSparkWorker/RNN bridge +普通TRT compute +GPU PCG；不是完整TRT-LLM Executor。
- Context、prefill48、latent80、latent_suffix40(prefix48)形状全部B16。
- Full4-stepCFM singleenqueue F310/P258; no droppedattention/conv prompt context.
- WholeVocoder engine directFP8Conv,109 inheritedcustomFIR/Snake/FIR plugins, allFP32 ABI. No newGPU math.
- Slot/headmajorKV、accepted-onlycommit、RNN每round state0、两轮父CUDA Graph、prefix/latent reuse、CPU8继承。没有tail/microbatch/textcompute dedup。

## Integration evidence

3warm/10waves/160requests migration diagnostic, concurrent agents exploring other GPUs; not finalexclusive measurement.

| Boundary | P50 / P95 / P99 ms |
|---|---|
| 受理完成→整波最后PCM |106.96 /115.15 /116.50|
| 含受理→整波最后PCM |108.11 /116.19 /117.54|

AR/CFM/Vocoder zero fallback;160rows GPUcodes, zero speech-codehostrows; CFM/Vocoder each10Graph hits andprefill/latent each10hits. Real acoustic head16uniquePCM,finite,no saturation. Acceptance38.91%, roundP50/P95/P99=8/12/13, KVhead71/76/77. Power15s and5warm/30wave finalmeasurement pending parent-coordinatedexclusive window.

## Precision coverage and floating audit

Target72/Draft12/Context4/Prefill72/Latent72/Suffix72 nativeF4GEMM implementationlayers；CFM168F4GEMM+64nativeF8Conv；Vocoder72nativeF8correlation+109FP32FIR。Actual implementation counts do not equal logical recipe role counts.

Important inheritedgap: lowprecisionrole vocoder.stages.2–5.ups.0 declaredFP8QDQ, but4ConvTranspose actualFP32 FFMA tactic/IO. SourceB64 has samegap; do not claim all76lowvocoderroles are nativeFP8. FullactualIO/tactic in migration_summary.json. Phase2 profile and graphrewrite may repair this.

Same-recipe relativeL2 reporting: Target logits10.4508%, CFM1.3150%, Vocoder9.5176%; unquantized CFM1.7835%, Vocoder18.7638%. NVIDIA RNN adapter onidenticalTRThidden proposalq relativeL2=0.0001061%. Operation logic validation uses nofixedL2gate. No CER/MOS certification. Loaderroute metadata `weight_identity_verified:false` retained as source metadata; explicitbuild/model-source SHA checks verified all8 against source, plusARreference checkpoint identities.

## Handoff

Migration complete, GPU5released. Next optimization: profilecosts; assess4FP32deconvs, frameworkfusion/layout/memoryconversion, smallbatchtactics/FULL-versusMODERATE/L2/actualauxstreams, bursts1/2/4 andCPUworkers, prefill/contextoverlap. TailB8reuse only afterfullrecipe/model/IOsemantic checks and matchedE2E; paths alone not identity. B128serialB64 acoustic outcomes not applied toB16.

Reproduce `artifacts/mixed_b16/migration_queue.py`, then `validate_queue.py`; environment andcommands recorded there/status. Build artifacts/sourcepreserved; no publication. Detailedhistory in artifacts/mixed_b16/history.
