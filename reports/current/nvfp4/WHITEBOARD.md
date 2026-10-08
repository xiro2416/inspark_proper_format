# NVFP4 B64 current release

Goal: latest Draft900/CFM800,four steps,90BF16/227nativeNVFP4 learned roles,operation logic preserved. GPU7 serial; RTX6000D SM120/156SM/L2=112MiB/600W; hardware reference /workspace/.codex/skills/gpu_parameters/rtx6000d-sm120.md. TRT11.3.0.99,ModelOpt0.47.0.

Original NVFP4 P50=498.01ms. Selected corrected NVFP4:30waves P50/P95/P99=357.12/361.93/362.20ms,power=406.31/493.67/501.17W. Same-period FP8=343.33/347.30/347.75ms. Improve NVFP4 P50 by28.29%;still4.02% slower thanFP8. No text-dedup gain; no AR/CFM/Vocoder fallback.

Retained: single whole-window Gather per learned convolution,static indices,existing complete-halo FP32 FIR fusion with explicit FP32/LINEAR ABI,official Target Attention(original BF16 Q/K/V andmask,scale1,outputBF16 rounded thenFP32),existing static GC guard. All required low-precision GEMMs nativeFP4; staticKV80 anddevice-round/KV/Graph scheduling preserved. Full four-stepCFM oneengine L5/FULL/L2=56MiB.

Correctness: dtype ABI bug in first FIR candidate caused saturated PCM; all results of that candidate were withdrawn. Corrected109plugin IO contracts inspected, BF16-upstream/two-shape probe exact, real64head outputs distinct andfinite. No fixedL2gate; float audit reports substantial NVFP4 differences, no CER/MOS/statistical certification. GC619.9ms pause independently explained prior slowwave; existing static-object freeze enabled, normal requestGC retained.

Detailed current results:RESULTS.md,summary.json,float_audit.json. Local experiments under artifacts/nvfp4_b64; selected deployment fir_fixed_deployment.json,finaltest fir_fixed_final.json,profile fir_fixed_profile.json. Correct Gather+Attention comparator P50=414.24ms. Portable bundle reconstructed-weight smoke3waves passed.

Rejected routes/searches: FIR-up pointwise slower; tap-major onlytiny benefit; nativeTRT-LLM quantizer adaptation bit-exact but no gain andnoGraph capture; aux2 actual0 andFull/L2variants producedsame priorcandidate binary. Do not claim global optimum or unresolvedframework limits solved. DirectFP4Conv not inTRT11.3 support table; explicit window/quantization overhead remains. FP8default retained; NVFP4B64 independent asset pin.

Rawlogs, withdrawncandidates and investigation history remain local; current public code/assets include only selected implementation. Publication status recorded in artifacts/nvfp4_b64/publication.json after remote hash verification.

Published HF revision d3fa9e80e2997a7007c839c07a386297141f6b6a,52assets(remote file size/LFS hashes verified). Independent Target/Vocoder download hashes match;3actualB64waves pass reconstructed-model/bundle route,zero fallback.21CPU geometry/mask/identity/API tests pass.
