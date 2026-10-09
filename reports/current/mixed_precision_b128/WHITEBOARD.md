# Mixed B128 current best

Source reports/current/mixed_precision; target same checkpoints,backend,precision recipe and GPU7. /workspace/.codex/skills/gpu_parameters/rtx6000d-sm120.md:156SM,112MiB L2,83GiB,600W; TRT11.3.0.99/Torch2.13 CUDA13.2. No repeated-textcompute reuse,no fixedL2gate,no artificialbuildtimeout.

Migration completed:8 exactB128 engines,original mixed quantization135/92/90,full4-stepCFM/directFP8Conv/109FP32FIR; KV80,slot/headmajorPCG/commit andGraph restored;latent-prefix precision label whitelist adapted without removing hashes/IOguards. Baseline audit128uniquePCM/finite,zero fallback.

Selected: CFM64 full4step×2 serial(FULL,L2=56MiB) andVocoder64 wholeengine×2(MODERATE,autoL2),snapshotoutputs beforecontextreuse; graphs stillserveB128. Prefill/context readonlyviews+overlap; CPU32;existingKVfusion. AR128→64→8 afteroriginal10/13 eligibility,exactmatchingnewbucketengines L5FULL,actualactive18then1,requestKV/RNGrestored. BF16firstquarterandoutsideprecisionunchanged. Native_coverage accounts16engines,customFIR/KVexplicit,no newGPUmath.

Matched5warm/30waves+15spower: baselineP50559.03→selected493.89ms(11.65%),P95497.27,P99498.46. PowerP50/P95/P99=413.30 / 506.42 / 513.60W. ZeroAR/CFM/Voco fallback;selected128uniquePCM,no saturation,CFM same-recipeL2~1.16%,Voco~9.60%,ARallfields reportonly. Peakboard~31.64GiB.

IndependentCFM+Voco CG overlap probe128.2serial vs135.0parallel,bitwiseequaloutputs,notretained. Voco64FULL finalbuildcompare inexperiments;no broadclaimfromnegprobe. Profiletext21.8/prefill57.4/AR168/latent27.4/CFM105.4/Voco156.6ms(nonadditive diagnostic). FIRandprotectedmathremaincostly,deeperdesignrequiresnew evidence,lastconfirmedARgain~4ms;6FIRshapes×8schedulesallcurrent256/4best,weighted31.14msB64,outputsidentical;stopfurtherindependentpursuit. Notglobal optimum orCER/MOScertification.

Reproduce: benchmarks/benchmark_unified_first_chunk.py run --gpu7 --batch128 --deployment artifacts/mixed_b128/selected_deployment.json --config artifacts/mixed_b128/runtime.yaml --manifest /workspace/A_0924/artifacts/sm120_0924/unified/cases.json --warmups5 --waves30 --power-seconds15 (flagsseparatednormally). Localhistory/artifacts maintained. Selected bundle SHA checks/materialize/3 actual B128 waves passed,zero fallback; publication last step.
