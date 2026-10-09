# INT8 B32/760 throughput reoptimization

Goal: improve full unsplit B32/760 deployed E2E throughput on single GPU6/400W. User authorized GPU6 after GPU1 lock contention. SmoothQuant alpha=.5 first4 floating/last12 INT8, weights/scales unchanged. Row/path mapping primary, ordinary TF32/order differences allowed; CER/UTMOS/SIM-o auxiliary, no invented numeric threshold. Prepared CPU shaping/H2D through all32 orderedPCM; front-end/init/warmup/capture/WAV writes excluded. Throughput = completed rows over matched request wall span, including inter-request validation gaps.

Hardware: [RTX4090 SM89 profile](/workspace/.codex/skills/gpu_parameters/rtx4090-sm89-48g.md),128SM/72MiBL2/49140MiB,938GB/s measuredTriad. GPU6 400W configured; unmodified rawNVML50ms samples. Existing idle16206MiB allocation retained; no exclusive memory attribution.

Semantic reference: original eager/INT8 and ../quality-mapping-review.json. Original published bundle d1542a79cd19d660c8fa, b32_wp/a1007_delivery_graph. Earlier GPU1 median611.984ms/396.419W is historical only. Source original FM73.967ms; costly normal materialization, FFN projection/activation and nonlinear work were investigated. Strongest level5/FULL framework search already used.

Current best local candidate: static FM batch32/frame760 IO metadata, original ops/external weights/plugins, strongest TensorRT rebuild. Bundle270f2c741d495edac7b0, FM SHA f28711d5ebfe7bad0422dffb74c9dd71af4e657255e5cb826e50df63eb71b799. Default registry/publishedoriginal unchanged; explicit --bundle required, no remote upload this round. Dynamic texttoken52–141 unchanged. Original graph/sharedworkspace/arenaISTFT/transfer+PCMoverlap/staticweightpack and PCMchunk6/workers4 retained.

Final same-context ABBA control/candidate/candidate/control,30warm+100requests/block: control mean624.464ms/P99630.647/Q51.18309/power396.821W; candidate mean619.153ms/P99625.889/Q51.62462/power396.516W. Q +0.863%. Both candidate blocks faster than both controls; prior separate-context ABBA40 +.556% with one pair flat. Small measured gain, clock/power drift present; no crossGPU/longduration guarantee. NVML sampledmax original400.868/candidate401.827W is not configured powerlimit.

Validation: sevenfull32 cases (fourrefs,token52/141,mixedcomplete texts) preserve all input/path mapping, independent32row PCM, directgraph state/wave exact. Parameter/scope hashes match. Worker with explicitbundle passes; non760600/759/761/920 rejected. Fourref12WAV/route: CER3.419%→2.991%, UTMOS2.4916→2.4011, SIM.6525→.6393. Small objective metric declines disclosed; primaryCER0. No claimof qualityimprovement.

Rejected: FFNtile and nonlinearQKTF32 micro negative; K16 microbenefit failsE2E; nonlinear split initiallaunch missing2/3output invalidates+1.70% (history048). Corrected ASTgridQ64/z3, actualTRT380/760 all384exact/finite, E2E -.186%. DeepFFNfusion finite/exact minioutput but bestslow~79%; nofullbuild. CPUchunk4 -.257%,LUT -.997%, allselectedPCM identical; threadincrease micro slower. No newpublicruntime/kernel change retained. Isolated rejected source archived experiment/archived-src; .work retained.

Completion: retained original+candidate file/source hashes and deployedworker verified. Consequential tested fusion/materialization/distribution/CPU opportunities resolved negative under theseconditions; stop independent pursuit, no globaloptimum/power-onlycause claim. No outstanding requiredwork.

Reproduce: [result/commands](THROUGHPUT_REOPTIMIZATION.md); explicitworker candidate100requests GPU6. [Final audit](reoptimization/final-audit.json), [statistics](reoptimization/final-summary.json), [ABBA](reoptimization/run_steady_final.py). Detailed history043–050. Review only relevant history on resumption.

Publication follow-up: user authorized GitHub/HF update; B32 selected static760 bundle, prior dynamic bundle preserved on Hub; see reoptimization/publication.json. Optimization-stage no-upload statements above are historical.
