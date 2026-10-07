# Current InSpark release

- Keep src/inspark_infer, configs/current, benchmarks, scripts, tests and reports/current layout.
- Publish only current selected FP8 B1/B8/B64/B128 and INT8 SmoothQuant B1/B8/B64 routes, plus the plain FP32 semantic reference.
- Draft is selected online step900; CFM is bilingual40k stage1 step800 with four quarter intervals. Keep checkpoint, per-component calibration and engine identities consistent.
- Keep weights, ONNX, engines, environments and caches out of Git. Pinned assets live in private xirr/index_pipeline.
- Preserve request-owned RNG, PCG residual law, committed-hidden isolation, KV ownership, EOS, cancellation and streaming outputs.
- Source code must not import another workspace project. Downloaded model assets are data dependencies.
- GPU experiments use one physical GPU at a time, GPU7 on this server, with serial builds and validation. No automatic time limit or stopping unrelated workloads.
- Existing approved kernels/plugins may migrate; this release task does not introduce new optimization routes or new GPU math.
- Correctness checks follow operation logic; floating audits are reported without a fixed L2 acceptance threshold. Do not count repeated-text reuse as general performance.
