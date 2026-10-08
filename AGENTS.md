# Current InSpark release

- Keep src/inspark_infer, configs/current, benchmarks, scripts, tests and reports/current layout.
- Publish current selected FP8 B1/B8/B64/B128, INT8 SmoothQuant B1/B8/B64 and newly authorized native NVFP4 B64 and reviewed NVFP4-GEMM/FP8-Conv mixed B64, plus the plain FP32 semantic reference. NVFP4 retains the current protected-role list; unsupported 16-bit/FP8 fallback must not be labelled native NVFP4.
- Draft is selected online step900; CFM is bilingual40k stage1 step800 with four quarter intervals. Keep checkpoint, per-component calibration and engine identities consistent.
- Keep weights, ONNX, engines, environments and caches out of Git. Pinned assets live in private xirr/index_pipeline.
- Preserve request-owned RNG, PCG residual law, committed-hidden isolation, KV ownership, EOS, cancellation and streaming outputs.
- Source code must not import another workspace project. Downloaded model assets are data dependencies.
- GPU experiments use one physical GPU at a time, GPU7 on this server, with serial builds and validation. No automatic time limit or stopping unrelated workloads.
- Existing approved kernels/plugins may migrate; the reviewed mixed B64 route is authorized; no new GPU math is needed for this route.
- Correctness checks follow operation logic; floating audits are reported without a fixed L2 acceptance threshold. Do not count repeated-text reuse as general performance.

## Independent ZipVoice A_1007 integration

- ZipVoice A_1007 targets INT8 B1/B2/B4/B8/B16/B32/B64, frames600/760/920 and tokens52/78/141. Read docs/zipvoice-a1007.md and reports/sm89/zipvoice/a1007/WHITEBOARD.md for current acceptance.
- ZipVoice uses its own Python3.12/Torch2.11+cu130/Triton3.6 environment and private xirr/zip_pipeline assets; do not merge it with the Index pipeline environment or registry.
- Only GPU1 is authorized for ZipVoice work. GPU7 directions above apply to Index pipeline work; this task performs no Index GPU experiments or new Index math.
- Preserve original SmoothQuant alpha=.5 weights/scales, first4 floating/last12 INT8 and complete unsplit model batch. Baseline plugin packages/engines remain intact during independent candidate validation.
- Complete all7 migration targets before the separately requested optimization phase. Validate private publication and fresh downloads before retiring old local/cloud ZipVoice files and obsolete history. Preserve the concurrently updated Index release.
- Engine binaries and model weights stay outside Git. No automatic time limit and no stopping unrelated workloads.
