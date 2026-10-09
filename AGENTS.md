# Current InSpark release

- Keep src/inspark_infer, configs/current, benchmarks, scripts, tests and reports/current layout.
- Publish current selected FP8 B1/B8/B64/B128, INT8 SmoothQuant B1/B8/B64 and newly authorized native NVFP4 B64 and reviewed NVFP4-GEMM/FP8-Conv mixed B64/B128, plus the plain FP32 semantic reference. NVFP4 retains the current protected-role list; unsupported 16-bit/FP8 fallback must not be labelled native NVFP4.
- Draft is selected online step900; CFM is bilingual40k stage1 step800 with four quarter intervals. Keep checkpoint, per-component calibration and engine identities consistent.
- Keep weights, ONNX, engines, environments and caches out of Git. Pinned assets live in private xirr/index_pipeline.
- Preserve request-owned RNG, PCG residual law, committed-hidden isolation, KV ownership, EOS, cancellation and streaming outputs.
- Source code must not import another workspace project. Downloaded model assets are data dependencies.
- GPU experiments use one physical GPU at a time, GPU7 on this server, with serial builds and validation. No automatic time limit or stopping unrelated workloads.
- Existing approved kernels/plugins may migrate; the reviewed mixed B64/B128 route is authorized; no new GPU math is needed for this route.
- Correctness checks follow operation logic; floating audits are reported without a fixed L2 acceptance threshold. Do not count repeated-text reuse as general performance.

## Independent ZipVoice A_1007 integration

- ZipVoice A_1007 targets INT8 B1/B2/B4/B8/B16/B32/B64, frames600/760/920 and tokens52/78/141. Read docs/zipvoice-a1007.md and reports/sm89/zipvoice/a1007/WHITEBOARD.md for current acceptance.
- ZipVoice uses its own Python3.12/Torch2.11+cu130/Triton3.6 environment and private xirr/zip_pipeline assets; do not merge it with the Index pipeline environment or registry.
- Only GPU1 is authorized for ZipVoice work. GPU7 directions above apply to Index pipeline work; this task performs no Index GPU experiments or new Index math.
- Preserve original SmoothQuant alpha=.5 weights/scales, first4 floating/last12 INT8 and complete unsplit model batch. Baseline plugin packages/engines remain intact during independent candidate validation.
- Complete all7 migration targets before the separately requested optimization phase. Validate private publication and fresh downloads before retiring old local/cloud ZipVoice files and obsolete history. Preserve the concurrently updated Index release.
- Engine binaries and model weights stay outside Git. No automatic time limit and no stopping unrelated workloads.

## Current authorized SM120 FP8 work

- User-approved SM120 FP8 work uses only physical GPU3, serially, at its existing 600W limit. The historical GPU1/INT8 instructions above describe the preserved SM89 release.
- Build only FP8 targets B1/B2/B4/B8/B16/B32/B64: FM first4 floating/last12 eligible Linear/Conv W8A8 E4M3 with frozen per-tensor max calibration. Text/Vocos and sensitive operations stay floating.
- Read reports/sm120/zipvoice/fp8/WHITEBOARD.md before experiments. Complete seven migrations before new optimization.
- Use independent bilingual calibration and test data; keep original model/checkpoint, duration and full unsplit batch. Preserve SM89 hashed source closures and Index routes.
- Publish by normal GitHub commits and additive private HF sm120/fp8 paths, without history rewrite or asset deletion. Credentials must never enter tracked files or logs.

## Authorized B128 continuation

- User requests ZipVoice SM120 FP8 B128 migration first, then optimization. Keep frozen model/scales/backend and unsplit batch, single GPU3 at existing600W.
- B128 uses independent namespaces and writable cache copies; preserve published seven-batch source closures. Read reports/sm120/zipvoice/fp8/b128/WHITEBOARD.md before experiments.

## Authorized parallel Index mixed migration

- Latest user authorization: mixed NVFP4 GEMM/FP8 Conv B4/B16/B32 migration first, optimization second, using three agents concurrently. This overrides the older GPU7-only serial rule for this task. B4 owns physicalGPU4, B16 GPU5, B32 GPU6; each experiment sees and uses oneGPU. Preserve external workloads.
- All three migrations must validate before optimization begins. Final baseline/selected measurements run in coordinated exclusive windows while other agents pause GPU work.
- Separate worktrees, artifacts and writable caches. All worktrees share /workspace/.cache/inspark/gpu-locks; do not use sharedGPU escape hatches. Frozen source checkpoints/calibration/engines are readonly.
- Parent integrates common code, deployment registry and Git/HF releases. Workers do not publish. Existing90BF16 protection and4stepCFM remain; floating audits are reporting-only. No repeated-textcompute reuse.
