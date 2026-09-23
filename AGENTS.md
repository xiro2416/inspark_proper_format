# Codex deployment contract

This is an inference-only repository. SM89 TensorRT implementation, reports and
audits are in scope; preserve historical SM120 evidence separately. Do not add
training code or official model weights.

For the SM89 selected-shape TensorRT builder and future multi-SM extension,
follow `docs/trt-build-for-codex.md`. The CLI exists, but successful build,
numeric certification and performance remain separate evidence requirements.

## Current entry points and reference path

- Entry point: `python -m inspark_infer.cli` through `scripts/run.sh`; `acc-clear` remains a CLI alias.
- Installable source: `src/inspark_infer`; the runner works from the repository root.
- Reference runtime config: `configs/common/runtime_reference.yaml` (TF32 disabled).
- CLI default deployment: `configs/hardware/sm89/sm89_eager_fp32.json`.
- Experimental request-isolated TRT profiles: `configs/hardware/sm89/sm89_trt113_safe_b{1,4,8}.json`.
- Original fast TRT profiles retain legacy shared device RNG and are not isolation-certified.
- Historical SM120 deployment: `configs/hardware/sm120/sm120.json`; not recertified by migration.
- Model manifest: `configs/common/model_sources.json`.
- Universal Draft proposes seven codec tokens; Target verifies eight positions
  including the anchor and accepts a prefix.
- Fixed first-head graph batches use device-resident acceptance-prefix,
  residual, commit and Context-slot state when KV<=128. One all-ready/fallback
  scalar remains at each round boundary. The full parent-round Graph is disabled.
- CFM is the hash-pinned Oracle500 derivative and always uses two intervals
  `[0,.5]` and `[.5,1]` with CFG disabled.
- First output is 44 acoustic frames with eight frames of right context.
- Backends are explicit: reference PyTorch, offline torch.compile, project
  Triton/CUDA kernels and TensorRT 11.3 where the selected profile supports it.
  Serving must not compile, capture graphs or autotune new shapes online.

## Current implementation goal

The previous B1/B4 milestone and SM89 audit are preserved under `reports/sm89/`.
The active two-stage goal is: finish the root-package migration, then deliver a
single-GPU SM89 B1/B4/B8 fixed-first-chunk TensorRT 11.3 bundle builder with
audits and an explicit private Hugging Face artifact cache. Publish only to
`xiro2416/inspark_proper_format` after verification. Do not stop after the
first stage. Six hours is a target, not a hard stop.
Only physical GPU 4 is authorized for this task by the latest user direction; run GPU work sequentially.
Existing external GPU processes must remain untouched. Explicit shared-GPU
runs must record pre-existing memory and workload interference.
All task files, caches and locks belong under `/workspace`.

## Deployment rules

1. Run `scripts/bootstrap.sh`, then `scripts/download_models.sh`; downloading models must not auto-install historical hardware plans.
2. Never upload or vendor official GPT/S2Mel/BigVGAN/W2V-BERT/MaskGCT/CampPlus
   weights. Download them from the pinned upstream revisions.
3. Use exactly one physical GPU per test. `--gpu` is a physical GPU index and
   the runtime takes a cooperative lock before initializing CUDA.
4. SM120 plan files are measured choices, not portable formulas. The preflight
   must reject another architecture or changed source/software/model identity.
5. Do not silently tune during request admission. Any new-device plan must be
   prepared and validated offline, then versioned explicitly.
6. Preserve request-owned KV, accepted-prefix/EOS behavior, stream ordering,
   reference VAD policy and the 44-frame first PCM contract.
7. `configs/hardware/sm120/sm120_pre_device_commit.json` is the rollback for the device-control
   chain. The graph policy supports B1..B8/B16/B32; only sizes no larger than
   configured `max_batch` are prepared. Unsupported intermediate batches and
   longer KV must fall back rather than capture online.

## Porting to SM80/86

Use `docs/porting.md`. Keep the pipeline and numerical contracts, but rebuild the
precision policy, tile plans and graphs. SM80/86 have no native FP8 Tensor Core
path compatible with this release, so begin with BF16/cuBLAS/cuDNN and measure
before introducing custom kernels. Never weaken checks merely to make the
SM120 JSON load.
