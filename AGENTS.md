# Codex deployment contract

This is an inference-only repository. SM89 TensorRT implementation, reports and
audits are in scope; preserve historical SM120 evidence separately. Do not add
training code or official model weights.

## Current entry points and reference path

- Entry point: `python -m acc_infer_clear.cli` through `scripts/run.sh`.
- Installable source: `inference/src/acc_infer_clear`; the runner changes cwd to `inference`.
- Reference runtime config: `inference/configs/runtime_reference.yaml` (TF32 disabled).
- CLI default deployment: `inference/configs/sm89_eager_fp32.json`.
- Experimental request-isolated TRT profiles: `inference/configs/sm89_trt113_safe_b{1,4,8}.json`.
- Original fast TRT profiles retain legacy shared device RNG and are not isolation-certified.
- Historical SM120 deployment: `inference/configs/sm120.json`; not recertified by migration.
- Model manifest: `inference/configs/model_sources.json`.
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

See `IMPLEMENTATION_PROGRESS.md`. First finish independent SM89 B1/B4
Target/Draft/CFM/Vocoder TensorRT 11.3 first-head profiles, then publish to
`xiro2416/inspark_proper_format`, reorganize and perform the full audit.
Do not stop after the first stage. Six hours is a target, not a hard stop.
Only physical GPU 6 is authorized for this task; run GPU work sequentially.
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
7. `configs/sm120_pre_device_commit.json` is the rollback for the device-control
   chain. The graph policy supports B1..B8/B16/B32; only sizes no larger than
   configured `max_batch` are prepared. Unsupported intermediate batches and
   longer KV must fall back rather than capture online.

## Porting to SM80/86

Use `PORTING.md`. Keep the pipeline and numerical contracts, but rebuild the
precision policy, tile plans and graphs. SM80/86 have no native FP8 Tensor Core
path compatible with this release, so begin with BF16/cuBLAS/cuDNN and measure
before introducing custom kernels. Never weaken checks merely to make the
SM120 JSON load.
