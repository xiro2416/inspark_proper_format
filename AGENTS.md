# Codex deployment contract

This is the inference-only release. Do not search for training code, experiment
reports or profiler traces; they are intentionally absent.

## Current production path

- Entry point: `python -m acc_infer_clear.cli` through `scripts/run.sh`.
- Runtime config: `configs/runtime.yaml`.
- Deployment config: `configs/sm120.json`.
- Model manifest: `configs/model_sources.json`.
- Universal Draft proposes seven codec tokens; Target verifies eight positions
  including the anchor and accepts a prefix.
- Fixed first-head graph batches use device-resident acceptance-prefix,
  residual, commit and Context-slot state when KV<=128. One all-ready/fallback
  scalar remains at each round boundary. The full parent-round Graph is disabled.
- CFM is the hash-pinned Oracle500 derivative and always uses two intervals
  `[0,.5]` and `[.5,1]` with CFG disabled.
- First output is 44 acoustic frames with eight frames of right context.
- No torch.compile, TensorRT, vLLM or online autotuning is used.

## Deployment rules

1. Run `scripts/bootstrap.sh`, then `scripts/download_models.sh`.
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
