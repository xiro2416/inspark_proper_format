# InSpark Marlin

Inference-only, first-packet-oriented IndexTTS2 runtime with Universal DSpark,
batched Target verification, native FP8 kernels, explicit CUDA Graphs and a
two-step distilled CFM. This repository does **not** contain training code,
datasets, profiling reports or official IndexTTS2 weights.

The published execution plans are validated only for NVIDIA RTX 6000D
(SM120, 156 SMs), CUDA 12.8, PyTorch 2.8.0+cu128 and Triton 3.5.0. Do not copy
the SM120 tile plans to SM80/86; see [PORTING.md](PORTING.md).

## Pipeline

```text
incremental text
  -> first segment splitter
  -> GPT prefill
  -> Universal Draft backbone + 7-token RNN Proposal
  -> Target verifies 8 positions and accepts a prefix
  -> repeated AR rounds until the first acoustic unit is ready
  -> latent/acoustic conditioning
  -> Oracle500 CFM, 2 steps [0, .5, 1], CFG=0
  -> BigVGAN
  -> 44-frame first PCM chunk, then streaming tail
```

The graph policy can represent B16/B32 for legacy experiments, while the shipped
Unified AR deployment is validated and explicitly limited to B1..B8. The runtime
defaults to `max_batch: 8`; selecting a larger batch with the Unified AR config is
rejected rather than silently falling back. The
first-packet Prefill and Latent routes use logical 48/80 buckets; overflow and
variable tails remain explicit fallback paths. The default admission batch is
8 and can be changed with `--batch`.

For a fixed first-head graph batch with KV no longer than128, the current
runtime keeps acceptance-prefix decisions, residual sampling, token/length/EOS
commit and Draft Context slot scatter on the GPU. The host reads one compact
all-ready/fallback status per AR round; a rare device residual fallback restarts
that group through the prior exact path. Tail, unsupported intermediate batches
and longer KV retain the established implementation. The complete parent-round
CUDA Graph candidate was slower and is not enabled.

### Unified AR

The SM120 default uses one fixed backend per semantic AR role; batch size no
longer selects cuBLAS, compiler Triton, explicit-pipeline or Full-M backends.

- Target M8..64 and Draft M7..56 share one combined-QKV kernel family with
  BN32/BK128 and a fixed two-stage shared ring. Draft always performs one QKV
  GEMM, not three projections.
- Deep Q/K and persistent K caches use E4M3 with two block32 scales per head.
  QK uses native FP8 MMA with FP32 accumulation. V cache, Softmax and P×V remain
  FP32; the protected shallow quarter remains FP32 throughout.
- Target K and Draft Context K write directly to request slots. Attention writes
  the Out-GEMM consumer layout directly.
- Out, Up and Down each use one fixed role-specific backend/schedule across all
  B1..B8 shapes. Their norm, GELU and residual boundaries remain explicit.
- Initial Prefix K conversion is one precompiled direct-slot kernel. Serving
  performs no Triton compilation, CUDA Graph capture or torch.compile call.

Same-process alternating tests on one RTX6000D measured first-PCM P50 changes of
32.53→31.48ms (B1), 60.91→58.52ms (B4), and 74.74→71.18ms (B8). A 256-stream
quality gate measured UTMOS -0.31% and CER +0.446 percentage points. Final custom
kernels passed memcheck; the shared-pipeline Target/Draft replays passed racecheck.
These measurements are workload-specific, not service guarantees.

Roll back the complete Unified AR update with `configs/sm120_pre_unified_ar.json`.

The selected device path was validated on256 full streamed utterances against
the previous release: UTMOS changed by -1.38% and CER by +0.00893. On the
published RTX6000D setup, sustained B8 first-PCM P50/P95 changed from about
91.23/100.39ms to73.10/81.14ms. This is a hardware/workload-specific result,
not a latency guarantee. Roll back only the device-control update with
`configs/sm120_pre_device_commit.json`.

## Install

Prerequisites: Linux, an RTX 6000D/SM120 GPU, a CUDA 12.8-compatible driver,
CUDA 12.8 toolkit/NVCC, a C++ compiler, `git`, `ffmpeg`, and
[uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/xiro2416/inspark_marlin.git
cd inspark_marlin
bash scripts/bootstrap.sh
bash scripts/download_models.sh
```

`download_models.sh` downloads immutable, hash-pinned files from their
original repositories and only downloads project-derived weights from
`xirr/inspark_marlin`. It then installs the measured SM120 plan mappings under
the local GPU identity. It does not benchmark or tune kernels.

## Generate one streamed request

```bash
bash scripts/smoke_test.sh /absolute/path/to/reference.wav
```

Or call the CLI directly:

```bash
bash scripts/run.sh -m acc_infer_clear.cli \
  --gpu 0 --workers 1 --batch 1 \
  --config configs/runtime.yaml \
  --deployment configs/sm120.json \
  --ref-audio /absolute/path/to/reference.wav \
  --text '他正在整理文件。' \
  --output outputs/example.wav
```

The reference is VAD-cropped to at most three seconds and cached locally.
`--emotion` accepts eight floats. For a long-lived service, keep one process
and model instance alive; initialization loads all weights, compiles the CUDA
extensions and selected device-state kernels, then captures the configured
graphs. Online compilation and capture are forbidden after preparation.

## Incremental protocol

Use `--stdin-stream` and send one JSON object per line. Supported operations
are `open`, `text`, `end`, `run`, `tick`, `drain`, `cancel`, and `release`.
Audio events are emitted as NDJSON with base64 PCM S16LE at 22,050 Hz. Read
[`src/acc_infer_clear/cli.py`](src/acc_infer_clear/cli.py) for the exact compact
wire contract.

## Model provenance

Official weights are never mirrored here:

- `IndexTeam/IndexTTS-2`: GPT, S2Mel, BPE and conditioning assets.
- `facebook/w2v-bert-2.0`: speech encoder.
- `nvidia/bigvgan_v2_22khz_80band_256x`: vocoder.
- `amphion/MaskGCT`: semantic codec.
- `funasr/campplus`: speaker model.

Only the Universal DSpark, ASG and Oracle500 two-step CFM derivative weights
live in `xirr/inspark_marlin`. Exact revisions, destination paths and hashes
are in [`configs/model_sources.json`](configs/model_sources.json).

This is a derivative work and is not endorsed, warranted or guaranteed by the
original IndexTTS2 right-holder. The original right-holder disclaims liability
for modifications in this derivative. Read [LICENSE](LICENSE) before use.
