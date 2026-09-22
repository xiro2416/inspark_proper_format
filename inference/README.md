# InSpark inference

Inference-only IndexTTS2: one model implementation, explicit eager/compile,
Triton/CUDA and TensorRT backends. Current hardware work is **RTX 4090 / SM89**.
No ZipVoice implementation or empty model scaffold is advertised.
[Architecture and compatibility](ARCHITECTURE.md).

## Status and evidence

Independent B1/B4 and retained B8 first-head profiles execute Target + Draft
backbone + Oracle500 CFM + BigVGAN through **TensorRT 11.3**. Prefill, proposal
RNN/sampling, latent/context processing and unsupported tails remain outside the
four engines. Seven-token Draft/eight-position verify, two-step CFM/CFG=0 and
44-frame first-PCM semantics are unchanged.

**TRT numerical equivalence has not passed.** Stage-one strict audits failed for
AR and Vocoder, and for B4 CFM. Direct/graph AR consistency and execution routing
passed. These profiles remain experimental, not accuracy-certified releases.
See [SM89 stage-one evidence](../reports/sm89/trt113_b1_b4_stage1/README.md) and
[engine/build details](../TENSORRT113_B1_B4.md). Stage-two real-input, quality,
compile and stability measurements are tracked in
[implementation progress](../IMPLEMENTATION_PROGRESS.md).
Actual eager/compile trials are in the [stage-two reports](../reports/sm89/stage2/README.md).
All-four compile and the B4 Draft/Vocoder subset fail sampled numerical gates;
only the measured B1 subset passed those boundary checks, not a general release
gate. The default remains pure FP32 eager. See [audit/reproduction commands](docs/SM89_AUDIT.md).

Five measured waves after two warmups on the task's shared GPU6 gave all-first-PCM
P50 **41.55 / 74.26 / 115.78 ms** for B1/B4/B8. This measures the legacy device-RNG
first-head route, not complete-EOS latency, an isolated-device guarantee, or the
new request-isolated route. Numerical failure is not waived by speed.

## Run

Linux/CUDA, local pinned model assets and one visible GPU are required. From the
repository root, `bash scripts/bootstrap.sh` sets up the primary environment;
`bash scripts/download_models.sh` downloads manifest-pinned assets. Use
`scripts/bootstrap_trt113.sh` for the isolated TensorRT 11.3 build environment.
Weights, ONNX/engines, audio and environments are local artifacts outside the
package, never included in Git.

Build source-attested independent acoustic engines with
`bash inference/scripts/build_trt113_acoustic.sh 6 1 audited_source` (repeat with
batch 4), and AR engines with `bash inference/scripts/build_trt113_ar.sh 6 1`
(repeat with batch 4). Set `ACC_TRT113_SITE` as described by the bootstrap script.
The `audited_source` variant preserves the original stage-one engine files.
The safe B1/B4 configs select these new plans; retained B8 artifacts have legacy
provenance and must not be described as source-attested merely because they load.

```bash
# Pure same-weight eager: full EOS, including runtime overhead.
ACC_TRITON_TOOLCHAIN=default bash scripts/run.sh \
  benchmarks/benchmark_reference.py --gpu 6 --batch 1 \
  --deployment configs/sm89_eager_fp32.json \
  --json-out outputs/eager_fp32_b1.json

# Offline Inductor warmup, then sealed shape dispatch on the same model.
ACC_TRITON_TOOLCHAIN=default bash scripts/run.sh \
  benchmarks/benchmark_reference.py --gpu 6 --batch 1 \
  --deployment configs/sm89_compile_bf16.json \
  --json-out outputs/compile_bf16_b1.json

CUDA_VISIBLE_DEVICES='' bash scripts/run.sh -m pytest -q tests
bash scripts/run.sh -m acc_infer_clear.cli --help
```

The runner changes directory to `inference/`. Editable installation after
bootstrap: `uv pip install --python .venv/bin/python --no-deps -e inference`.
`runtime_reference.yaml` disables TF32 before model/reference loading. The old
`sm89_fp32.json` uses custom kernels/graphs and is **not** the pure eager reference.

## Backend and request contracts

| Configuration | Arithmetic/execution | Request RNG |
| --- | --- | --- |
| `sm89_eager_fp32.json` | Same-weight FP32 PyTorch reference | Per-request legacy stream |
| `sm89_eager_bf16.json` | PyTorch BF16 matrix/conv/RNN, FP32 interfaces/norms | Per-request legacy stream |
| `sm89_compile_bf16.json` | Same BF16 callables; PyTorch 2.8 / bundled Triton 3.4 | Per-request legacy stream |
| `sm89_trt113_safe_b{1,4,8}.json` | Four native engines on supported shapes; isolated, numerically experimental | Per-request legacy stream |
| `sm89_bf16_trt113_full_b{1,4,8}.json` | Original fast device-controlled head, numerically experimental | Shared device stream; not isolation certified |

Compile components are independently selected in `compile_components`. Only
explicit offline warmup can compile; unseen signatures use eager. Compiler
errors are recorded and either raise or use the explicitly selected fallback.
Do not use isolated custom Triton 3.5 with PyTorch 2.8 Inductor. Existing custom
kernel and TRT plans are opt-in; shape/dtype/layout checks and counters expose
fallbacks. Each Engine owns execution contexts and temporary buffers. One
execution owner schedules each model; arbitrary concurrent host calls on one
Engine are unsupported. CLI/NDJSON and worker command loops serialize execution
while supporting multiple active requests.
`sm89_compile_bf16_draft_vocoder.json` selects only those two compile boundaries.
The reference benchmark writes its report before exiting: 1 means execution
failure, 2 means a measured compile numerical failure; `numerical_pass=null`
means timing-only, not an accuracy pass. Historical reports retain their original
exit behavior and explicit JSON gate results.

## Audit policy

`guardrails` supplies reusable tools; `tests` specifies cases; `benchmarks`
measures operators and actual end-to-end paths, including conversion, memory
traffic and launch costs. Gates: FP32 `atol=1e-5, rtol=1e-4`; BF16 arithmetic
`atol=1e-2, rtol=1e-2`, even with FP32 output storage. Nonfinite/missing results
fail closed. Quality is separate: 256 full-EOS pairs, UTMOS relative decline ≤3%
and character CER absolute increase ≤0.02.

Historical SM120 results/configs remain in
[HISTORICAL_SM120.md](HISTORICAL_SM120.md). They are not recertified for SM89 or
this layout migration. Do not rewrite old source hashes to bypass validation.
