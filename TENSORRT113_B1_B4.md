# SM89: independent B1/B4 TensorRT 11.3 profiles

This is the first implementation milestone, not the complete repository audit.
The original local B8 source snapshot is preserved in commit `813eb3d`.
Historical SM120 results in the original README are not evidence for SM89.

## Scope

One Engine owns its TensorRT contexts, graph buffers and KV pools. B1 and B4
have independent fixed-batch engines and deployment profiles; B8 is retained.

| Component | TRT engine boundary | Limits |
| --- | --- | --- |
| Target | 24 blocks, final normalization and logits | Verify 8; KV128; identity slots |
| Draft | Three-layer backbone, projection and base logits | Propose 7; context KV128; identity slots |
| CFM | Oracle500 two-step solver, CFG0 | Prompt258 + generated52; 80 mel channels |
| Vocoder | BigVGAN with in-engine alias-free and deconvolution plugins | 52 mel frames; FP32 interfaces, BF16 learned convolutions |

“All TRT” describes these engine boundaries. It does **not** mean every operation
is a native TensorRT kernel: Vocoder plugins use existing Triton/PyTorch compute.
Prefill, proposal RNN, sampling, context commit, conditioning and streaming tails
remain outside these engines. Out-of-range shapes/slots use explicit fallback;
corrupt hashes, incompatible engine IO and enqueue failures raise errors.
The canonical context pool retains long histories; compact KV128 mirror writes
are bounded and the device loop yields before crossing its capacity.

## Reproduce

Install the base environment and pinned model files using the existing bootstrap
and download instructions, then install isolated native TRT 11.3:

```bash
bash scripts/bootstrap_trt113.sh
export ACC_TRT113_SITE="$PWD/.venv-trt113/lib/python3.11/site-packages"
# These commands are sequential and use only the selected physical GPU.
bash scripts/build_trt113_ar.sh 6 1
bash scripts/build_trt113_acoustic.sh 6 1
bash scripts/build_trt113_ar.sh 6 4
bash scripts/build_trt113_acoustic.sh 6 4
bash scripts/run.sh scripts/validate_trt113_first_chunks.py \
  --gpu 6 --batch 1 --deployment configs/sm89_bf16_trt113_full_b1.json \
  --reference /absolute/path/to/reference.wav \
  --output outputs/trt113_full_b1_first_chunks.json
```

For B4 change both batch and deployment profile. The reference must yield the
fixed P258 prompt; another prompt length may run through fallback but cannot
pass the strict all-TRT first-chunk gate. Engine/ONNX files, model weights and
audio are local artifacts, not Git payloads. Rebuild on the target device;
hash-pinned plans are not portable performance guarantees.

The default GPU lease requires an idle device. This task explicitly used
`ACC_GPU_ALLOW_SHARED=1` on physical GPU6, preserving 21,854 MiB of pre-existing
allocations. Therefore the measurements are not dedicated-device guarantees.

## Audit interpretation

The first-chunk validator checks actual native Target/Draft execution counts and
the captured CFM/Vocoder route at replay, not merely the configured backend.
Module validators compare native direct and graph execution with same-input pure
PyTorch FP32/BF16 references. BF16 arithmetic may retain FP32 tensor interfaces.
Numerical gates use `atol=1e-2, rtol=1e-2` for BF16 and `atol=1e-5, rtol=1e-4`
for FP32; these declared policies are not fitted to candidate errors.

Cache/metadata invariants are exact. Shape mismatch, missing evidence, empty
output and NaN/Inf fail closed. A numerically failed candidate is not promoted
by a good cosine score or single-operator speedup. First-chunk routing, floating
point, full-EOS quality and sustained-load results are separate claims.

Initial legacy-AR first-chunk smoke: B1 P50 42.564 ms (five requests), B4 all-ready
P50 74.236 ms (five four-request waves), with all four TRT routes and no fallback.
These are preliminary routing results before source-attested AR rebuilds, not
numerical or quality passes. The subsequent reports supersede these timings.

The new builders record actual checkpoint/config and converted-constant hashes.
Old acoustic artifacts without export provenance remain `legacy_unverified`;
later hashing current weights does not retroactively attest their origin.

Implementation references: [TensorRT precision control](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/precision-control.html),
[engine compatibility](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/engine-compatibility.html).
