# Source layout and backend boundaries

The installable package is `inference/src/acc_infer_clear`. This checkout contains
IndexTTS2 with the existing Universal DSpark flow and Oracle500 CFM student; it
does not contain a ZipVoice implementation. No placeholder model or HTTP service
is provided.

| Directory | Responsibility |
| --- | --- |
| `models/indextts2` | One model implementation: upstream modules, loading, references, DSpark math, CFM solver and audio/streaming semantics |
| `ops/eager`, `ops/matrix.py` | PyTorch reference arithmetic and unified prepared-matrix dispatch |
| `ops/triton`, `ops/cuda` | Optional compute kernels; CUDA source and attribution are wheel resources |
| `ops/tensorrt` | Existing TensorRT adapters and native 11.3 engine execution, not a separate model tree |
| `ops/planning` | Hardware/shape contracts, offline kernel-plan selection and cache identity |
| `quantization` | Weight conversion, FP8 packing, precision policy and offline representative-shape measurements |
| `runtime` | Engine, scheduling, slots, requests, cache and execution lifecycle |
| `api` | Existing CLI and NDJSON request protocol |
| `guardrails` | Reusable numerical comparisons and audit gates; scenarios remain in `tests` |

`acc_infer_clear.cli`, `acc_infer_clear.config` and
`acc_infer_clear.models.loader` retain thin public compatibility exports. Old
private implementation-module paths are not duplicated. Benchmark-only code is
under `benchmarks`, not the shipped runtime package.

## Precision and execution compatibility

| Route | Arithmetic/toolchain contract | Failure or fallback |
| --- | --- | --- |
| Pure eager | Same model/flow, FP32 or BF16; no project Triton kernel import required for BF16 matrix conversion/forward | Reference deployment rejects custom-kernel flags and FP8/auto precision |
| `torch.compile` | Same eager callables; PyTorch 2.8 and its bundled Triton 3.4, `ACC_TRITON_TOOLCHAIN=default` | Offline warmup only; unknown signatures run eager; compiler errors raise unless explicitly configured otherwise; counters expose both paths |
| Prepared BF16 matrices | FP32 interface with BF16 input/weight/bias computation and output cast back to input dtype, unchanged from the previous implementation | Explicit precision validation; grouped or nonzero-padding learned convolutions are unsupported |
| Prepared FP8 matrices | E4M3 per-output scales, padded `[K, ceil(N/32)*32]` weights; native-FP8 device and offline-selected Triton tile required | Unsupported hardware raises; no online tile search or implicit eager dequantization |
| Custom Triton/CUDA | Existing hardware-, shape- and source-bound plans; isolated custom Triton 3.5 where required | A historical plan is not recertified by moving its source files; mismatches must be rebuilt and audited |
| Native TensorRT 11.3 | Exact engine IO dtype/shape/batch and required plugin contract; `ACC_TRT113_SITE` selects isolated TRT without replacing main-environment NumPy/PyTorch | Existing explicit wrapper fallback/error behavior and execution counters remain observable; a fallback is not a successful TRT measurement |

The matrix adapters in `ops/eager/matrix.py` call the offline configuration
helpers in `quantization/precision.py` at construction, then route forward
execution through `ops/matrix.py`. The class identity is preserved for existing
`isinstance` checks. The old Triton `pack_weight` name is only a compatibility
export of `quantization/weights.py`; packing itself does not import Triton.

BF16 conversion retains the original rounding boundaries; it is not a claim of
FP32 equivalence. FP8 packing is not calibration or a quality audit. Numerical
and quality gates must be evaluated separately for the actual candidate.
The full preparation path still records the installed Triton version in its
plan-cache identity; use the PyTorch-bundled installation for eager/compile.
This metadata import is separate from executing project Triton kernels.

Target compilation uses a thin cache-format adapter around the original body:
it performs Transformers' own tuple-to-`DynamicCache` conversion before the
transformer call and restores tuple outputs afterward. This removes only the
legacy-cache deprecation logging branch; there is no global logger suppression
or model copy. CPU toy-GPT2 tests cover exact outputs, KV tensors, input-cache
nonmutation and fullgraph capture; actual-model Inductor numerics still require
their separate GPU audit.

## Resource paths and checks

Model weights, engine/ONNX artifacts, virtual environments and isolated toolchains
remain at repository root (`../models`, `../artifacts`, `../.venv`,
`../.toolchains` relative to `inference`). Config paths are resolved relative to
their configuration file. The runner changes working directory to `inference`;
its default script artifact paths therefore use `../artifacts`. Root
`scripts/run.sh` forwards to this runner. Do not relocate large artifacts into
the source package.

From repository root:

```bash
CUDA_VISIBLE_DEVICES='' bash scripts/run.sh -m unittest discover -s tests
CUDA_VISIBLE_DEVICES='' bash scripts/run.sh -m acc_infer_clear.cli --help
```

The source-layout CPU tests check public exports, external resource paths,
packaged CUDA sources/notices, unchanged BF16 reference arithmetic and offline
FP8 packing. CPU success does not establish GPU numerics, concurrent runtime
stability, model quality or speed. SM89 results must be identified separately
from historical SM120 results; consult the hardware-specific reports linked by
the README for actual evidence and limitations.
