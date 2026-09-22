# Portable Planner V2

Planner V2 separates optimization policy from model execution. It describes
hardware, tensor shapes, semantic operator roles, layouts and schedules in a
versioned manifest. The shipped SM120 deployment remains unchanged until a V2
manifest passes the same component, service and quality gates.

## Selection contract

- One semantic role has one backend and primary global layout for a given
  device/model manifest. Shape buckets may select different tiles, warp counts
  and shared-memory stages.
- SM80/86 use BF16 Tensor Core candidates. SM89 and later architectures may use
  native FP8. Norm, Softmax and accumulation remain FP32 unless separately
  validated.
- Formulae prune candidates; a target-GPU offline calibration selects them.
  Serving performs no tuning, compilation or graph capture.
- A candidate may regress first-packet latency or sustained throughput by at
  most5%. UTMOS may drop by less than3%, and CER may increase by less than2
  percentage points.
- If no fixed backend meets the performance gate, the manifest records a
  legacy exception and its removal condition instead of hiding a fallback.

## Resource and latency model

For a tile `BM×BN×BK`, `W` warps and `P` shared stages:

```text
shared/CTA = P × (BM×BK×bytes(A) + BK×BN×bytes(B)) + epilogue workspace
registers/thread ≈ ceil(BM×BN/(32W)) + operands + indices + epilogue
resident CTA = min(shared, registers, threads and architectural CTA limits)
jobs(tiled) = ceil(M/BM) × ceil(N/BN) × splitK
jobs(full-M) = ceil(N/BN) × splitK
```

The ranking model keeps launch, quantization, Tensor Core, DRAM, L2, shared,
dependency, reduction, layout and epilogue costs separate:

```text
T = launch + quant
  + max(tensor, DRAM, L2, shared, dependency)
  + reduction + layout + epilogue
```

Missing measured terms stay missing. The planner does not infer scoreboard
latency from utilization percentages or claim an analytic optimum.

Shared-memory layouts are scored against the actual warp access pattern. XOR
parameters minimize bank multiplicity, but final selection still uses component
latency. Full-M is generated only when physical BM covers logical M and resource
limits remain legal. Split-K is disabled unless the role explicitly permits its
numerical order and reduction.

## Artifact lifecycle

Every manifest is pinned to:

- physical CUDA device properties;
- model asset identity;
- inference source hash;
- PyTorch, CUDA and Triton versions.

`shadow` manifests can be attached to a deployment but cannot change a kernel.
`candidate` manifests are for explicit offline A/B only. Runtime schedule
application requires `validated` status and an exact identity match.

Generate architecture inventories without claiming performance:

```bash
PYTHONPATH=src python scripts/planner_v2.py inventory \
  --sm 80 --sms 108 --batches 1 8 16 32

PYTHONPATH=src python scripts/planner_v2.py candidates \
  --sm 89 --sms 128 --role target:qkv
```

Generate a source- and hardware-matched shadow manifest on the target machine:

```bash
ACC_CLEAR_TRITON=native bash scripts/run.sh scripts/planner_v2.py shadow \
  --current-config configs/runtime.yaml \
  --batches 1 2 3 4 5 6 7 8 16 \
  --matrix-dtype bf16 \
  --output /tmp/planner-v2-shadow.json
```

The current device probe reports launch, effective DRAM/L2 copy rates and
preferred-precision Tensor throughput. It intentionally leaves shared throughput
and dependency latency incomplete until Nsight Compute evidence supplies them.

## Porting workflow

1. Inventory real first-head and tail shapes on the target model.
2. Query hardware resources and measure primitive rates.
3. Generate no more than eight legal candidates per role/shape.
4. Calibrate candidates sequentially on one GPU, outside serving.
5. Check numerical output, then whole-component CUDA Graph latency.
6. Run the manifest's supported batch gates. The current SM89 manifest uses
   B1--B8/B16; B32 is excluded because the full head graph does not fit and an
   eager fallback would not be a fair comparison.
7. Run the fixed 256-case UTMOS/CER gate.
8. Mark the manifest validated only after all gates pass.

SM80/86/89 support is a framework capability until those devices complete this
workflow. Do not reuse the SM120 manifest or claim performance from a synthetic
profile.
