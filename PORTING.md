# Porting the optimization strategy

The committed plan is an RTX 6000D/SM120 deployment, not a universal GPU plan.
For another GPU, reuse the structure and validation method—not the selected
tiles or FP8 assumptions.

## What the SM120 path does

- Keeps the shallow quarter of each serial model in BF16/FP32 where required;
  eligible middle/deep projections and learned convolutions use native FP8.
- Maintains request-owned fixed KV slots so verification reads history by slot
  and true length instead of rebuilding a temporary padded history each round.
- Captures Prefill, Draft, Proposal, Target and first-head acoustics for the
  explicit batch inventory `1..8,16,32`.
- Uses logical 48/80 first-head Prefill/Latent buckets. Variable tail work is
  not graph-captured.
- Uses shape-specific Target projection/attention/pointwise routes, exact
  LayerNorm-to-quantization and residual epilogues.
- Uses staged global-to-shared acoustic kernels with selected shared layouts,
  buffer rotation and fragment prefetch where the measured component latency
  improved.
- Uses Full-M only on the B8 projection shapes where the integrated graph won;
  rejected Stream-K and oversized BM paths are not part of the release.

## SM80/86 procedure

1. Add a new device profile; leave `configs/sm120.json` immutable.
2. Start with BF16 cuBLAS/cuDNN. SM80/86 must not select the SM120 native-FP8
   implementation.
3. Record real M/N/K and convolution signatures for B1/B8/B16/B32 first-head
   traffic. Generate candidates from local registers, shared memory, SM count
   and supported MMA instructions.
4. Measure complete components before selecting a tile. Rank by
   `critical-path cumulative time × achievable reduction`, not by a stall ratio
   alone.
5. For each hotspot inspect hardware roofs, Eligible/Issued Warps and top stall
   reasons, then use Source/SASS to locate the dependency. Validate changes by
   an A/B latency counterfactual.
6. Recreate graph buckets only after shapes and operator routes are fixed.
7. Run token/KV/acceptance/EOS checks, full streamed audio comparisons and the
   project quality gates before publishing a new device plan.

Do not assume that higher occupancy, fewer bank conflicts, more stages or a
larger BM is automatically faster. Component and first-packet latency are the
final selectors.

