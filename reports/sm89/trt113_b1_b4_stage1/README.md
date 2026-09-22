# SM89 stage-one evidence

Recorded on physical GPU6, NVIDIA GeForce RTX4090 / SM89, TensorRT11.3.0.99,
PyTorch2.8.0+cu128. Existing external allocations (21,854 MiB) were preserved.
The source implementation milestone is `d646331`; the reports retain their
individual revision/source fingerprints and artifact hashes.

## What passed

- Independent B1/B4 Target+Draft+CFM+Vocoder fixed-head TRT routes: two warmups,
  five measured waves per batch, no native-count or acoustic-route fallback.
  Enqueue-to-all-first-chunks P50: B1 41.554 ms, B4 74.262 ms. Small diagnostic
  samples on a shared device are not deployment latency guarantees.
- Four actual GPU cache-boundary/status tests passed: canonical K2048 writes,
  compact K128 clipping without neighbour corruption, capacity120/121 boundary,
  and status-bit combinations.
- 73 CPU tests passed, four GPU-only tests skipped in the CPU run. Command:
  `CUDA_VISIBLE_DEVICES='' PYTHONPATH=.toolchains/triton350:src .venv/bin/python -m pytest -p no:cacheprovider -q tests`.
- B1/B4 AR direct versus graph output comparisons and preserved cache regions
  passed. New AR metadata binds engines to the actual checkpoint/config hashes.

## What did not pass

| Comparison | B1 mismatches | B4 mismatches |
| --- | ---: | ---: |
| CFM native versus BF16 eager | 0 | 19 |
| CFM BF16 eager versus FP32 | 0 | 3 |
| Random-mel Vocoder native versus BF16 eager | 7,327 | 30,173 |
| Target native logits versus BF16 eager, identical rounded prefix cache | 4,816 | 18,639 |
| Draft native base logits versus BF16 eager, identical FP32 prefix cache | 5,233 | 33,122 |

All use the unchanged BF16 rule `abs(candidate-reference) <= .01 + .01*abs(reference)`.
Target selected states/new KV also have failures; inspect the full JSON, not only
this table. Passing final/hidden tensors does not override failing logits/KV.
The same-input FP32 comparisons and BF16 conversion comparisons are retained.

These are **diagnostic inputs**, not complete same-token speech trajectories:
AR uses real prefill caches plus fixed verify tokens; CFM uses real prompt/style
but zero future conditioning; Vocoder uses random Gaussian mel. They demonstrate
that a blanket numerical-parity claim is false, but do not establish perceptual
quality on real audio. The 256-case CER/UTMOS and complete-EOS soak belong to
stage two. No full quality, seed-isolation or concurrency-stability pass is claimed.

## Interpretation and files

`*_audit.json` files preserve every measured comparison, dtype, tolerance and
failure; `*_first_chunks.json` files contain actual routing windows and latency.
`*_initial.json` records precede the source-attested AR rebuild. Build metadata
contains hashes, not weights or engine binaries. Legacy acoustic metadata has no
export weight attestation and must not be upgraded retroactively.

The CFM mean includes 258/310 masked prompt frames. AR timings include diagnostic
cache reset/packing and have asymmetric preprocessing; do not use them as fair
operator speedups. Vocoder direct enqueues include Python plugin launch overhead,
while CUDA Graph replay bypasses it; these are distinct execution paths.

The first B8 attempt was rejected by the GPU-utilization preflight after the
previous job. That failure is retained separately; no other GPU process was killed
or memory freed to force a run.
After an idle preflight, the retained B8 profile passed the same two warmups/five
measured waves with all four TRT routes and no fallback; all-ready P50 115.784 ms.
