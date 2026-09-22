# SM89 update benchmark: `19e7f62` → `b55cf6a`

## Test contract

- One physical GPU: GPU6, NVIDIA GeForce RTX 4090 (SM89).
- Deployment: `configs/sm89_bf16_triton_device_control.json`.
- Reference: `/workspace/index-tts/data/audio/old/mingxiang_gao.wav`.
- Corpus: the same 32 texts and random emotions used by the previous-version run.
- `head_batch_barrier=true`; latency is the time until all requests in a group have their first chunk.
- Two warmups per batch. Latency covers all 32 texts; sustained power replays requests for at least 8 seconds.
- Power is instantaneous board power sampled by NVML about every 20 ms.

## Results

Values in parentheses are relative to the previous `19e7f62` measurement. The old run did not include B4.

| Batch | All-first-chunk median | P95 | Sustained throughput | Mean / peak power | Board energy / request |
|---:|---:|---:|---:|---:|---:|
| 1 | 61.46 ms (-1.44%) | 74.47 ms (-1.97%) | 16.28 req/s (+1.34%) | 193.52 / 202.65 W (-0.36% / -0.29%) | 11.887 J (-1.68%) |
| 4 | 87.90 ms | 90.55 ms | 42.69 req/s | 231.27 / 250.45 W | 5.417 J |
| 8 | 146.96 ms (+9.53%) | 154.82 ms (+1.65%) | 56.33 req/s (-1.99%) | 260.49 / 297.94 W (-1.67% / -2.66%) | 4.624 J (+0.34%) |
| 16 | 260.43 ms (+1.64%) | 275.24 ms (+6.46%) | 59.29 req/s (+0.06%) | 279.27 / 376.50 W (-0.55% / +5.77%) | 4.710 J (-0.61%) |

All measured device rounds succeeded: B1/B4/B8/B16 had 32/8/4/2 successes and zero fallback. CFM and vocoder used the requested real batch sizes.

## Interpretation

The active SM89 path has no demonstrated throughput or energy regression. B1 and B16 are effectively unchanged in sustained throughput; B8 sustained throughput is 1.99% lower and energy is 0.34% higher. The B8 group-latency median moved more because it contains only four groups: P95 moved only 1.65%, and the two prior same-corpus runs also showed noticeable B8 median variation. This update's unified AR implementation is gated behind `unified_ar`, which the BF16 SM89 deployment does not enable.

Raw data: `outputs/profile_sm89/mingxiang_random32_device_control_b55cf6a_b1_b4_b8_b16_power8s.json`.
