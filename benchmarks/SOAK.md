# Complete-EOS soak

`soak_requests.py` reuses the real 256-case text/seed/emotion manifest and one explicitly selected real reference. It validates request scheduling and ownership, not eager/TRT numerical parity or model quality. No GPU results are implied by the CPU tests.

Run from the repository root with the same reference used for the candidate audit. This task is restricted to physical GPU 6; the script requires an explicit `--gpu` and has no default GPU:

```bash
ACC_TRITON_TOOLCHAIN=custom bash scripts/run.sh benchmarks/soak_requests.py \
  --gpu 6 --reference /workspace/index-tts/data/audio/old/mingxiang_gao.wav \
  --seconds 10 --concurrency 1 4 --strict-isolation \
  --output /workspace/A_inspark_marlin/.work/soak_smoke.json
```

The full default tiers are 1/4/8/16 admitted concurrent requests for at least 600 measured seconds each. `--batch 8` caps model execution at microbatch 8: 16 requests do not mean a B16 TensorRT engine. Matching `configs/hardware/sm89/sm89_trt113_safe_b{1,4,8}.json` files are selected automatically. An explicit `--deployment` overrides that selection, including for eager/compile/BF16 comparisons.

The default runtime configuration is `runtime_reference.yaml` (TF32 disabled).
Each report records source/configuration hashes, effective tier configuration,
physical GPU and pre-existing shared-device allocation. These are execution
provenance, not a replacement for checkpoint-bound numerical/quality reports.

```bash
ACC_TRITON_TOOLCHAIN=custom bash scripts/run.sh benchmarks/soak_requests.py \
  --gpu 6 --reference /workspace/index-tts/data/audio/old/mingxiang_gao.wav \
  --seconds 600 --concurrency 1 4 8 16 --batch 8 --strict-isolation \
  --allow-oom-skip-concurrency 16 \
  --output /workspace/A_inspark_marlin/.work/soak_600.json
```

Preparation, error/RNG probes and warmup have separate timers and are excluded from `elapsed_s`. The measured interval includes admission, inference, host receive, cleanup and monitoring/reporting overhead. Each finite wave drains to full EOS (or its explicit cancellation), so head-first scheduling cannot starve tails by continuously inserting new heads. The final wave may extend beyond the requested duration; it is drained, not truncated.

## Explicit pass gate

- Every completed request must have contiguous PCM chunks, a legal 44-frame first packet (or earlier EOS), and final EOS. Cancelled requests never count as completed throughput.
- The observed admitted count must equal the requested concurrency. Worker `scheduler_max_batch` and measured Target/CFM/Vocoder batches must respect the configured microbatch cap. Target batch counts use measurement-window deltas, excluding preparation/warmup.
- At least one **completed** request must record actual Target KV length >=129 (`--min-kv-length`, minimum 129). Selecting a long text or merely observing a cancelled long request does not satisfy this gate. The report keeps request/case IDs, KV length, code count and text length as witnesses.
- Empty-input errors are followed by cleanup and same-ID reuse. Real request RNG probes check that unrelated inference/cancellation cannot alter a waiting witness's RNG/CFM-noise hashes; admission/cancellation cannot alter an unchanged active row; and cancellation followed by same-ID/same-seed recreation restores the initial hashes. These explicit diagnostic boundaries are outside timing. They do not claim token equivalence after different numerical acceptance branches.
- Every drained wave must have no sessions/rows/errors or leaked/duplicate Target/Draft slots. Partial-wave errors preserve the phase, pending request IDs/cases/seeds and a best-effort runtime snapshot.

Memory budgets are operational defaults, **not previously measured guarantees**. They are explicit in `pass_gate.memory` and may be adjusted on the command line before a run:

| Flag | Default | Gate |
| --- | --- | --- |
| `--max-live-growth-mib` | 64 | Maximum fully drained CUDA allocated growth over the post-warmup baseline, including intermediate peaks |
| `--max-rss-growth-mib` | 256 | Same bound for worker process RSS |
| `--max-live-slope-mib-per-min` | 1 | Maximum CUDA allocated linear trend over the latter half of drained samples |
| `--max-rss-slope-mib-per-min` | 16 | Same late-window trend bound for worker RSS |
| `--memory-min-samples` | 3 | Minimum late-window samples for a trend conclusion |
| `--memory-min-span-seconds` | 60 | Minimum late-window span for a trend conclusion |

CUDA reserved/peak memory is recorded separately; allocator caching alone is not called a leak. A final return to baseline cannot hide an earlier drained live-memory budget violation. A low final growth cannot hide a sustained excessive late-window slope. Missing metrics fail closed.

Runs shorter than 600 seconds can only be `smoke_passed`, with `soak_qualified=false`; absolute memory/long-KV/RNG gates still apply, but a short smoke cannot qualify its memory trend. A >=600-second run must have sufficient trend samples and pass both growth and slope limits. This establishes only the bounded tested window, not an indefinite leak-free guarantee.

### Cold-window failure and a separate warmed control

The [first formal SM89 run](../../reports/sm89/stage2/runtime/soak_600_initial.json)
passed B1 but failed B4's RSS growth budget: 287.71 MiB exceeded 256 MiB after
one warmup wave. Its late RSS slope passed. Early growth followed by a plateau
is not, by itself, proof of a particular cache or proof that no slow leak exists.
The original failed report is retained unchanged. Its combined diagnostic
incorrectly labeled the absolute failure as a trend failure too; subsequent code
reports these predicates independently without changing the overall pass gate.

An explicitly separate B4 control may exercise the corpus before establishing
the measured baseline, using the same configuration and unchanged budgets:

```bash
ACC_TRITON_TOOLCHAIN=custom bash scripts/run.sh benchmarks/soak_requests.py \
  --gpu 6 --reference /workspace/index-tts/data/audio/old/mingxiang_gao.wav \
  --seconds 600 --concurrency 4 --batch 4 --warmups 128 --strict-isolation \
  --output /workspace/A_inspark_marlin/.work/soak_b4_warmed_control.json
```

At B4, 128 warmup waves admit 512 requests without deliberate cancellation.
For the current 256-case selection rule this covers all case IDs, but not every
possible numerical, cancellation or operator-shape branch. `warmup_seconds` and
`warmups` are retained separately; none of that time counts toward the 600-second
measured window. A warmed pass never erases the cold growth failure or its
preparation cost. Neither run establishes indefinite stability or eager parity.

## Allowed OOM and status

Only explicit CUDA out-of-memory errors in a tier named by `--allow-oom-skip-concurrency` can be skipped. The parser never allows 1/4/8 in this list. Non-OOM failures, including memory-gate failures or illegal memory accesses, always fail the run.

An authorized B16-concurrency OOM is stored as `skipped_cuda_oom`, including the error and any partial evidence. The other successful tiers remain separately recorded. The overall status becomes `completed_with_allowed_skip` and exits successfully because the requested skip policy was completed; **`full_soak_pass=false`, `soak_qualified=false`, and `all_requested_tiers_passed=false` remain explicit**. It is not reported as a B16 stability pass. Any 1/4/8 failure exits nonzero and cannot be hidden by a later/allowed skip.
