# InSpark Proper Format

Inference code, installation and SM89-specific audit results are documented in
[inference/README.md](inference/README.md). Local weights and generated engines
remain outside the package. Historical SM120 evidence is labeled separately.

Current TensorRT B1/B4/B8 profiles execute, but strict numerical parity has
**not passed**; they remain experimental. See the README for reference and
request-isolated configurations and the measured scope.

The two-stage implementation and SM89 audits are complete. Paired 256-case
quality gates passed. Ten-minute 1/8/16-concurrency soaks passed; B4 exceeded the
RSS budget after one warmup wave and passed a separately documented fully warmed
control. The original failure is retained in the
[audit results](reports/sm89/stage2/README.md). Pure FP32 eager remains the default.
