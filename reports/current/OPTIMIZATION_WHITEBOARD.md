# Current release migration

Goal: preserve selected execution recipes on Draft900/CFM800, publish only the seven current configurations and validated asset identities.

Hardware: NVIDIA RTX6000D / SM120 / 156 SM / L2 112MiB / power cap600W. Reference profile: /workspace/.codex/skills/gpu_parameters/rtx6000d-sm120.md. Same-shape transfer; no new search or performance claims from old checkpoints.

Original reference: A_0924 measured selected deployment snapshots. New semantic reference: retrained Draft900 and bilingual40k CFM800, four quarter intervals. New activation calibration:128 real first-chunk requests, Draft/CFM; Target/Vocoder component specs unchanged.

Retained: per-batch search parameters and profiles, device-round scheduling, KV/head-major layouts, CUDA Graphs, context/prefill/latent paths, current CFM/Vocoder graph rewrites and B128 microbatch/plugins.

GPU7 serial queue:20 affected Draft/context/CFM engines. No time limits. Reused components require matching tensor/role identities. Current identity/relocation tests:6 passed. Seven real routes passed 30waves/15seconds power with zero AR/CFM/Vocoder fallback. All seven AR/acoustic floating audits complete, finite, no L2 gate. Local materialized-weight relocation:3real B1 waves passed.

History: build logs and original snapshots remain local in artifacts/current_release. Current selected publication summary replaces historical reports.

Published private HF engine revision: 2edb087d47eefc95c0c008c71e27c21a1158c896;20 rebuilt engines and237 initial runtime dependencies, final239assets plusmanifest. Weights revision96933af87262eec17d685050723d4f768d2f8778. Detailed validation:VALIDATION.md. No new optimization search; migration objective achieved; remote asset hashes verified; independently downloaded B1 Draft/CFM load and3realwaves passed. Migration complete; no additional optimization search.
