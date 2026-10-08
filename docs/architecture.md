# Current repository architecture

Two independent models share the src/inspark_infer layout and command dispatcher.

- Index pipeline: Draft online900, Target IndexTTS2, CFM bilingual40k step800 four intervals, BigVGAN. Current FP8 B1/B8/B64/B128 and INT8 SmoothQuant B1/B8/B64 are selected in configs/current. NVIDIA DSparkWorker bridge plus ordinary TensorRT compute; CUDA Graph/KV/retained plugins and microbatch schemes are component-specific. Private assets and exact runtime versions are pinned in configs/current/release.json and environment.json.
- ZipVoice-Distill: retain concurrent upstream INT8 B16/B24/B32/B64 integration, isolated environment and private asset registry. See zipvoice-int8.md. Its model operators and delivery policies are unchanged by the Index migration.

Historical Index optimization routes are removed from current code and remain in Git history. Assets stay out of Git. Each model's loader checks its own source/profile/environment contracts; engine-load and short-test success are distinct from numerical or quality certification.
