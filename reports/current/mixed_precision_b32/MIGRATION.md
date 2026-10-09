# B32 混合精度：阶段一迁移完成

GPU6，RTX6000D/SM120，TRT11.3.0.99/Torch2.13+cu132；与源B64保持相同checkpoint、135NVFP4/92FP8/90BF16配方，校准SHA逐字一致。8个精确B32 engine均重新导出/构建，KV80、CFM310帧/prompt258/四步单enqueue、Vocoder52帧。没有重训练/重标定。

## 迁移证据

- 所有engine L5/FULL、aux0实际0、tacticsauto、timingcache开启；CFM L2=56MiB，其余auto；独立缓存。部分FULL Myelin/L2TC候选被TRT跳过，成功engine不代表搜索穷尽。
- Target官方Attention GS；CFM168原生NVFP4 GEMM＋64原生FP8卷积；Vocoder72原生FP8 correlation、40保护correlation、4个实际Float deconv(up2–5)，109已有FIR插件全FP32 IO。名义76FP8角色不等于76个原生FP8算子。
- head-major独立slot、GPU PCG/commit、RNN每轮state=0、两轮父CUDA Graph、prefill/latent缓存复用、CPU8、staticGC保护及关闭文本计算去重已继承。未直接移植B128微批及尾压缩。
- 一次真实声学头输出有限、32行不同PCM、无饱和；AR真实一次Draft/Target、条件q重现proposal、checkpoint身份验证通过；浮点审计在float_audit_migration.json。无L2门槛，也不作CER/MOS保证。

## 暂定性能（不是最终独占测量）

| 测量 | P50 / P95 / P99（ms） |
|---|---|
| 受理后首chunk | 159.16 / 163.92 / 164.89 |
| 含受理首chunk | 161.55 / 165.74 / 166.68 |

3波预热、10波、零AR/CFM/Vocoder fallback；采样整进程峰值约15.68GiB。功率未正式测量。并行环境共享CPU/IO，正式5波预热30波及15秒功率须父agent安排。

## 后续优化交接

两波profile诊断：text=13.75，prefill=21.81，AR=68.92，latent=7.33，CFM=28.24，Vocoder=36.66ms。这些带instrumentation/等待的scope可能重叠，不能直接相加作为E2E。
- 优先检查prefill只读view/context重叠、现有KV融合、后期B8真实压缩条件、CPU worker数；未完成请求slot/RNG/commit必须保持。
- 声学profile热点已有FIR、保护卷积、NVFP4 GEMM及布局；先对当前完整B32 engine确认pattern/layout/tactic，再按目标batch对FULL/MODERATE/L2/aux和微批做有依据探针。
- 4Float deconv合计两波仅1.457ms（每波约0.729ms），最大理论消除约0.46%E2E；不可只因名义FP8未生效就假设巨大收益。零插值Conv等已有graph改写须保留stride/pad/output_padding/groups/weightflip，若有新增materialization应E2E核算。

当前没有开启阶段二；GPU6已释放，等待所有三个migration验收。复现：artifacts/mixed_b32/migrate.py及validate.py；真正部署artifacts/mixed_b32/deployment.json，native_coverage.json记录8个engine SHA/实际参数/IO。
