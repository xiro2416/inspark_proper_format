# B4/B16/B32 mixed precision migration and optimization

三agent分别独占GPU4/5/6并行迁移，三档全部验证后再优化。最终baseline/selected按排他窗口依次测量，5warm/30waves/另15秒功率，全样本保留。数据为受理完成到整批首chunkPCM，显存为全进程NVML采样峰值。原checkpoint和保护规则保持；浮点审计报告无固定L2gate，未认证长期稳定性/CER/MOS。

| Batch | 首chunk P50/P95/P99 ms | 整卡功率P50/P95/P99 W | 峰值GiB | 对本档迁移基线P50改善 |
|---:|---|---|---:|---:|
| 4 | 56.94 / 65.84 / 67.93 | 198.80 / 269.94 / 290.44 | 13.52 | 12.61% |
| 16 | 105.83 / 117.49 / 119.21 | 219.12 / 411.24 / 417.79 | 13.73 | 2.59% |
| 32 | 155.72 / 158.73 / 160.94 | 281.00 / 479.21 / 484.21 | 15.62 | 3.69% |

B4：CPU4、单轮Graph、late8、prefill视图/重叠及已有KV写融合；同图匹配证明Target FP8更快，选63NVFP4/164FP8/90BF16，Target四图原生FP8、Draft/Context保持NVFP4。专属packedweights另存，原135/92/90版本保留。

B16：prefix视图/重叠、KV融合、单轮Graph、late10、CPU8。B32：KV融合、CPU16、CFM MODERATE/L2=56MiB、其他FULL，保留2轮Graph/late12。两档均135/92/90原配方，无重训练/重标定。

所有新8引擎精确shape/native/模型校准/ABI及实际路由验证通过；正式与3waveportable均零fallback。B16/B32请求码/PCM/state与各自baseline保持一致；B4精度改变后接受率/rounds变化，报告独立列出，不能把局部GEMM收益直接当E2E。

覆盖例外：Vocoder四处FP8 Q/DQ ConvTranspose实际FP32 deconv，源B64已有；该shape开销低，记录而不误称全76角色nativeFP8。FIR109插件保持FP32 ABI、已有math复用。强搜索/Graph命中不自动证明收益；失败候选按各档证据拒绝。

B32 ARhidden→latent复用被实际position合同阻断：原eager/native首speech位置2，而latent位置1。真实1019个需要位置逐一映射发现embed全部不同，保留原suffix路径，未静默改变AR/latent语义。

量化权重及engine发布在私有xirr/index_pipeline；代码/配置/聚合报告在GitHub。当前三档portable bundle已SHA校验/materialize并运行3wave，码与PCM与正式对应请求一致。同GPU和匹配记录环境可复用，Graph在新进程重捕获。原B64/B128/FP8/INT8发布保留。各档RESULTS和summary中含详细build/fusion/precision/浮点/候选/停止理由。
