# B32 混合精度最终结果

先迁移后优化；RTX6000D/SM120，物理GPU6。最后排他窗口中其他本任务实验全部暂停，同cases/config，5波预热、30波测量、另测15秒持续整批E2E功率；全部样本保留。

| 方案 | 受理后首chunk P50/P95/P99（ms） | 含受理 P50/P95/P99（ms） | 功率 P50/P95/P99（W） | 采样峰值显存（GiB） |
|---|---|---|---|---|
| 精确B32迁移baseline | 161.69 / 167.88 / 168.82 | 163.59 / 170.00 / 170.92 | 274.89 / 465.85 / 472.53 | 15.68 |
| 优化后selected | 155.72 / 158.73 / 160.94 | 157.46 / 160.52 / 162.84 | 281.00 / 479.21 / 484.21 | 15.62 |

受理后P50降低 **3.69%**。这是整套方案收益；CFM MODERATE的同进程局部Graph仅约0.50ms改善，不能将整体约5.97ms全归因CFM。

## 保留方案及实际覆盖

- 已有Target KV写入融合，文本CPU8→16，CFM L5/MODERATE/L2=56MiB；其他7个精确B32 engine L5/FULL/aux0，Vocoder完整单engine，原两轮父Graph/late12保持。
- 相同Draft900/CFM800和135NVFP4/92FP8/90BF16配方。CFM四步单engine实际168原生FP4 GEMM＋64FP8卷积；Vocoder72FP8 correlation、40保护correlation、4Float deconv以及109已有FP32FIR插件；非完整TRT-LLM Executor。
- baseline与selected30波均零AR/CFM/Vocoder回退。960请求码/接受/EOS/KV长度逐字段相同；均Target374步、Draft390步、Graph194回放，speechcodeshostrows0。
- Selected声学审计输出32行不同PCM、有限、无饱和；浮点差异报告不设L2门槛，没有CER/MOS认证。
- 两方案延迟及功率均无超过各自P95×1.10的样本；全样本含最大值保存于formal_*_samples.json。统计单位为波，30波不是960个独立延迟样本，也不代表长期稳定性。

## 逻辑边界与未保留项

AR hidden复用未采用：真实完整一波1019必要speech位置均映射，但AR首speech位置2而latent为1（BOS0），embedding全部不相同；同真实prefix原eagerforward的缓存长37/40、mask长39/42，亦得到AR位置2。原库已有双合同，未调整语义，保留suffix路径。

CPU32、late8/10、burst1/4、prefixviews/context overlap组合、Vocoder MODERATE、tailB8 after12及FIR调度未改善并未保留；详见OPTIMIZATION.md。更深FIR/量化/布局/保护计算未证明无空间；已有低成本途径已核查，较大latent消除被逻辑阻断，按父agent指令结束本轮，不宣称全局最优。

selected/config/backup路径和状态见formal_summary.json；实际引擎SHA/IO/precision/plugin覆盖见optimization_summary.json。所有原source与迁移资产保持，worker未发布。父agent接手资产打包/下载验证及Git/HF统一发布。
