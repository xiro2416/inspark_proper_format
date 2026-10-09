# B32 混合精度：迁移后优化交接

阶段一见MIGRATION.md。本轮checkpoint、135NVFP4/92FP8/90BF16、四步CFM、KV80、GPU PCG/RNN和独立请求所有权不变。当前selected已经固定，GPU6空闲，正式测量由父agent排他协调。

## 保留与测量口径

| 改动 | 邻近短测证据 | 说明 |
|---|---|---|
| 已有Target KV写入融合 | 20波159.80→158.37ms | 保留；320请求codes/PCM/接受/EOS/KV长度逐字段相同 |
| 文本CPU workers8→16 | 20波158.34→156.20ms | 保留；CPU32未改善 |
| CFM FULL→MODERATE，L2仍56MiB | 同进程交替stageGraph24.641→24.143ms | 真实GPU收益约0.50ms；不能把受host波动影响的20波161.14→154.69ms全归因该改动 |

selected短测3波预热、20波P50约154.69ms，零回退。该值不是正式最终结果。最终仍需父agent安排baseline和selected各5波预热/30波/15秒功率；不从不同并行CPU环境计算可靠总百分比。

## 实际执行覆盖

- 8个精确B32 engine；Draft/Target/context/prefill/latent/suffix L5FULL/aux0，CFM L5MODERATE/L2=56MiB/aux0，Vocoder L5FULL/autoL2/aux0。全部Timingcache开启、tacticsauto、TF32关闭；作用中的引擎SHA、IO和quantization manifest在optimization_summary.json。
- CFM168原生NVFP4 GEMM＋64原生FP8卷积。Vocoder72原生FP8 correlation、40保护correlation、4个Float deconv(up2–5)，109已有FIR插件全FP32 IO。名义精度角色和实际kernel覆盖不同；不是纯官方kernel方案，已有FIR/KV自定义实现复用。
- NVIDIA DSparkWorker/RNN桥接＋普通TRT计算，非完整TRT-LLM Executor。原两轮父Graph、late12和完整声学B32保留；没有采用尾B8压缩或声学微批。
- selected声学输出32行不同、全部有限、无饱和；同配方/未量化浮点审计报告不设L2门槛；AR引擎未变化，既有AR审计加320请求状态等同性验证覆盖本次调度。未做CER/MOS认证。

## 未保留的候选

- prefill只读views/context重叠组合159.65ms，对当时158.93ms无收益；不据单个组合推断所有重叠无效。
- CPU32、late8/10、Graphburst1/4均比对应当前best慢；原2轮Graph与late12保留。
- Vocoder MODERATE独立stageGraph35.823ms，比FULL35.632ms慢；E2E短测亦未改善。
- FIR6个真实shape×4种已有block/warp配置，全部原256/4最佳且输出逐字相同；加权15.088ms/波，未改插件。
- 同配方尾B8 after12三engine身份验证通过，但20波155.58→161.18ms。仅最后一轮空间，state迁移/prime/两轮小桶Graph使20波物理Target250→260、Draft260→280；不保留，不把该结果归因未证实的L2竞争。

## AR hidden复用的逻辑阻断

真实32请求完整一波已记录各active round token、position、历史keep、只提交anchor/accepted的索引轨迹；1019个需要的speech hidden位置全部映射，但1019个input embedding都不同。AR首speech用位置2，latent suffix首speech用位置1（BOS位置0）；两个路径使用同一个embedding模块。

以同一真实prefix调用原GPT2InferenceModel.forward并在Transformer入口截获input_embeds：32请求cached_mel_emb长度37/40，attentionmask长度39/42，原生成首speech位置也为2，input embed逐字等于Native位置2。项目eager ARCore亦采用同公式。故这是原AR/latent双合同的复用前提不满足，未发现本次Native新增off-by-one。保留原suffix重算，未修改AR或latent位置语义。详见latent_contract.json；完整轨迹仅在本地artifacts。

## 剩余与停止

selected两波诊断：prefill21.23、AR67.15、latent7.18、CFM26.76、Vocoder36.58ms；scope可能重叠且含instrumentation，不能相加为E2E。
FIR/保护卷积/量化和布局仍耗时，更深机制并未证明无空间。已有廉价调度及形状相关参数探针已完成，较大的latent消除被逻辑差异阻断，4Float deconv约0.729ms/波（最大消除约0.46%E2E），不展开大重构。按父agent收尾指令结束本轮，等待正式窗口；不声称全局最优。

固定selected：artifacts/mixed_b32/selected_deployment.json；config runtime.yaml；backup deployment.json。原所有source/迁移资产/缓存保留。当前无任何发布。
