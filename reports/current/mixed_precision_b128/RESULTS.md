# B128 NVFP4 GEMM / FP8 convolution：先迁移，再优化

## 条件和测量

RTX6000D/SM120，物理GPU7串行构建及实验；TRT11.3.0.99、Torch2.13+cu132、CUDA13.2。沿用B64同一配方：90个BF16保护角色、135个NVFP4 GEMM角色、92个FP8卷积角色；其余参数保持原精度，没有重校准或重训练。Draft150k online900、CFM双语40k stage1 step800完整四步；Target/BigVGAN原始权重。

128条现有中英文本、4个3秒VAD音色。每项预热5波、30波，功率另测15秒连续完整波次。关闭相同文本计算复用。统计单位为整波最后一个PCM到达；30波不是3840个独立样本。短测不构成长时间稳定性或音质认证。

| B128方案 | 受理后首chunk P50 / P95 / P99 ms | 含受理 P50 / P95 / P99 ms | 整卡功率 P50 / P95 / P99 W |
|---|---|---|---|
| 完整 B128 迁移基线 | 559.03 / 561.16 / 562.16 | 568.15 / 570.21 / 571.04 | 415.77 / 501.70 / 513.64 |
| 优化后，AR 保持 B128 | 497.85 / 501.36 / 502.09 | 506.00 / 508.95 / 509.01 | 430.12 / 504.28 / 512.30 |
| 最终选中：声学微批＋AR 压缩 | 493.89 / 497.27 / 498.46 | 503.03 / 505.26 / 506.24 | 413.30 / 506.42 / 513.60 |

最终受理后P50降低 **11.65%**，约 **65.14ms**。原生NVFP4/FP8覆盖、模型身份、校准身份、Graph路径及零回退已验证。选中方案完整进程NVML采样峰值约 **31.64GiB**，分阶段及allocator峰值见summary.json。功率越高不扣分，也不以dummy工作增加功率。

## 迁移基线

先为B128重新构建8个精确shape引擎：Draft、Target、context、prefill、latent、latent suffix、CFM和Vocoder。B128 CUDA Graph在新进程重新捕获，static KV80/head-major独立slot、GPU PCG/commit、RNN每轮state=0和仅提交接受hidden的语义继承。没有直接拿B64静态engine当B128 engine。

基线所有新引擎builder5/FULL、aux0实际0、tactics auto、timing cache开、TF32关闭；CFM L2=56MiB，其余auto112MiB。实际workspace约83GiB，tactic shared-memory上限1GiB，未设置任意构建超时。CFM构建约652秒、Vocoder约964秒；出现部分Myelin/L2TC tactic异常并被跳过，最终引擎成功生成，执行及native覆盖随后独立验证。

迁移解决了latent-prefix校验白名单缺少混合精度标签的问题；binding、校准SHA与角色哈希验证保持。真实B128声学审计有限、128个不同PCM，基线短测零回退后进入优化阶段。

## 最终执行与优化

| 组件/区域 | 最终实现 | 构建及运行选择 |
|---|---|---|
| Draft/Target | 普通TRT原生NVFP4 GEMM＋BF16保护；NVIDIA DSparkWorker/RNN桥接、GPU PCG及KV | 精确B128/B64/B8，builder5/FULL、aux0、L2 auto；KV80，head-major arena、Graph、已有KV写入融合 |
| AR后段 | 在满足原有资格条件时，由B128压缩到B64，末段到B8 | GPU request state/KV/RNG迁移并恢复；实际记录18个请求进入B64、1个进入B8，不丢计算；不声称完整TRT-LLM Executor |
| Prefill/latent | B128专用TRT、缓存prefix的latent suffix | 只读KV/context视图、prefill/context重叠、保留prefix复用与请求所有权 |
| 输入/声学条件 | 沿用既有分组及2条condition stream | 文本worker8→32，不复用重复文本计算 |
| CFM | 同一个B64完整四步solver串行调用两次，并及时保存输出 | NVFP4 GEMM＋直接FP8卷积；builder5/FULL、L2=56MiB、aux0；B128 Graph覆盖两次enqueue和copy，完整时间轴/halo不变 |
| Vocoder | 同一个B64完整TRT Vocoder串行调用两次，并及时保存输出 | FP8卷积＋BF16保护＋109个已有FP32 FIR插件；builder5/MODERATE、L2 auto、aux0；B128 Graph覆盖两次enqueue和copy |

本次没有新增GPU math kernel。KV融合和FIR属于已有custom实现的复用，不能称作纯官方TRT算子。最终16个引擎包括合法B128 fallback及B64/B8 bucket/microbatch依赖。Inspector层计数不是逻辑角色数量；native_coverage.json给出每个实际engine的精度与插件覆盖。

## 闭环筛选

短测逐项累计：CFM微批558.5→537.1ms；Vocoder微批→530.2；prefix视图/重叠→517.4；CPU32→506.8；KV融合→499.3；AR压缩→493.9。各候选只作初筛，最终30波确认整套收益，并单独重测未压缩方案确认AR压缩净收益。

CFM/Vocoder跨请求并行微探针用真实输入、CUDA Graph完整fork/join：串行约128.2ms、并行约135.0ms，输出相同，因此未接入。这只排除该资源分配及shape下的方案，不证明所有重叠策略无效。

B64同图Vocoder进一步做builder5/FULL搜索，以补齐最终微批路径的构建对照；结果及不保留理由见experiments.json。强搜索不自动等于更快或全局最优。

## Profile与正确性

选中方案两波诊断：text约21.8ms、prefill57.4、DSpark168.0、latent27.4、CFM105.4、Vocoder156.6。带Profiler的scope可能重叠并包含提交/等待，不能把这些值相加当作无Profiler E2E。FIR和已保护BF16计算仍是明显成本；生成算子、量化及layout也占时间。kernel_hotspots仅统计当前profile窗口，不据kernel名称单独判定融合最优或错误。

选中实际声学输出有限、128个不同PCM、无饱和输出；所有最终对照零AR/CFM/Vocoder回退。CFM对同配方相对L2约1.16%，Vocoder约9.60%；完整AR/logit/KV及未量化参考差异在float_audit.json。按用户标准检查运算逻辑、mask、KV/commit、PCG、EOS、ownership与ABI，不采用固定L2阈值。冻结一次round审计不能认证整条采样分布、CER或MOS。

## 复现、资产及停止理由

源B64方案/权重/缓存均保留。目标原始构建、trace、淘汰候选、阶段记录在本地artifacts/mixed_b128及history。配方SHA与B64相同，HF已发布混合低精度权重可复用，目标只新增engine/deployment资产。scripts/package_unified_bundle.py打包实际选中依赖；api.release.materialize用相同最新未量化权重重建加载容器，Graph在目标进程捕获。同GPU及匹配的记录环境可复用，不承诺跨TRT/Torch/CUDA ABI可用。

此次已修复迁移接口问题，按profile完成微批、视图/重叠、输入并行、KV融合和AR压缩，最终闭环验证；继续并行的简单方案无收益，FULL构建对照也单独检查。更深FIR/保护计算/量化融合仍可能有空间，需要新的设计和独立profile证据；没有证明全局最优，也不因未到600W而推断一定存在收益。最后一项确认的AR调度收益约4ms；独立并行和FULL对照均无收益，现有FIR的6类形状×8组block/warp也均以256/4最快，结束本轮，保留当前最佳。

FIR调度探针使用真实量化参考中间值，比较block64/128/256/512和warps4/8；6类形状全部以现有256/4最快，输出均有限且与基准相同。加权B64总时间约31.14ms，各配置中选择最快仍为31.14ms；这是微探针，不是E2E。

可搬运bundle的全部文件SHA校验、最新权重materialize及3波真实B128验证已通过，零AR/CFM/Vocoder回退。
