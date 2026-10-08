# B64 NVFP4 GEMM + FP8 convolution

## 匹配测量

RTX6000D / SM120，物理 GPU7 串行；TRT11.3.0.99 / Torch2.13+cu132。每项预热5波、30波、单独15秒持续功率；128条现有中英文本轮转，4个3秒VAD音色缓存预热。关闭相同文本计算复用。首chunk统计单位为整波最后一个PCM到达，30波并非1920个独立样本。P99属于短测分位数，不代表长期稳定性。

| 方案 | 受理后 P50 / P95 / P99 ms | 含受理 P50 / P95 / P99 ms | 整卡功率 P50 / P95 / P99 W |
|---|---|---|---|
| 原 FP8 | 343.23 / 347.31 / 347.98 | 347.28 / 351.86 / 352.39 | 401.11 / 480.02 / 483.43 |
| 原 NVFP4 | 361.51 / 364.88 / 365.40 | 365.44 / 368.93 / 369.53 | 403.63 / 496.18 / 506.36 |
| FP8 + 同款新 Vocoder | 295.67 / 300.94 / 301.13 | 300.22 / 304.99 / 306.09 | 388.56 / 504.11 / 506.36 |
| NVFP4 GEMM + FP8 Conv | 278.69 / 282.88 / 283.73 | 283.17 / 287.28 / 288.17 | 330.69 / 481.67 / 496.68 |

相对采用同款新Vocoder的FP8，选中方案受理后P50降低约5.7%；相对原FP8降低约18.8%。后者同时包含Vocoder改图收益，不能全部归因于NVFP4。功率是独立持续E2E重放采样，未添加dummy负载。

## 保留的实现

- Draft/Target：既有普通TRT原生NVFP4 GEMM、NVIDIA DSparkWorker/RNN桥接、GPU PCG和commit；不是完整TRT-LLM Executor。RNN每轮state=0，只提交接受的位置。沿用head-major独立KV slot、static KV80、两轮父CUDA Graph按需重放、prefill/latent复用、关闭文本计算去重。
- CFM：最新双语40k stage1 step800，完整四步solver单engine。51个NVFP4 GEMM角色、16个FP8 Conv角色、20个BF16保护角色。builder5 / FULL / L2=56MiB / aux0 / tactics auto / timing cache开。Inspector实际168个原生FP4 GEMM、64个FP8 Conv；四步展开和融合导致kernel数量不等于角色数量。
- Vocoder：76个FP8 Conv角色、40个BF16保护角色，直接卷积，不构造FP4 im2col窗口。单TRT engine复用已有109个FIR AOT插件，全部FP32 IO；本次未新增GPU math kernel。builder5 / MODERATE / L2 auto(112MiB) / aux0 / tactics auto / timing cache开。Inspector实际96个FP8卷积和55个BF16卷积实现层，不能直接当作逻辑角色计数。
- 全局135 NVFP4 /92 FP8 /90 BF16角色；其他参数精度保持原状。权重是最新Draft150k online900及CFM800；Target/BigVGAN原始权重。所有reuse经过对应组件精度、校准角色和张量身份校验。
- 既有FIR dtype ABI保护和static对象GC保护保留，常规请求GC仍开启。WORKSPACE实际约83GiB、TACTIC_SHARED_MEMORY实际1GiB，是构建器上限，不宣称数学意义无限。

## 实测筛选

初始混合方案沿用分段FP8Vocoder，短测约330ms。完整FP8Conv+FIR图使Vocoder诊断时间约126.5→78.3ms，最终E2E约279ms。aux=2候选实际aux=0、短测280.6ms，无收益；AR全部改回FP8候选约284.1ms，未保留。已有FP32 FIR插件复用，不把插件称作原生TRT算子。

## 正确性和浮点审计

所有最终对照零AR/CFM/Vocoder回退。实际输出有限、64个不同PCM，饱和比例约0.00023%。审计记录同配方及未量化参考差异，无固定L2门槛。CFM相对L2约1.31%/2.03%；Vocoder约9.05%/16.82%。AR在同一冻结真实KV和proposal下审计，不能据此认证完整采样分布、CER或MOS。具体Draft/Target/logit/KV差异见float_audit.json，不隐去较大差异。

## 复现及交付

可搬运bundle：/workspace/inspark_nvfp4/artifacts/mixed_b64/bundle；低精度权重：artifacts/mixed_b64/weights。bundle manifest记录每个engine/依赖SHA，materialize由最新未量化权重恢复加载容器，CUDA Graph在目标进程重新捕获。同GPU/相同记录的TRT、Torch/CUDA环境可复用，不承诺跨ABI可用。

配方生成：scripts/make_mixed_recipe.py；权重导出：scripts/export_nvfp4_weights.py；ONNX导出：scripts/export_nvfp4.py；FIR规范化：scripts/canonicalize_nvfp4_fir.py。engine_selection.json记录实际选中参数与诊断stage，summary.json保留分位数、功率、显存、接受率、物理轮次。

## 剩余与停止原因

此次完成按算子混合精度方案和针对暴露开销的图级优化。没有证明全局最优；Vocoder FULL、更广L2/tactic组合、AR状态读取及ready-row冗余工作仍可进一步探索，当前没有匹配E2E收益证据。保留已证明的方案，结束本次交付；不把一项负结果或仍未饱和600W解释为所有方向耗尽。原始日志、trace及淘汰候选仅存本地artifacts/mixed_b64。

可搬运bundle已通过全部文件SHA校验、最新权重materialize及3波真实B64加载验证，AR/CFM/Vocoder均零回退。相关原始结果保留于artifacts/mixed_b64/portable_validation.json。
