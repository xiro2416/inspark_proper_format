# NVFP4 B64 当前实现与验证

Draft：150k 在线 step900；CFM：双语40k第一阶段 step800、四个 Euler 区间。GPU7单卡串行测试；最新选定版本预热5波、测量30波，另测15秒持续推理功率。128条中英评估文本、四个3秒VAD音色；关闭文本去重，不计相同文本计算复用加速。

| 配置 | 首chunk 入组确认→整批PCM P50/P95/P99 ms | 功率 P50/P95/P99 W | 显存采样峰值 GiB |
|---|---|---|---:|
| NVFP4 original | 498.01 / 502.93 / 816.24 | 403.92 / 486.60 / 493.18 | 21.78 |
| NVFP4 selected | 357.12 / 361.93 / 362.20 | 406.31 / 493.67 / 501.17 | 20.40 |
| FP8 matched | 343.33 / 347.30 / 347.75 | 398.95 / 483.36 / 488.42 | 23.59 |

选定 NVFP4 的 P50 相对原始 NVFP4 改善 **28.29%**；仍比同期 FP8 慢 **4.02%**。FP8 默认保留。所有选定测量 AR/CFM/Vocoder 回退为0。原始 NVFP4 中的慢波保留，未过滤异常；当前30波无超过自身P95的110%的波次。短测不代表长期稳定性认证。

## 保留的优化

| 组件 | 实际方案 | TRT参数/调度 |
|---|---|---|
| Target | 原生NVFP4线性算子；官方Attention，原BF16预缩放Q、K/V和verification mask；Attention.scale=1，输出BF16后恢复FP32 | B64,Q8,历史KV80；builder5/moderate/aux0，CUDA Graph |
| Draft | 原生NVFP4骨干/低精度context K/V；原NVIDIA RNN映射与GPU PCG；每轮RNN初始state=0，只提交接受位置 | Q7非因果proposal slots、KV80；builder5/moderate，context5/full；head-major KV、两轮父Graph |
| Prefill/latent | 静态NVFP4引擎；prefill KV用于cached latent suffix | Q48/Q80/suffix40；builder5/moderate；static IO与已选Graph |
| CFM | 完整四步Solver单engine，保留完整Transformer上下文和原WaveNet tail/halo语义 | F310/P258，builder5/FULL，L2=56MiB，aux0 |
| Vocoder | 520逐tap Gather合为76整窗Gather；固定窗口索引；109已有FIR/Snake/FIR融合算子，明确FP32/LINEAR ABI；低精度learned算子全部原生FP4 GEMM | mel52→PCM13312，builder5/moderate/aux0，CUDA Graph；未新增GPU数学kernel |
| Runtime | 沿用input packing/条件并行/slot KV；启用已有StaticGCGuard | 自动请求GC仍启用；setup对象冻结，Engine.close解除；不新增中间D2H |

全体317个指定learned算子中90个BF16保护、227个NVFP4，沿用按层/stage首四分之一取整的现行策略。RNN/KV/embedding/norm/FIR等角色外参数保持原存储精度。NVFP4：官方ModelOpt0.47 max PTQ，E2M1 W4A4、K block16、E4M3 block scale、FP32 global scale，动态activation block scale。权重打包本身不证明原生执行，所有必需GEMM另查inspector/CUDA trace。

## 搜索与修复记录

- 固定FIR-up pointwise图移植实测更慢；不采用。tap-major布局只带来约1.7ms短测变化，当前保持原[channel,tap]量化分组。
- 官方TRT-LLM fp4_quantize适配输出逐值一致，但代表性算子约1.49ms对1.47ms，且当前TRT拒绝CUDA Graph捕获；不采用。
- 初始FIR AOT接入有FP32指针ABI/BF16格式宣告错误，造成PCM饱和。该版本及所有派生延迟均撤回。修复显式FP32边界、FP32-only格式和ABI断言后，整网109个FIR的全部实际IO均为FP32；64请求有64个不同PCM，饱和比例约0.03%。这是逻辑错误修复，不是放宽数值门槛。
- 独立定位到一次619.9ms的Python generation2 GC暂停，对应约983ms首包慢波；用现有静态对象GC保护消除本轮30波异常。
- aux2在初始FIR图上stream assignment失败，实际aux0；FULL/L2=0触发编译器异常，FULL/L2=56MiB也产生相同binary。这些结果不冒充有效并行/更优tiling，也不宣称搜索穷尽；修复后发布选用已实测的moderate/aux0。

## 浮点审计

比较相同最新权重和量化配方的Torch参考，并报告未量化参考。数字是开根号相对L2百分比，无固定L2验收阈值。Target Attention加入BF16输出舍入边界，float32接口保持；mask/KV/query缩放逻辑不变。

| 输出 | TRT相对同配方 % |
|---|---:|
| draft_hidden | 11.568399 |
| target_logits | 11.603065 |
| target_selected | 11.682937 |
| target_kv_append | 13.102063 |
| CFM_same_recipe | 1.375310 |
| CFM_unquantized | 2.069233 |
| Vocoder_same_recipe | 27.574074 |
| Vocoder_unquantized | 66.983466 |

所有审计输出有限；RNN对照使用相同TRT hidden和实际proposal链，条件q能复现proposal。NVFP4的声学浮点差异较大，保留报告；本次不声明CER/MOS、音质或统计分布等价。完整审计见float_audit.json。

## 剩余开销与边界

当前NVFP4原生GEMM不是最初主要瓶颈；窗口展开和动态block quantization付出了额外数据与执行成本。TRT11.3直接Conv支持表没有FP4，本实现用标准图几何加原生FP4 GEMM，未将BF16 Conv或FP8当作原生FP4。官方量化算子替换在已测条件下无净收益。普通TRT与NVIDIA DSpark/RNN桥接共同执行；不是完整TRT-LLM Executor/AutoDeploy。当前选中版本可用且有实测改善，尚未超过FP8，不宣称全局最优。

官方接口依据：[TensorRT11.3 Convolution](https://docs.nvidia.com/deeplearning/tensorrt/11.3.0/_static/operators/Convolution.html)、[TensorRT11.3 Attention](https://docs.nvidia.com/deeplearning/tensorrt/11.3.0/_static/operators/Attention.html)。原始日志、失败候选、CUDA trace和逐波数据仅留本地实验归档。

发布：私有HF revision `d3fa9e80e2997a7007c839c07a386297141f6b6a`。独立下载选中Target/Vocoder engines并核对SHA，重建原模型容器后3波真实B64首包加载/执行通过，回退0；这项检查仅验证可移植性，不取代上表匹配测量。
