# SM89 当前方案与 TensorRT 公平对比

> 此文档记录第一版（40 个小动态 engine）的结果，已由大图、静态形状、充分 tactic 搜索的 [TensorRT V2 对比](TENSORRT_SM89_V2_COMPARISON.md) 取代。不要用本页数据代表 TensorRT 的最终表现。

## 比较口径

- GPU：单张 RTX 4090 48 GB（SM89，物理 GPU6）。
- 两边相同：权重、BF16 GEMM/Conv 与 FP32 敏感接口、BF16 RNN、Slot Draft/Target、批量 Target 验证、device-side acceptance/residual/round、prefix/head CUDA Graph、head batch barrier、32 条文本、随机情绪种子 `20260920`、2 次预热、每档 8 秒持续功耗采样。
- 当前方案：保留 CFM norm/RoPE Triton 融合与 BigVGAN alias-free 融合。
- TensorRT 方案：从共同精度转换后的 eager 子图编译，不叠加上述项目专用算子融合。离线编译 40 个完整子图：24 个 Target MLP、3 个 Draft MLP、13 个 CFM FFN。
- 两边均未启用 FP8/INT8；TensorRT `enabled_precisions={FP32,BF16}`。
- Planner V2 在原配置中为 `apply=false` 的 shadow，不改变运行时。为避免本次源代码变更触发其 source-hash fail-closed，两个本次比较配置都不加载该 shadow；这不改变任一算子或调度。
- 功耗/能耗采用 8 秒持续回放窗口，避免用单次短请求的稀疏传感器采样下结论。

## 结果

TensorRT 相对当前方案的变化以 TensorRT / 当前计算；延迟为全部请求首 chunk 的中位数。

| Batch | 当前延迟 ms | TRT 延迟 ms | TRT 延迟变化 | 当前吞吐 req/s | TRT 吞吐 req/s | TRT 吞吐变化 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 58.305 | 75.700 | +29.83% | 17.320 | 12.875 | -25.66% |
| 4 | 76.448 | 101.362 | +32.59% | 50.548 | 38.799 | -23.24% |
| 8 | 113.149 | 144.401 | +27.62% | 69.106 | 52.536 | -23.98% |
| 16 | 192.524 | 281.253 | +46.09% | 76.278 | 59.708 | -21.72% |

| Batch | 当前平均/峰值 W | TRT 平均/峰值 W | 平均功耗变化 | 当前 J/请求 | TRT J/请求 | 能耗变化 | TRT 显存增量 MiB |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 192.10 / 203.34 | 180.39 / 184.78 | -6.09% | 11.092 | 14.011 | +26.32% | +3178 |
| 4 | 218.40 / 250.80 | 228.36 / 243.02 | +4.56% | 4.321 | 5.886 | +36.22% | +3178 |
| 8 | 262.32 / 277.46 | 246.29 / 291.98 | -6.11% | 3.796 | 4.688 | +23.50% | +3178 |
| 16 | 277.51 / 326.21 | 271.80 / 381.02 | -2.06% | 3.638 | 4.552 | +25.12% | +3178 |

## 结论与边界

当前方案在本机 SM89 和当前 workload 上全面占优：首 chunk 延迟低 21%–32%（B16 低 31.5%），持续吞吐高 28%–35%，单请求能耗低 19%–27%，并少用约 3.1 GiB 显存。TensorRT 的较低平均瓦数不等于更省电；其执行时间更长，因此四档能耗都更高。

TensorRT 的主要劣势来自大量短小、交错执行的 AR 子图：每层独立 engine 边界和动态 profile 的开销无法被其 MLP 融合收益抵消。项目现有的外层 CUDA Graph 已经压低 PyTorch/cuBLAS 调度开销，而 CFM/声学的形状专用 Triton 融合更贴合固定首 chunk 路径。

该 TensorRT 分支不是“整个模型被一个 TensorRT engine 吞下”：带 Slot KV 副作用的 attention/验证链必须保留在共同 scheduler；单独 projection GEMM 已实测 TensorRT 慢于 cuBLAS；BigVGAN 卷积仍回退到共同 BF16/cuDNN。以上回退均在 TensorRT plan 中显式记录，且回退成本已计入端到端结果。

端到端音频质量门尚未对 TensorRT 分支执行，因此本报告只作性能/功耗结论，不宣称 TensorRT 音频与当前方案逐样本等价。

## 产物

- 当前配置：`configs/sm89_bf16_current_fair.json`
- TensorRT 配置：`configs/sm89_bf16_tensorrt_fair.json`
- TensorRT plan：`artifacts/tensorrt_sm89_bf16/plan.json`
- 当前原始结果：`outputs/profile_sm89/current_fair_random32_b1_b4_b8_b16_power8s.json`
- TensorRT 原始结果：`outputs/profile_sm89/tensorrt_fair_random32_b1_b4_b8_b16_power8s.json`
