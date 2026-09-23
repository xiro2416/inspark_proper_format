# Device-Control B8：Draft / Target 功耗快速探索

## 方法

- GPU6，RTX 4090 / SM89，功耗上限 400 W。
- 部署：`configs/sm89_bf16_triton_device_control.json`，只测 Batch 8。
- 音色：`/workspace/index-tts/data/audio/old/mingxiang_gao.wav`。
- 将 B8 CUDA Graph 连续 replay 约 5 秒，使板级功耗传感器进入稳态。
- `draft_backbone`：Draft Transformer backbone，包括其中的 BF16 GEMM、MLP、attention 和 norm。
- `draft_full`：Draft backbone 加 Proposal RNN/采样 graph。
- `target_verify`：Target 对 8 个位置的 verification graph，包括 BF16 GEMM、MLP、attention 和 norm；不含 acceptance/commit。
- 单个微秒级 GEMM 短于板级传感器刷新周期，因此不能可信地逐 GEMM 读瓦数；连续 replay 得到的是该算子族的稳态平均功耗。

## 结果

模型和图已驻留、无计算时的空载基线为 79.64 W。

| 隔离路径 | 平均功耗 | 峰值功耗 | 平均耗时/call | 扣空载 J/call | SM clock | throttle |
| --- | ---: | ---: | ---: | ---: | ---: | :---: |
| Draft backbone | 217.50 W | 242.55 W | 0.431 ms | 0.0594 J | 2679 MHz | 无 |
| Draft + Proposal | 208.04 W | 242.51 W | 0.732 ms | 0.0940 J | 2685 MHz | 无 |
| Target verification | 247.36 W | 256.90 W | 2.650 ms | 0.4445 J | 2685 MHz | 无 |
| 完整 B8 首 chunk | 269.19 W | 276.10 W | 145.71 ms/组 | 27.62 J/组，3.45 J/request | 2672 MHz | 无 |

`draft_full` 比 `draft_backbone` 平均瓦数略低，但单次能耗更高：额外 Proposal/RNN 阶段功率密度较低，同时把一次调用从约 0.431 ms 拉长到 0.732 ms。

## 为什么没有接近 400 W

1. **B8 的 token 维度仍很小。** Target verification 的主要 GEMM 只有 8 个位置/请求；已有 kernel profile 中部分 CUTLASS BF16 GEMM 每次只有 8 个 CTA，而 RTX 4090 有 128 个 SM，无法让整卡同时保持高 Tensor Core 占用。
2. **多数投影 GEMM 是显存带宽/权重读取受限。** 已有 roofline 数据显示相关 shape 的算术强度约 8–60 FLOP/byte，投影达到各自 DRAM roofline 的约 35–79%。这类 kernel 不会表现出大型训练 GEMM 的 350–400 W 功率密度。
3. **Attention 更偏延迟受限。** 短序列 attention 的 grid 很小，部分路径接近单 CTA/grid，SM 并行度不足。
4. **执行由很多短 kernel 组成。** GEMM、norm、RoPE、attention、Proposal、accept/commit 和 graph replay 之间存在 launch/依赖边界；device-control 只把主要状态留在 GPU，仍保留每轮一个 all-ready/fallback host 标量边界。
5. **不是限频。** 所有隔离窗口保持约 2.67–2.70 GHz，驱动 throttle reason 为空，温度最高 52 °C；400 W power limit 没有触发。

因此当前低功耗首先代表工作负载并行度和算术强度不足，而不是有大量可通过提高 power limit 获得的性能。B8 下 Target verification 的单次耗时和能耗都约为 Draft backbone 的 6–7 倍，后续若继续优化 AR 算子，应优先看 Target 的小 M projection/attention 调度和轮数，而不是调高功耗上限。

原始结果：`outputs/profile_sm89/device_b8_draft_target_power.json`。
