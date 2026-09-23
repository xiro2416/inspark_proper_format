# Device-Control B8：逐个单 GEMM 稳态功耗

## 口径

- 单张 RTX 4090 / GPU6，400 W power limit；实验配置为 `configs/sm89_bf16_triton_device_control.json`。
- 使用当前模型的真实权重、bias、dtype 和 B8 实际 M shape。
- 每个 GEMM 单独连续执行：2 秒稳态预热，再采样 4 秒；空载基线 86.17 W。
- `M×K×N` 表示输入 `[M,K]` 乘权重 `[K,N]`。
- 相同 layer role 的层具有相同 shape/kernel，测一个真实 layer 权重并列出每 AR round 的重复次数。
- 连续执行同一个 GEMM 会使权重进入 L2，因此这是该 GEMM 的稳态功耗/吞吐上限，不是把一次真实 AR round 中各行能耗简单相加的依据。
- Slot Draft/Target 的 score/value attention 使用 fused Triton kernel，不存在独立 cuBLAS `QKᵀ`、`PV` GEMM；本表覆盖全部实际 Linear/Conv1D GEMM。

## Draft backbone

| GEMM | M×K×N | dtype | 次/round | 平均/峰值 W | μs/call | TFLOPS | 峰值利用率 | 边际 mJ/call |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| attention Q | 56×1280×1280 | BF16 | 3 | 153.82 / 153.95 | 13.57 | 13.52 | 8.2% | 0.918 |
| attention K | 56×1280×1280 | BF16 | 3 | 147.27 / 154.40 | 14.70 | 12.48 | 7.6% | 0.899 |
| attention V | 56×1280×1280 | BF16 | 3 | 138.62 / 140.75 | 17.40 | 10.54 | 6.4% | 0.913 |
| attention O | 56×1280×1280 | BF16 | 3 | 141.85 / 142.34 | 16.97 | 10.81 | 6.5% | 0.945 |
| MLP expand | 56×1280×5120 | BF16 | 3 | 263.57 / 303.07 | 14.45 | 50.79 | 30.7% | 2.564 |
| MLP contract | 56×5120×1280 | BF16 | 3 | 247.79 / 250.43 | 17.15 | 42.79 | 25.9% | 2.772 |
| vocab head | 56×1280×8194 | FP32/TF32 | 1 | 287.92 / 288.35 | 38.27 | 30.70 | 37.2% | 7.721 |

## Draft context 与 Proposal

| GEMM | M×K×N | dtype | 次/round | 平均/峰值 W | μs/call | TFLOPS | 峰值利用率 | 边际 mJ/call |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| selected-hidden projection | 64×6400×1280 | FP32/TF32 | 1 | **398.62 / 400.65** | 20.65 | 50.79 | 61.5% | 6.451 |
| context K | 64×1280×1280 | BF16 | 3 | 170.77 / 174.04 | 13.72 | 15.29 | 9.3% | 1.160 |
| context V | 64×1280×1280 | BF16 | 3 | 170.92 / 173.85 | 14.50 | 14.46 | 8.8% | 1.229 |
| Proposal hidden | 56×1280×1280 | BF16 | 1 | 153.13 / 154.16 | 16.90 | 10.86 | 6.6% | 1.132 |
| Proposal state | 8×512×1280 | BF16 | 7 | 106.02 / 117.44 | 13.02 | 0.81 | 0.5% | 0.259 |
| Proposal vocab | 8×256×8194 | BF16 | 7 | 149.63 / 150.33 | 10.48 | 3.20 | 1.9% | 0.665 |

`selected-hidden projection` 的 throttle reason 为 `0x4`，即 software power cap；这是唯一真正触及 400 W 上限的单 GEMM。

## Target verification

| GEMM | M×K×N | dtype | 次/round | 平均/峰值 W | μs/call | TFLOPS | 峰值利用率 | 边际 mJ/call |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| attention QKV | 64×1280×3840 | BF16 | 24 | 224.88 / 244.86 | 15.18 | 41.46 | 25.1% | 2.105 |
| attention O | 64×1280×1280 | BF16 | 24 | 161.12 / 161.31 | 13.58 | 15.45 | 9.4% | 1.018 |
| MLP expand | 64×1280×5120 | BF16 | 24 | 286.08 / 321.40 | 14.06 | 59.65 | 36.1% | 2.811 |
| MLP contract | 64×5120×1280 | BF16 | 24 | 299.30 / 302.37 | 13.61 | 61.63 | 37.3% | 2.901 |
| vocab head | 64×1280×8194 | FP32/TF32 | 1 | 323.81 / 325.64 | 38.57 | 34.81 | 42.1% | 9.165 |

## 解释

1. 高功耗 GEMM 实际存在：Draft context projection 达到 399 W，Target MLP 为 286–299 W，Target vocab head 为 324 W。
2. Draft/Target attention O 和 Draft Q/K/V 只有约 139–161 W，Tensor Core 峰值利用率约 6–9%；Proposal state 只有 0.5%。这些小 M GEMM 无法铺满 128 个 SM。
3. 真实 AR 中高功耗 GEMM只持续约 14–39 μs，且每层使用不同权重；持续单 GEMM测试的 L2 weight reuse 在真实逐层推理中不存在。
4. 高低功耗 GEMM之间还穿插 norm、RoPE、fused attention、sampling、context scatter、commit，以及每轮一次 host 状态边界。因此完整 B8 首 chunk 稳态平均约 269 W，而不会接近 400 W。
5. 提高 power limit 只可能帮助已经触顶的 FP32 context projection；它每轮只出现一次、约 20.65 μs，对端到端收益极小。主要优化方向仍是减少 Target 24 层权重流量、减少 AR rounds，或减少小 M kernel/调度开销。

原始数据：`outputs/profile_sm89/device_b8_single_gemm_power_steady.json`。
