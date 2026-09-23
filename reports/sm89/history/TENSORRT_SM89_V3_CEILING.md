# SM89 TensorRT V3 能力上限实验

## 结论

V2 的失败主要不是 TensorRT 本身，而是错误地把 CFM 与 Vocoder 绑定替换。V3 对模块边界和 batch 独立搜索后，得到稳定的混合方案：

```text
Draft/Target：当前 device-side BF16 路径
CFM：TensorRT 严格整图两步 Solver
Vocoder：当前 Triton alias-free + BF16/cuDNN 路径
```

该方案在 B1/B8/B16 同时改善首 chunk、持续吞吐、功耗和能耗；B4 首 chunk 中位数与当前方案基本打平，持续吞吐和能耗更好。代价是约 394–480 MiB 额外显存，以及固定 batch/固定 310-frame CFM Engine。

## V3 相比 V2 修正了什么

- V2 只编译单次 CFM estimator，外部 Solver 仍调用 Engine 两次；V3 将两次 estimator、中间 `x + 0.5*v` 和 prompt mask 放入一个严格整图 Engine。
- `require_full_compilation=True`，不允许 PyTorch fallback。
- B1/B4/B8/B16 分别构建静态 Engine，允许每档独立 tactic 和内存计划。
- CFM/Vocoder 先分模块 CUDA Event 测量，再组合，不再假定同一后端必须覆盖两个模块。
- 实测 `FULL` cross-kernel tiling 和 72 MiB L2 budget；没有收益的候选被淘汰。
- 量化、Slot Draft/Target、批量验证、device-side acceptance 和外层 CUDA Graph 均保持不变。

## 分模块结果

下表为 50 次 CUDA Graph 重放的平均延迟；B1 的最终复核为 100 次，数值与第一次一致。

| Batch | 当前 CFM ms | V2 TRT estimator×2 ms | V3 TRT完整 Solver ms | 当前 Vocoder ms | V2 TRT Vocoder ms | 最优组合 |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 5.216 | 2.206 | 2.000 | 4.329 | 5.687 | TRT CFM + 当前 Vocoder |
| 4 | 9.919 | 4.724 | 4.533 | 9.342 | 16.655 | TRT CFM + 当前 Vocoder |
| 8 | 16.617 | 7.949 | 7.794 | 15.748 | 32.029 | TRT CFM + 当前 Vocoder |
| 16 | 30.957 | 14.698 | 14.537 | 30.978 | 69.296 | TRT CFM + 当前 Vocoder |

完整两步 Solver 比两次 estimator Engine 调用进一步减少约 1.1%–9.3%；收益随 batch 增大而缩小，因为 Engine launch 与中间状态边界占比下降。

Vocoder 呈相反趋势：当前 alias-free 融合在所有 batch 都明显优于 TensorRT。B1 开启 TensorRT `FULL` tiling 后，Vocoder 从 5.687 降到 5.504 ms，但仍比当前 4.329 ms 慢约 27.1%，因此没有继续为 B4/B8/B16 构建 FULL tiling Vocoder。

## FULL tiling 实验

- GPU 报告 L2 cache 为 75,497,472 bytes，直接作为 `l2_limit_for_tiling`，没有手填估算。
- B1 CFM Solver：NONE 1.9998 ms，FULL 2.0004 ms，无收益。
- B1 Vocoder：NONE 5.6872 ms，FULL 5.5039 ms，改善约 3.2%，仍不及当前 alias 路径。
- TensorRT 10.12 对 CFM 和 Vocoder 的大量 tiling tactic `0x3ea` 报内部断言 `g.nodes.size() == 0` 并跳过；Engine 仍成功生成，但不能声称所有 FULL 候选实际可用。

因此，当前设备/图上的最优 TensorRT CFM 使用 `tiling=NONE`；这不是遗漏开关，而是 B1 实测选择结果。

## 同环境端到端对比

双方均在 GPU6、相同 32 条文本、情绪种子 `20260920`、2 次预热、每档独立进程和 8 秒持续功耗窗口下重新测试。变化为 V3 hybrid 相对 current。

| Batch | 当前/Hybrid 首 chunk ms | 延迟变化 | 当前/Hybrid 吞吐 req/s | 吞吐变化 |
|---:|---:|---:|---:|---:|
| 1 | 44.393 / 43.078 | -2.96% | 21.891 / 22.781 | +4.07% |
| 4 | 78.585 / 79.531 | +1.20% | 49.565 / 52.367 | +5.65% |
| 8 | 121.729 / 105.977 | -12.94% | 70.090 / 71.705 | +2.30% |
| 16 | 194.762 / 179.571 | -7.80% | 73.814 / 82.730 | +12.08% |

| Batch | 当前/Hybrid 平均 W | 功耗变化 | 当前/Hybrid J/请求 | 能耗变化 | 显存增量 MiB |
|---:|---:|---:|---:|---:|---:|
| 1 | 202.05 / 193.92 | -4.03% | 9.230 / 8.512 | -7.78% | +394 |
| 4 | 231.51 / 231.78 | +0.12% | 4.671 / 4.426 | -5.24% | +430 |
| 8 | 263.51 / 241.63 | -8.30% | 3.760 / 3.370 | -10.37% | +468 |
| 16 | 279.30 / 267.56 | -4.20% | 3.784 / 3.234 | -14.53% | +480 |

B4 的首 chunk 差异只有 0.95 ms，且样本只有 8 个 batch group；延迟优先时可保留 current，吞吐/能耗优先时使用 hybrid。其余三档选择 hybrid。

## 质量门

32 条随机情绪音频全部保持相同采样率、样本数和 chunk 数，AR 调度没有漂移。

| 指标 | V3 hybrid vs current |
|---|---:|
| 相同长度 | 32 / 32 |
| PCM16 bit-exact | 0 / 32 |
| SNR 中位数 | 50.18 dB |
| SNR 最小值 | 19.55 dB |
| PCM16 RMSE 中位数 | 15.82 |
| PCM16 RMSE 最大值 | 407.50 |
| 最大绝对差最大值 | 7918 |

长度/结构门通过，但仍有少数数值离群；投入生产前需要补主观听测、ASR CER/WER、说话人相似度和情绪一致性。

## 尚未到达的 NVIDIA 栈上限

当前环境是 TensorRT 10.12 + Torch-TensorRT 2.8。它可以严格编译完整 CFM/Vocoder，但本机 API 不包含较新版本的 `IAttentionV2` 和 `IKVCacheUpdate`；Slot Draft/Target 还使用项目自定义 KV append/attention Triton kernel，无法直接作为普通 Torch-TensorRT 图完整接管。

后续 Draft/Target 上限实验应使用独立的新版本环境，保持当前环境不受影响：

1. 用原生 TensorRT Transformer/KV API表达完整 Target verify block 与 Draft proposal block，而不是逐 MLP Engine。
2. 保持相同 Slot KV、8-token Target 验证和 device-side acceptance；先只替换 block 内部计算。
3. 按 batch 和 KV bucket（64/128/256/512/1024/2048）分别搜索。
4. 对比当前自定义 KV attention、TensorRT fused attention、以及可能的 TensorRT-LLM Executor 路径。
5. 只有单 block 与完整 round 都获益后才进入端到端组合。

## 产物

- 构建脚本：`scripts/build_sm89_tensorrt_v3.py`
- 隔离基准：`scripts/benchmark_sm89_tensorrt_acoustic.py`
- 部署加载器：`src/acc_infer_clear/tensorrt_backend/deploy.py`
- Hybrid 配置：`configs/sm89_bf16_tensorrt_v3_hybrid.json`
- 严格 CFM Engine：`artifacts/tensorrt_sm89_bf16_v3`
- FULL tiling B1 实验：`artifacts/tensorrt_sm89_bf16_v3_full`
- 模块结果：`outputs/profile_sm89/tensorrt_v3_acoustic_b{1,4,8,16}.json`
- 端到端结果：`outputs/profile_sm89/{current_v3_recheck,tensorrt_v3_hybrid}_b{1,4,8,16}_power8s.json`
- 质量报告：`reports/tensorrt_v3_quality32_pcm.json`

