# SM89 单 GPU 优化落地结果

## 结论

当前可部署候选是 **BF16 cuBLAS/cuDNN + device-side AR + 既有 CFM 融合 + BigVGAN alias-free Triton 融合**。自定义 GEMM/Conv tile 没有进入运行时；Planner V2 以 fail-closed candidate manifest 绑定配置，在线调优关闭。

B32 不在 manifest、基准或质量门禁的支持范围内。支持批次固定为 `1,2,3,4,5,6,7,8,16`。

## Alias-free A/B

同一 GPU6、同一 max-batch 进程、相同 32 文本/音色/seed/emotion、2 次 warmup、8 秒稳定功耗窗口：

| Batch | 首 chunk 中位延迟 | 持续吞吐 | 每请求板级能耗 |
| ---: | ---: | ---: | ---: |
| 1 | 61.46 → 58.10 ms（-5.46%） | 16.28 → 17.59 req/s（+8.07%） | 11.89 → 11.34 J（-4.64%） |
| 4 | 87.90 → 76.49 ms（-12.98%） | 42.69 → 50.68 req/s（+18.71%） | 5.42 → 4.49 J（-17.16%） |
| 8 | 146.96 → 113.32 ms（-22.89%） | 56.33 → 68.46 req/s（+21.54%） | 4.62 → 3.74 J（-19.16%） |
| 16 | 260.43 → 205.48 ms（-21.10%） | 59.29 → 81.39 req/s（+37.29%） | 4.71 → 3.63 J（-23.02%） |

候选显存峰值约比基线少 320 MiB。完整 256 条 B1 音频的平均生成耗时为 1.112 → 1.062 秒（-4.50%）；这个数只作补充，不替代稳定首 chunk/power A/B。

## 质量门禁

- 固定种子：`20260921`。
- 256 条中文文本全部唯一，按主情绪做不重复的最大均衡，并在每类内部覆盖四个长度分位。
- 9 个参考音色各覆盖 28–29 条。
- baseline 与 alias 候选的 256/256 WAV 在采样率、长度和 PCM16 每个采样点上完全一致，最大差值为 0。
- 基线（也因此等于候选）：加权 WER 4.35%，平均 UTMOS 1.643，平均 SIM-o 0.4211；256/256 条均有效。

## Planner V2 边界

- BF16 公式清单：459 个签名、42 个 GEMM/Conv role、378 个 shape set、3024 个候选。
- 公式用于合法性和资源剪枝，当前不据解析排名启用 kernel。
- 设备 probe 已测 launch、DRAM、L2 和 BF16/FP8 Tensor 吞吐；shared-memory throughput 与 global dependency latency 缺失。
- 本机性能计数器由驱动限制为管理员使用，Nsight Compute 返回 `ERR_NVGPUCTRPERM`。因此 manifest 状态保持 `candidate`，所有 GEMM/Conv role 都显式记录 cuBLAS/cuDNN legacy exception，`planner_v2_apply=false`。
- 运行时不会加载 SM120 plan、不会在线 autotune，也不会应用这 3024 个解析候选。

## 产物

- 部署配置：`configs/sm89_bf16_triton_device_control_alias_candidate.json`
- Planner manifest：`configs/planner_v2_sm89_bf16_candidate.json`
- 固定质量集：`configs/sm89_quality_256.json`
- 公式候选：`reports/planner_v2_sm89_bf16_candidates.json`
- 设备 probe：`reports/planner_v2_sm89_probe_7802054/device_rates.json`
- PCM 对比：`reports/planner_v2_alias_quality256_pcm.json`
- 质量汇总：`reports/planner_v2_quality256_metrics.json`
- 生成与比较工具：`scripts/build_sm89_quality_corpus.py`、`scripts/run_sm89_quality_gate.py`

## 验证

- 9 个 contract/release 单元测试通过。
- SM89 preflight 通过，确认 BF16、无 custom plan、无在线调优、alias 融合开启。
- 绑定 candidate manifest 的 GPU6 B1 full-graph smoke 通过：首 chunk 104.17 ms，总时长 976.33 ms。

