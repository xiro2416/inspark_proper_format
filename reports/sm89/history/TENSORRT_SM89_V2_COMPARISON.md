# SM89 TensorRT V2 公平对比

> 本页已由模块独立选择、完整两步 CFM Solver 和 FULL tiling A/B 的 [TensorRT V3 能力上限实验](TENSORRT_SM89_V3_CEILING.md) 取代。V2 将 CFM 与 Vocoder 绑定替换，掩盖了 TensorRT CFM 的真实收益。

## 结论

TensorRT V2 已按“大图 + 静态形状 + 充分 tactic 搜索”的方式重做，不再沿用 V1 的 40 个小 MLP engine。它在 B1 获得小幅收益，但在 B4/B8/B16 均不如当前 SM89 方案。因此，本机上不应把 TensorRT 作为全 batch 的统一替代后端；若质量门进一步通过主观/语义复核，可只为 B1 使用 TensorRT，B4/B8/B16 保留当前方案。

## V2 怎样使用 TensorRT

- 从共同 BF16 eager 基线开始编译，不叠加当前方案的 CFM Triton norm/RoPE 和 BigVGAN alias-free 融合。
- 每个 batch 各编译一个完整 CFM estimator engine 和一个完整 BigVGAN engine；B1/B4/B8/B16 合计 8 个 engine，而不是逐层切成许多小 engine。
- 使用静态形状、TensorRT optimization level 5、8 GiB builder workspace、每个 tactic 5 次平均计时和持久 timing cache。每个大图均确认只有一个 TensorRT engine。
- 保持相同的 BF16/FP32 边界，不使用 FP8/INT8，也没有改变权重量化策略。
- AR Target/Draft、Slot KV、副作用 attention、批量 Target 验证、device-side acceptance 和外层 CUDA Graph 保持双方相同。TensorRT 的搜索发生在其能完整接管的 CFM/BigVGAN 大图内部；它不会自动重写项目级 speculative decoding 调度或有状态 KV 控制流。
- 固定本次 `mingxiang_gao.wav` 提示音的 CFM 长度 310；每次测试只加载对应 batch 的两个 engine，非测试 batch 不计入显存。

## 公平测试口径

- 单张 RTX 4090 48 GB（SM89，物理 GPU6），每次只使用这一张 GPU。
- 相同权重、32 条文本、随机情绪种子 `20260920`、同一参考音频、2 次预热。
- 延迟是全部请求首 chunk 的中位数；吞吐、平均/峰值板卡功耗和单请求能耗来自每档独立 8 秒持续回放。
- 两个后端分别按 B1/B4/B8/B16 独立进程启动，避免 B16 CUDA Graph/缓存影响低 batch 显存及结果。

## 实测结果

变化量均为 TensorRT V2 相对当前方案；延迟越低越好，吞吐越高越好。

| Batch | 当前延迟 ms | TRT V2 延迟 ms | 延迟变化 | 当前吞吐 req/s | TRT V2 吞吐 req/s | 吞吐变化 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 45.672 | 43.731 | -4.25% | 21.120 | 22.072 | +4.51% |
| 4 | 82.852 | 85.004 | +2.60% | 49.292 | 47.475 | -3.69% |
| 8 | 127.337 | 131.026 | +2.90% | 66.201 | 60.840 | -8.10% |
| 16 | 194.643 | 215.839 | +10.89% | 76.713 | 65.945 | -14.04% |

| Batch | 当前平均/峰值 W | TRT V2 平均/峰值 W | 平均功耗变化 | 当前 J/请求 | TRT V2 J/请求 | 能耗变化 | TRT V2 显存增量 MiB |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 197.10 / 200.71 | 205.06 / 210.43 | +4.04% | 9.332 | 9.291 | -0.45% | +910 |
| 4 | 222.28 / 242.23 | 250.87 / 267.46 | +12.86% | 4.510 | 5.284 | +17.18% | +1024 |
| 8 | 242.22 / 273.69 | 259.65 / 301.24 | +7.19% | 3.659 | 4.268 | +16.64% | +1192 |
| 16 | 280.05 / 327.92 | 288.77 / 356.93 | +3.11% | 3.651 | 4.379 | +19.95% | +1294 |

V2 相比 V1 的大幅退化已经被消除，说明“大图接管和 tactic 搜索”确实是正确的 TensorRT 用法。但 B4 以后，TensorRT 大图的收益仍不足以抵消当前 CFM/BigVGAN 专用融合与执行路径优势；batch 越大，当前方案的吞吐优势越明显。B1 的能耗基本持平，代价是平均功耗约高 8 W、显存多 910 MiB。

## 32 条音频质量门

质量门使用相同文本、随机情绪、随机种子和参考音频。32/32 的采样率、样本数和 chunk 数一致，证明 AR 输出长度与调度没有漂移；0/32 为 PCM16 bit-exact，这是 BF16 后端重排之后可预期但不能忽略的差异。

| 指标 | 结果 |
|---|---:|
| 相同波形长度 | 32 / 32 |
| SNR 中位数 | 49.55 dB |
| SNR 最小值 | 19.36 dB |
| PCM16 RMSE 中位数 | 12.79 |
| PCM16 RMSE 最大值 | 400.03 |
| PCM16 最大绝对差中位数 | 556.5 |
| PCM16 最大绝对差最大值 | 7887 |

中位结果较接近，但少数样本差异被 CFM/声码器的迭代和非线性路径放大。故当前结论是“结构/长度门通过，数值质量门有离群点”，尚不能直接宣称听感完全等价。B1 若要投入使用，仍应补听测、ASR CER/WER、说话人相似度和情绪一致性门槛。

## 推荐落地

1. 默认继续使用当前 SM89 后端处理 B4/B8/B16。
2. 暂不启用 TensorRT B1；先完成上述感知质量门。通过后，可把 B1 作为可选路由，收益约为延迟 -4.25%、吞吐 +4.51%，但要接受更高瞬时功耗和显存。
3. 不建议继续扩大通用 TensorRT 编译范围到有状态 AR 控制链。下一轮若继续探索，应针对 CFM 或 BigVGAN 分模块 profile，确认是哪一个 TensorRT engine 在 B4+ 退化，再只替换获益模块。
4. 当前 engine 固定提示长度 310，不具备任意音色/提示长度的通用性。若需要生产化，必须建立少量长度 bucket 或动态 profile，并重新测量调度开销与显存，不能直接外推本表。

## 产物

- V2 配置：`configs/sm89_bf16_tensorrt_v2_fair.json`
- V2 构建脚本：`scripts/build_sm89_tensorrt_v2.py`
- V2 plan：`artifacts/tensorrt_sm89_bf16_v2/plan.json`
- 性能原始数据：`outputs/profile_sm89/{current,tensorrt_v2}_fair_b{1,4,8,16}_power8s.json`
- 质量语料：`configs/sm89_tensorrt_quality32.json`
- 质量波形：`outputs/tensorrt_v2_quality32/{current,tensorrt_v2}`
- 质量报告：`reports/tensorrt_v2_quality32_pcm.json`
