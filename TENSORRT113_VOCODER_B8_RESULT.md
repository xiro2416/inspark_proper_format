# TensorRT 11.3 Vocoder B8 实验结论（SM89）

日期：2026-09-22  
设备：RTX 4090 48 GiB（SM89），GPU 6  
精度：保持 BF16 卷积与 FP32 接口，不引入 FP8/INT8 量化

## 实现

- TensorRT 11.3 strongly-typed 静态引擎：B8、80 mel bins、52 frames。
- 普通 Conv1d 交给 TensorRT optimization level 5 做 tactic timing/search。
- SM89 缺少可用强类型 tactic 的 6 个 ConvTranspose1d 使用 Quick Plugin，内部调用 BF16 cuDNN。
- 109 个 alias-free activation 使用现有两段 Triton 融合，并作为 TensorRT Quick Plugin 嵌入引擎。
- Target、Draft、CFM 延用既有 TensorRT 11.3 B8 实现，调度和 BF16 策略不变。
- 完整实验部署：`configs/sm89_bf16_trt113_full_b8.json`。

## 修复的实现错误

旧 Quick Plugin 把 TensorRT 传入的 legacy default stream 句柄 `0` 直接包装成
`torch.cuda.ExternalStream(0)`，导致 PyTorch/cuDNN 实际运行在另一条流上。TensorRT 在默认流上
提前消费尚未完成的输出，因此出现 `mean_abs≈0.49`、cosine 接近 0 的系统性错误。

修复规则：句柄为 0 时映射到 `torch.cuda.default_stream()`；只有非零句柄才使用
`torch.cuda.ExternalStream(stream)`。诊断性强制同步曾恢复到 cosine 1.0，最终实现不保留同步。

## 单模块验证

随机高斯 mel（偏离真实 CFM 分布）：

| 路径 | B8/F52 延迟 | max abs | mean abs | cosine | SNR |
|---|---:|---:|---:|---:|---:|
| 当前 BF16/cuDNN + Triton alias | 19.60 ms | - | - | - | - |
| TensorRT direct enqueue | 108.16 ms | 0.2969 | 0.02052 | 0.997960 | 23.89 dB |
| TensorRT CUDA Graph | 11.33 ms | 0.2969 | 0.02052 | 0.997960 | 23.89 dB |

Direct enqueue 慢是 115 个 Python Quick Plugin 回调的 CPU 开销；CUDA Graph replay 不再执行这些
Python enqueue 路径，因此是实际部署路径。剩余随机输入差异来自 TensorRT Conv tactic 与 cuDNN BF16
归约顺序，而不是流错误或量化。

## 真实音频同源 A/B

固定 16 条文本、音色、情绪、seed；两臂使用同一 Target/Draft/CFM，只替换 Vocoder：

| 指标 | 结果 |
|---|---:|
| 输出长度一致 | 16 / 16 |
| 平均 waveform cosine | 0.9999896 |
| 平均 SNR | 58.30 dB |
| 平均 SI-SDR | 58.30 dB |
| 平均 log-spectral distance | 0.217 dB |
| 平均绝对误差 | 5.69e-5 |

这表明随机高斯 mel 会显著放大误差；在真实 CFM mel 分布上，两条路径高度一致。

## B8 端到端 A/B（同机、同命令、2 warmup + 5 repeats）

| 路径 | 全部首 chunk 均值 | 吞吐 | 平均板卡功耗 | 单请求能耗 |
|---|---:|---:|---:|---:|
| TRT Target/Draft/CFM + 当前 Vocoder | 121.69 ms | 66.01 req/s | 92.57 W | 1.394 J |
| TRT Target/Draft/CFM/Vocoder | 105.50 ms | 75.86 req/s | 94.59 W | 1.247 J |
| 变化 | **-13.31%** | **+14.93%** | **+2.18%** | **-10.55%** |

功耗窗口只有约 0.5–0.6 秒，且 GPU 6 有约 21.9 GiB 外部常驻显存；功耗值只用于同次 A/B，
不替代此前的长时间稳态功耗测试。

## 裁决

完整 TensorRT B8 pipeline 已实现并通过端到端运行、CUDA Graph、真实音频同源 A/B 和短时性能门槛。
它保留为独立 SM89 B8 配置，不覆盖当前通用配置；B1/B4/B16 会按现有后端或 fallback 运行，不能把
本报告的 B8 结论外推到其他 batch。

