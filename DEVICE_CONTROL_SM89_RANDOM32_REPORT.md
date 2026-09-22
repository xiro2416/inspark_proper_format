# SM89 Device-Control：明响音色、32 文本随机情绪 A/B

## 测试条件

- Reference：`/workspace/index-tts/data/audio/old/mingxiang_gao.wav`，时长约 5.805 秒。
- Corpus：`configs/sm89_benchmark_32_texts.json`，32 条不同中文文本。
- 情绪：固定种子 `20260920`；随机方向归一化后乘随机总强度，确保 8 维权重均在 `[0,1]` 且总和不超过 1。
- 对比：现有 BF16 Triton 基线与 SM89 device-control 实验配置。
- 单个 worker、单张物理 GPU6、`max_batch=16`；每个部署进程内依次测 B1/B8/B16。
- 正式延迟测试每个 batch 档均覆盖完整 32 条文本：B1 为 32 组、B8 为 4 组、B16 为 2 组。
- `head_batch_barrier=true`；延迟为一组内全部请求首 chunk ready 的时间。
- 吞吐为首 chunk requests/s，不代表完整长文本生成吞吐。

## 32 文本延迟、吞吐和显存

| Batch | 部署 | 全部首 chunk median | P95 | median 吞吐 | 板级显存峰值 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | 基线 | 65.04 ms | 82.23 ms | 15.37 req/s | 30,036 MiB |
| 1 | Device control | 62.35 ms | 75.97 ms | 16.04 req/s | 30,100 MiB |
| 8 | 基线 | 190.87 ms | 200.81 ms | 41.97 req/s | 30,044 MiB |
| 8 | Device control | 134.17 ms | 152.32 ms | 59.65 req/s | 30,106 MiB |
| 16 | 基线 | 406.90 ms | 407.39 ms | 39.32 req/s | 30,056 MiB |
| 16 | Device control | 256.24 ms | 258.54 ms | 62.45 req/s | 30,112 MiB |

Device control 相对基线：

| Batch | median 延迟 | P95 延迟 | median 吞吐 | 额外显存 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | -4.1% | -7.6% | +4.3% | +64 MiB |
| 8 | -29.7% | -24.2% | +42.1% | +62 MiB |
| 16 | -37.0% | -36.5% | +58.8% | +56 MiB |

正式 32 文本测量中，实验配置的 device round 命中分别为 32/32、4/4、2/2，均为 0 fallback。

## 稳态板级功耗

短至 50–400 ms 的单组请求短于消费级 GPU 功耗传感器的有效刷新周期，因此功耗采用额外的持续重放窗口：每个 batch 连续重放相同 32 条文本至少 8 秒，并用约 20 ms 轮询记录板级瞬时功耗。每个窗口约 392–405 个采样点。

| Batch | 部署 | 平均功耗 | 峰值功耗 | 稳态吞吐 | 每请求板级能耗 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | 基线 | 180.80 W | 189.63 W | 15.09 req/s | 11.98 J |
| 1 | Device control | 194.22 W | 203.24 W | 16.07 req/s | 12.09 J |
| 8 | 基线 | 216.44 W | 285.98 W | 41.89 req/s | 5.17 J |
| 8 | Device control | 264.90 W | 306.07 W | 57.48 req/s | 4.61 J |
| 16 | 基线 | 227.85 W | 381.87 W | 41.02 req/s | 5.56 J |
| 16 | Device control | 280.80 W | 355.97 W | 59.25 req/s | 4.74 J |

Device control 相对基线：

| Batch | 平均功耗 | 峰值功耗 | 稳态吞吐 | 每请求能耗 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | +7.4% | +7.2% | +6.5% | +0.9% |
| 8 | +22.4% | +7.0% | +37.2% | -10.8% |
| 16 | +23.2% | -6.8% | +44.5% | -14.7% |

## 结论

- B8/B16：device-control 明显提高吞吐并降低首 chunk 延迟；虽然平均功耗更高，但每请求能耗分别降低约 10.8% 和 14.7%。
- B1：收益较小，平均功耗增加约 7.4%，每请求能耗基本持平略升；不应仅为 B1 默认开启。
- 显存代价约 56–64 MiB，可以忽略于当前约 48 GiB 卡容量，但实验配置仍不应在音质/CER 和 RNG 语义验证前替代默认配置。

## 原始结果

- `outputs/profile_sm89/mingxiang_random32_baseline_power8s.json`
- `outputs/profile_sm89/mingxiang_random32_device_control_power8s.json`
