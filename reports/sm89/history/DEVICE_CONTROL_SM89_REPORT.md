# SM89 Device-Control 实验报告

## 测试范围

- GPU：物理 GPU6，NVIDIA GeForce RTX 4090 / SM89；开始时板级显存占用 20 MiB。
- 代码：`19e7f62` 加本地 SM89 兼容修复。
- 基线：`configs/sm89_bf16_triton.json`。
- 实验：`configs/sm89_bf16_triton_device_control.json`。
- 单个 worker、单张 GPU；每个部署在一个 `max_batch=16` 进程内依次测试 B1/B8/B16。
- 每档 2 次完整 warmup，随后 10 次正式测量。
- `head_batch_barrier=true`，所以延迟是同批全部请求完成首 chunk 的时间。
- 吞吐是首 chunk 请求吞吐，不是长文本完整生成吞吐。
- 首 chunk 保持 44 acoustic frames（11,264 PCM samples）。

## 结果

| Batch | 部署 | 全部首 chunk median | P95 | median 吞吐 | 板级显存峰值 | Device round |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | BF16 基线 | 72.01 ms | 73.97 ms | 13.89 req/s | 29,694 MiB | 0 |
| 1 | Device control | 61.10 ms | 74.23 ms | 16.37 req/s | 29,756 MiB | 10/10，0 fallback |
| 8 | BF16 基线 | 177.24 ms | 203.37 ms | 45.14 req/s | 29,704 MiB | 0 |
| 8 | Device control | 134.26 ms | 155.26 ms | 59.60 req/s | 29,764 MiB | 10/10，0 fallback |
| 16 | BF16 基线 | 418.52 ms | 441.41 ms | 38.23 req/s | 29,714 MiB | 0 |
| 16 | Device control | 250.75 ms | 254.48 ms | 63.81 req/s | 29,768 MiB | 10/10，0 fallback |

相对基线：

| Batch | median 延迟 | P95 延迟 | median 吞吐 | 额外峰值显存 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | -15.1% | +0.3% | +17.9% | +62 MiB |
| 8 | -24.3% | -23.7% | +32.0% | +60 MiB |
| 16 | -40.1% | -42.3% | +66.9% | +54 MiB |

## 结论

1. 新链路在 B1/B8/B16 均被实际命中，正式测量共 30 次成功、0 次 residual fallback。
2. B8 和 B16 的延迟、吞吐收益显著，额外显存约 60 MiB，可以进入下一阶段验证。
3. B1 median 更好，但 P95 与基线相当；不能声称 B1 尾延迟得到改善。
4. 当前结果只验证性能、显存、44-frame 输出合约和设备路径命中。由于设备路径使用独立的 device RNG 语义，提升为默认配置前仍需固定数据集音质/CER 验证。

## 移植中发现并修复的问题

上游 Context scatter Graph 按 `max_batch` 捕获 metadata tensor，但 DeviceRoundHead 在 B1/B8 replay 时传入实际 batch 长度，导致 `Graph input signature changed`。本地修复将 scatter metadata 按捕获容量预分配并对非活动项补零，保留每档实际 token bucket；修复后 B1/B8/B16 均通过且无 fallback。

## 原始数据

- `outputs/profile_sm89/baseline_b1_b8_b16.json`
- `outputs/profile_sm89/device_control_b1_b8_b16.json`
