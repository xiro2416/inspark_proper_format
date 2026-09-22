# TensorRT 11.3 Draft KV 修复版：32 文本延迟与功耗

日期：2026-09-22  
设备：GPU6，RTX 4090 / SM89  
语料：`configs/sm89_benchmark_32_texts_v2.json`，随机情绪 seed 20260923  
功耗：每档 15 秒连续首 chunk workload，NVML board power

| Batch | 首 chunk P50 (ms) | P90 (ms) | 稳态吞吐 (req/s) | 平均功耗 (W) | 峰值功耗 (W) | 单请求能耗 (J) | GPU util | 平均 rounds | accepted_sum |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 41.635 | 45.519 | 23.664 | 206.05 | 213.14 | 8.708 | 82.32% | 7.063 | 25.531 |
| 4 | 72.670 | 82.308 | 53.370 | 235.79 | 250.07 | 4.418 | 74.66% | 7.688 | 24.375 |
| 8 | 107.860 | 111.602 | 68.495 | 248.43 | 279.71 | 3.627 | 67.50% | 7.625 | 23.969 |
| 16 | 未完成 | 未完成 | 未完成 | 未完成 | 未完成 | 未完成 | 未完成 | 未完成 | 未完成 |

## B16 阻塞证据

GPU6 上容器不可见的 host PID 845135 持有 21826 MiB。B16 部署捕获阶段，本次进程占用 26.03 GiB，整卡只剩 1.69 MiB；申请最后一个 2 MiB CUDA Graph allocation 时 OOM。该 PID 在容器 PID namespace 中不存在，当前权限无法释放。没有通过删减图、修改 batch 策略或迁移其他用户 GPU 来伪造 B16 可比结果。

## 原始结果

- `outputs/trt113_draft_new32/trt_draft_fixed_b1_power15s.json`
- `outputs/trt113_draft_new32/trt_draft_fixed_b4_power15s.json`
- `outputs/trt113_draft_new32/trt_draft_fixed_b8_power15s.json`

