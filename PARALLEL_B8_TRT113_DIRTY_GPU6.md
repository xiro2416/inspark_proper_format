# TensorRT 11.3 多 B8 lane：GPU 6 脏卡预实验

日期：2026-09-22  
设备：RTX 4090 48 GiB（SM89）  
部署：完整 TensorRT 11.3 Target / Draft / CFM / Vocoder B8

## 执行机制

- SM89 不支持 CUDA Programmatic Dependent Launch；PDL 至少要求 compute capability 9.0。
- 实际采用 cross-inference multi-streaming。
- 单进程共享 PyTorch 权重和不可变 TensorRT engines。
- 每个 B8 lane 独立持有 CUDA stream、TensorRT execution contexts、KV/cache、I/O 和 CUDA Graph。
- 只捕获 B8；没有捕获或比较 B16/B32，也没有为 B1–B7 分配实验 Graph。
- 所有实现均位于新增实验文件，没有修改现有 runtime/deployment/backend。

## GPU 状态与显存

| 状态 | 板卡显存 |
|---|---:|
| 启动前外部占用 | 21,854 MiB |
| 单 B8 lane 稳态总占用 | 40,289 MiB |
| 单 B8 lane 实验进程增量 | 约 18,435 MiB |
| 构建第 2 lane 失败时剩余 | 103 MiB |
| 第 2 lane 失败前进程占用 | 约 25.93 GiB |

第 2 lane 在部署/CUDA Graph 捕获阶段申请 240 MiB 时 OOM。因此本结果只说明当前脏卡的上限是
`1 × B8`；不能用它裁决干净 48 GiB RTX 4090 的真实并发上限。

## 单 B8 正式基线（已修正 host 调度）

输入与历史 105 ms 基线完全对齐：短文本“他正在整理文件。”、全零情绪、同一音色。
lane worker 在整个测试期间常驻；不再把每个 wave 新建/销毁 Python 线程池的 host 开销计入首 chunk。
5 秒稳态窗口，共 48 个 wave：

| 指标 | 数值 |
|---|---:|
| 并发请求 | 8 |
| 全部首 chunk 平均 | 103.59 ms |
| 全部首 chunk 中位数 | 103.56 ms |
| 全部首 chunk P95 | 114.73 ms |
| 吞吐平均 | 77.56 req/s |
| 平均板卡功耗 | 263.04 W |
| 峰值板卡功耗 | 289.24 W |
| 单请求能耗 | 3.391 J |
| 平均 GPU 利用率 | 70.89% |

同一并发 wave 内的 code 与 PCM 隔离检查逐项一致，PCM 最大差值为 0 LSB。

历史结果为 105.50 ms；用原基准程序在当前脏 GPU 6 原样复测为 107.31 ms。修正后的
103.59 ms 与历史水平一致。此前报告的 118.65 ms 是实验脚本每个 wave 都创建并 join
`ThreadPoolExecutor` 所产生的 host 调度开销，不是 vocoder 或 TensorRT kernel 回归。

## 当前裁决

- 实验实现和单 lane 路径有效。
- 脏 GPU 6 无法容纳第 2 lane，因此没有 2–6 lanes 的延迟、吞吐或功耗数据。
- OOM 前第 2 lane 已增加约 7.7 GiB PyTorch 分配；去除 21.3 GiB 外部进程后，至少 2 lanes
  在容量上有明显可行性，但必须在干净卡上实测，不能用线性外推代替。
- 修正后的机器可读结果：`outputs/parallel_b8_trt113_dirty_gpu6_aligned_persistent_workers.json`。
- 当前原基准复测：`outputs/trt113_full_b8_vocoder_ab_recheck_dirty_gpu6.json`。
