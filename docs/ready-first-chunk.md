# Ready C：B32/B64/B128 首 chunk 调度

## 实际执行

本方案沿用最新 Draft900 / CFM800 权重与原量化配方，不重新训练或校准。AR 是 TensorRT 静态计算引擎与官方 DSpark worker/RNN 路径，**并非整套 TensorRT-LLM Executor**。CFM 保持四步。

| 区域 | Ready C 的实现 |
|---|---|
| admission / prefill | 使用指定 admission batch 的原优化路径；保留请求、prefix、RNG 归属；不计相同首段文本复用收益 |
| Draft / Target | 两个 speculative round 一张 Graph；全局第12轮后 verify-only，未完成请求在转移后 prime 一次 |
| AR 缩批 | B32→16→8；B64 可选48/32/16/8；B128 可选64/48/32/16/8；完整迁移21项状态，不重新 prefill 或播种 |
| 就绪观察 | 每个 Graph 回放后一个 packed ready/status D2H；新就绪组另一次 packed 元数据 D2H；不导回完整 KV 或概率表 |
| latent | 按原请求映射拷贝原 admission prefill 的 keep/KV/BOS hidden，到私有B16 prefix bank，运行既有 latent suffix；不复用 AR hidden 替代 latent |
| CFM / Vocoder | 复用完整 B16 TensorRT 引擎与已有融合/plugin；独立 context、缓冲和 Graph；真实尾组补齐16，丢弃虚拟输出 |
| 交付 | 声学组完成后立即调用 owner PCM callback，再继续剩余 AR；串行执行，无 D 模式的 AR/声学并发 |

三种对照：`baseline` 是原完整生产屏障；`barrier16` 等全部 AR 就绪再逐组声学；`A` 提前交付而不缩 AR。`C` 提前交付并缩 AR。B128 原生产屏障自身已有尾缩批，不能把其与固定 AR 的 A 强行要求逐 token 相同。

## 选择与复现

发布 registry 为 mixed B32/B64/B128 分别记录 `default_scheduler` 和 `scheduler_variants`。`--scheduler auto` 使用所选默认，`--scheduler barrier` 保留旧版 pin，`--scheduler ready_c` 显式选新版本。其他 batch/precision 的默认不受影响。

```bash
inspark fetch --asset-dir ./assets --precision nvfp4_fp8 --batch 32 --scheduler ready_c
inspark infer --asset-dir ./assets --precision nvfp4_fp8 --batch 32 --scheduler barrier \
  --gpu 0 --ref-audio voice.wav --text '待合成的文本。' --output result.wav
```

直接加载部署时，字段是 `first_chunk_scheduler: ready_c` 与 `ready_scheduler_plan`。Graph 在新进程重新捕获；engine 只可复用于匹配 GPU/SM/TRT 环境。标准打包与 materialize 路径可由 CLI 使用，无需 ready 工作区的额外环境变量。

验证与正式测量使用 `scripts/validate_ready_scheduler.py`、`benchmarks/benchmark_ready_first_chunk.py`；打包使用 `scripts/package_ready_bundle.py`。完整结果见 `reports/current/ready_first_chunk/RESULTS.md`。

## 指标与边界

先分别计算每一波的逐请求中位数、逐请求P95、前16和最后一个请求的真实 PCM callback 延迟，再计算跨波 P50/P95/P99。不把同波请求当作独立样本混合成P99。报告同时保留 admission 与 postadmission 两种起点。功率是独立持续服务窗口的整卡读数。

所有对照均驻留相同候选引擎，显存是这一共同配置的采样峰值，不能等同于原单一路径显存。验收侧重逐请求早交付；整批最后一个交付的退化如实报告。C 与 A 的形状/采样差异、较小 batch 的状态转移开销也单独报告。B32 本次 A 的请求中位P50比 C 快约1.7ms，C 缩批并非在每一档都会更快；保留两者证据，不宣称 C 是所有调度候选的最优。

这里验证固定 admission 的首 chunk 与请求清理，未验证持续到达/补槽、公平性、整句吞吐或 CER/MOS。B48 是 AR-only 桶；审计采用真实B64冻结历史非连续裁剪，不意味着有完整B48声学部署。EOS/取消/异常路径的 CPU 测试与自然模型输出分别标注。
