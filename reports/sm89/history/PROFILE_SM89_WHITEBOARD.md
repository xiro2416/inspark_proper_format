# RTX 4090 / SM89 Profile 白板

> 状态：本轮计划已执行到可安全落地的 candidate 边界。结果见 `SM89_PLANNER_V2_RESULT.md`。
> BigVGAN alias-free 融合已通过性能、功耗、显存和 256 条逐采样质量门禁；
> GEMM/Conv 自定义 schedule 因硬件计数器权限和完整校准证据不足而保持 fail-closed。
>
> 本轮只制定计划，不运行 GPU 测试、不选择新算子、不据结构猜测热点。

## 1. 目标

在单张、无外部负载的 RTX 4090 上，对当前生产候选配置进行端到端 profile，回答以下问题：

1. 首 chunk 的时间分别消耗在 frontend、Prefill、Draft/Target、Latent、CFM、Vocoder 和 D2H 的什么位置？
2. 每个主要模块内部累计耗时最高的 CUDA 算子是什么，调用次数是多少？
3. 热点属于计算、显存带宽、延迟、kernel launch、同步还是调度阻塞？
4. B1/B8/B16 下吞吐、首 chunk 延迟、整卡功耗及每个首 chunk 能耗如何变化？
5. 下一项优化应按 `关键路径累计时间 × 可实现降幅` 排在什么顺序？

## 2. 当前部署基线

- 硬件：NVIDIA GeForce RTX 4090，SM89，128 SM，约 48 GiB。
- 软件：PyTorch `2.8.0+cu128`，CUDA `12.8`，Triton `3.5.0`。
- Runtime：`configs/runtime.yaml`。
- Deployment：`configs/sm89_bf16_triton.json`。
- 精度：Target、Draft、CFM、Vocoder、RNN 和 learned convolution 使用 BF16；高精度接口按现有实现保留。
- 已启用 CFM Triton 融合：Adaptive RMSNorm/modulation、QKV 后处理及 RoPE。
- 已拒绝：SiLU×gate、`torch._scaled_mm` FP8、Triton W8A8、动态量化 GEMM、LayerNorm+FP8 quant。
- CUDA Graph：Target、Draft、Proposal、Prefix 和 first-head acoustics 启用。
- 首段调度：`head_batch_barrier=true`。
- 首 chunk 合约：44 acoustic frames，即 11,264 PCM samples。
- 单次只允许使用一张物理 GPU。
- 不使用 `torch.compile`，不做在线调优，不加载 SM120 设备绑定计划。

## 3. 已有证据及边界

已有数据只包括：

- CFM 融合候选微基准；
- 四种 GEMM 通用路径微基准；
- Norm+FP8 quant 微基准；
- 固定输入的单请求首包/总时长和 PCM 一致性；
- 非 barrier 并发测试中的整卡瞬时功耗窗口。

这些数据不能提供完整模块占比、top kernels 或硬件 stall 归因。

`StreamingCore.phase()` 当前使用 host `perf_counter()` 包围异步 CUDA enqueue。未经 CUDA Event 或显式同步校准的 `host_ms` 不得解释为模块 GPU 时间。

### 作废数据

- 最近的 barrier B8 重测受到其他任务插入影响：三轮延迟和功耗明显不一致。
- 该次 barrier B8 及未完成的 B16 不得进入正式结论。
- 不得把并发数 `batch=N` 自动解释为真实 `cfm_batch=N`；必须从事件字段记录实际 CFM/Vocoder batch。

## 4. SM120 策略与 SM89 可移植范围澄清

### 4.1 FP8 事实纠正

RTX 4090 / SM89 支持 FP8 Tensor Core。当前不启用 FP8，并非硬件不支持，而是现有代表性 shape 的实测结果显示完整 FP8 路径更慢：

- BF16 cuBLAS GEMM：约 22.9–27.4 μs；
- `torch._scaled_mm` 预量化：约 48.5–60.0 μs；
- Triton 预量化 W8A8：约 30.7–31.4 μs；
- 包含动态量化的路径：约 70.0–87.4 μs。

量化、scale 处理及额外 kernel launch 抵消了小/中 M shape 上的 FP8 Tensor Core 收益。因此当前决策是“实测否决”，不是“SM89 不支持”。若实际 profile 发现新的大 M 热点，可以针对该精确 shape 重新验证，但不得据硬件能力直接启用 FP8。

### 4.2 已经复用的通用优化

当前 SM89 profile 已启用：

- BF16 cuBLAS/cuDNN；
- Slot Target、Slot Draft 和 request-owned KV；
- batched Target verification；
- Target、Draft、Proposal CUDA Graph；
- first-head CFM/Vocoder CUDA Graph；
- 48-token Prefill 与 80-token Latent 基础 graph buckets；
- head batch barrier；
- CFM Adaptive RMSNorm/AdaLN modulation 融合；
- CFM QKV 后处理/layout 与 RoPE 融合。

`prefix_graphs=true` 已经为每个支持的 batch 捕获 Prefill 48 和 Latent 80 两个固定长度。因此不得再把“48/80 prefix buckets”描述为尚未移植或 SM120 独有能力。

当前没有启用的是更进一步的 SM120 `prefix_padding_plan`。该计划不仅包含 padding，还包含设备绑定的 projection/GEMM 选择和离线 tile；在“不做自定义 tile”的约束下不能直接复用。

### 4.3 尚未完成、需要 profile 后 A/B 的通用优化

#### BigVGAN alias-free 融合

仓库已有不依赖 FP8 的 Triton `upsample + Snake` 融合，可以通过 `acoustic_kernels="alias"` 单独启用，不需要 GEMM tile 搜索。

它尚未进入 SM89 部署的原因是还没有完成该路径的数值、完整模块和 graph-enabled 首 chunk A/B，而不是存在硬件限制。它应进入 profile 后的首批候选。

#### Acoustic/AR overlap

当前 `overlap_acoustics=false`。此功能不依赖 SM120，也不使用第二张 GPU，但必须区分适用场景：

- `head_batch_barrier=true` 时，首段会等待整组 rows 全部 ready；
- 此时进入声学阶段前通常已经没有 `remaining rows`，所以首 chunk 上没有 AR 可以与 CFM/Vocoder 重叠；
- overlap 主要可能改善 variable tail、非 barrier 首段或请求进度不一致的场景；
- 在单张 4090 上，声学和 AR 也可能争抢 SM、Tensor Core 或显存带宽，导致负收益。

因此 overlap 必须在长文本、多 chunk 和实际 tail workload 上单独 A/B。不能用 barrier 首 chunk 结果判断它，也不能因为 SM120 默认开启就直接视为 SM89 收益。

#### Batched Proposal RNG

该优化不依赖自定义 tile，可以复用，但原实现声明 `request_seed_bitwise=false`：批量采样的统计分布正确，不保证与逐请求 seed 路径 bitwise 一致。

启用前必须确认产品语义：

- 若固定 seed 必须产生完全一致的 token/PCM，则保持关闭；
- 若只要求统计分布、音质和请求间独立性，则完成质量及性能 A/B 后可启用。

#### BF16 residual + norm 融合

SM120 的 Target norm→quant 和 residual epilogue 与 FP8 packed weight、自定义 GEMM 绑定，不能原样用于 BF16 cuBLAS。可移植的是融合思想：实现不含 GEMM tile 的 `residual add + LayerNorm/RMSNorm (+ AdaLN)` Triton reduction/pointwise kernel。

只有 profile 证明该链路在首 chunk 关键路径占比足够高时才实现。

#### Draft BF16 QKV 后处理融合

原 SM120 Draft QKV fusion 要求 FP8 packed weight，不能作为 BF16 开关直接启用。可以另行实现只融合 split/layout/RoPE 的 BF16 后处理 kernel，但必须先确认 Draft QKV 后处理是累计热点。

### 4.4 明确排除的 SM120 专用部分

以下路径包含 SM120 MMAv2/Gluon、shared-memory staging、buffer rotation、fragment prefetch、shape-specific tile 或设备/源码 hash 绑定的离线计划，属于当前明确不做的自定义 tile/流水线 kernel：

- FP8 implicit Conv；
- BF16 custom Conv tile plan；
- acoustic pipeline 与多级 acoustic refinement；
- AR pipeline 中的设备绑定 GEMM；
- Target Full-M、Target Seven 专用 projection/attention 计划；
- SM120 prefix padding projection plan；
- SM120 plan 文件的直接复用。

### 4.5 当前能力矩阵

| 能力 | SM89 当前状态 | 后续动作 |
| --- | --- | --- |
| BF16 cuBLAS/cuDNN | 已启用 | 保持基线 |
| Slot Target/Draft、request-owned KV | 已启用 | profile 验证占比 |
| Target/Draft/Proposal graphs | 已启用 | profile graph replay 与显存 |
| 48/80 Prefill/Latent buckets | 已启用 | 不重复移植 |
| First-head CFM/Vocoder graphs | 已启用 | B1/B8/B16 profile |
| Head batch barrier | 已启用 | 分析吞吐与等待成本 |
| CFM RMS/AdaLN、RoPE 融合 | 已启用 | 端到端归因 |
| BigVGAN alias-free 融合 | 尚未 A/B | profile 后首批候选 |
| Acoustic/AR overlap | 尚未 A/B | 重点测 variable tail |
| Batched Proposal RNG | 尚未启用 | 先确认固定 seed 语义 |
| BF16 residual+norm 融合 | 尚未实现 | 热点成立后实现 |
| Draft BF16 QKV 后处理融合 | 尚未实现 | 热点成立后实现 |
| SM120 prefix padding 专用计划 | 不直接移植 | 基础 48/80 桶已覆盖通用部分 |
| FP8/custom GEMM/Conv pipeline/Full-M | 当前排除 | 除非用户改变范围 |

## 5. 开始 profile 的硬性前置条件

开始前逐项确认：

- [ ] 选中的物理 GPU 显存占用不高于驱动空载值，GPU utilization 为 0。
- [ ] `nvidia-smi --query-compute-apps` 中没有其他计算进程。
- [ ] 取得项目的 cooperative GPU lease。
- [ ] 整个 profile 期间保持同一个服务/模型进程，不在 batch 档位之间释放 GPU。
- [ ] 对外部进程无法实现硬隔离时，使用显存保护；若仍发生外部负载，整组数据作废。
- [ ] 只暴露一张物理 GPU，确认 `torch.cuda.device_count() == 1`。
- [ ] 固定 GPU clocks/power policy 的当前状态并记录；无管理员权限时不得声称 clocks 已锁定。
- [ ] 固定 reference audio、文本、seed、配置文件及代码 revision。
- [ ] 完成 warmup，排除模型加载、Triton JIT 和 CUDA Graph capture。
- [ ] 先运行 SM89 preflight，确认未加载任何 SM120 plan。

## 6. 测试矩阵

正式基线：

| 并发 | Head barrier | Head graphs | 用途 |
| ---: | :---: | :---: | --- |
| 1 | yes | yes | 单请求低延迟基线 |
| 8 | yes | yes | 主生产候选 |
| 16 | yes | yes | 高并发上限候选 |

B32 已知在约 48 GiB 上捕获 BigVGAN head graph 会 OOM，因此从本计划、manifest 支持范围和所有性能对比中排除。不得用 graph-disabled/eager fallback 与 B1--B16 的完整 graph 路径比较，也不把 B32 容量路径作为本轮交付项。

每种配置至少：

- 2 次不计入结果的完整 warmup；
- 10 次稳定测量；
- 报告 median、p90、p95、min/max，不只报告 mean；
- 记录每个请求的实际 `cfm_batch`、`vocoder_batch`、AR rounds 和 accepted-token 数。

## 7. 第一层：端到端和宏观模块时间

使用 CUDA Event 或等价的 stream-aware GPU 计时，不能直接使用现有 host `perf_counter()` 作为 GPU 时间。

需要分别测量：

1. text/frontend CPU 时间；
2. GPT Prefill；
3. Draft proposal；
4. Target verification；
5. acceptance/sample 及必要的 D2H/sync；
6. 全部 AR rounds 累计时间；
7. latent；
8. acoustic condition/length regulator；
9. CFM step 1；
10. CFM step 2；
11. BigVGAN vocoder；
12. PCM D2H；
13. scheduler/host gap；
14. 从统一起点到第一个请求首 chunk；
15. 从统一起点到全部请求首 chunk。

CUDA Event 应记录在实际执行 stream 上，只在测量窗口之外统一 synchronize。若存在 acoustic stream，必须分别记录 stream 依赖和重叠，不能简单相加成关键路径。

输出表：

| Batch | 阶段 | GPU ms/call | calls | 累计 GPU ms | 关键路径占比 | CPU/同步 ms |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |

## 8. 第二层：算子和 CUDA timeline

### Nsight Systems / NVTX

启用现有 `trace_ranges`，补齐必要的 NVTX range，采集 warmup 后的稳定请求：

- CUDA Graph replay 与 graph 外 eager 路径；
- kernel timeline、launch gap 和 CPU 阻塞；
- stream 间 wait/event；
- H2D/D2H 和隐式同步；
- allocator 活动；
- CFM 两个 step；
- Vocoder 各 stage；
- 每轮 Draft/Target verification。

同时保留两类诊断：

1. graph-enabled：代表生产关键路径；
2. graph-disabled：仅用于恢复 PyTorch operator/module 归因。

两类数据不得混在同一性能表中。

### PyTorch Profiler

在 graph-disabled 诊断运行中启用 CPU/CUDA activities、record shapes、module/range 标记。输出：

- CUDA self time 排名前 30 的 operator；
- CUDA total time 排名前 30 的 operator；
- 调用次数；
- 实际输入 shapes；
- 分模块聚合结果。

## 9. 第三层：硬件瓶颈与阻塞

只对前两层确认的累计时间最高的 3–5 个 kernel 使用 Nsight Compute，不做全链路盲扫。

每个热点至少记录：

- achieved occupancy；
- SM active / Tensor Core utilization；
- DRAM throughput 及理论峰值比例；
- L2 hit rate；
- registers/thread；
- shared memory/CTA；
- active/eligible/issued warps；
- top warp stall reasons；
- grid/CTA 数量与 wave 数；
- kernel launch 数及平均/累计时间。

判定标签只能来自计数器证据：

- compute-bound；
- memory-bandwidth-bound；
- latency/dependency-bound；
- occupancy/register/shared-memory-bound；
- launch-bound；
- synchronization/serialization-bound；
- batch/shape under-utilization。

## 10. 功耗与能耗

短 kernel 不能直接依赖 `nvidia-smi power.draw.instant` 做模块归因。模块功耗测试应：

1. 记录空载/P-state/clocks/温度基线；
2. 将被测模块隔离并重复到至少 1–2 秒；
3. 持续采样 NVML board power；
4. 报告平均功耗、峰值功耗、基线扣除后的能耗；
5. 重点报告 `J/request` 和 `J/first chunk`，而不是只报告 W；
6. 监测温度、SM clock 和 power/thermal throttle，发现降频则数据作废或单列。

输出表：

| Batch/模块 | 平均 W | 峰值 W | GPU 时间 | J/call | 温度 | SM clock | 是否降频 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | :---: |

## 11. 正确性门槛

任何诊断或优化前后都必须验证：

- [ ] 首 chunk 为 11,264 samples；
- [ ] token/KV/accepted-prefix/EOS 行为不变；
- [ ] 固定 seed 的 PCM 与 BF16 基线一致，或在事先定义的容差和音质门槛内；
- [ ] B1/B8/B16 都通过；
- [ ] 无首请求优先级、barrier 或调度语义的意外改变；
- [ ] 无在线 tuning；
- [ ] 无 SM120 plan 泄漏到 SM89。

## 12. 优化决策规则

完成 profile 后，候选按以下方式排序：

```text
优先级 = 首 chunk 关键路径累计时间 × 保守可降低比例
```

不得仅因以下指标好看就选择优化：

- 单个 kernel 微基准加速比；
- occupancy 上升；
- stall 比例下降；
- 理论 FLOPS 更高；
- FP8/Tensor Core 可用；
- 更大的 batch 或 tile。

候选只有同时满足以下条件才能进入部署：

1. 完整模块累计时间下降；
2. graph-enabled 首 chunk latency 改善；
3. 正确性门槛通过；
4. 显存和功耗没有不可接受的回退；
5. 在 B1/B8/B16 中明确说明适用范围和 fallback。

## 13. 暂存候选，不代表已确定优先级

以下项目必须等 profile 后重新排序：

- BigVGAN alias-free upsample + Snake 融合；
- CFM residual add + Adaptive RMSNorm/AdaLN 融合；
- CFM 两步之间不变量及 cross-attention K/V 缓存；
- CFM cross-attention Q/K/V layout + RoPE 融合；
- B32 selective graph/eager fallback 和 CUDA Graph 显存策略（本轮明确排除）。

当前不继续投入：

- FP8 `torch._scaled_mm`；
- Triton W8A8 GEMM；
- 动态 quantize + GEMM；
- LayerNorm + FP8 quant；
- SiLU×gate 单独融合；
- SM120 shape/tile plan 直接移植。

## 14. 下次开始时的执行顺序

1. 阅读本白板和项目 `AGENTS.md`。
2. 检查 `git status`，保留现有用户修改。
3. 检查所有 GPU，等待真正空闲的一张卡。
4. 占用且只暴露这一张 GPU，保持模型进程常驻。
5. 运行 preflight，记录硬件、软件、配置和代码 revision。
6. 先采集 B1/B8/B16 宏观 CUDA Event 基线。
7. 再采集 Nsight Systems/NVTX timeline。
8. 用 graph-disabled PyTorch Profiler 做算子归因。
9. 对 top 3–5 kernels 使用 Nsight Compute。
10. 做模块重复功耗/能耗测试。
11. 汇总热点表和阻塞证据。
12. 根据决策规则提出下一项优化；未完成上述步骤前不新增生产算子。

## 15. 最终应交付的结果

- B1/B8/B16 首 chunk latency 分布；
- 各模块 GPU 时间、调用次数、关键路径占比；
- top kernels 和实际 shapes；
- top kernels 的硬件计数器与瓶颈分类；
- 模块功耗及 `J/first chunk`；
- 显存峰值、graph pool 占用；
- 调度等待和 barrier 影响；
- 有证据排序的优化候选列表；
- 原始 profile 命令、结构化 JSON/CSV 和可复查报告。

## 16. 2026-09-22 Draft/Target TensorRT 11.3 实验结论

原 32 请求结论见 [TENSORRT113_DRAFT_TARGET_RESULT.md](TENSORRT113_DRAFT_TARGET_RESULT.md)，160 请求复测见 [TENSORRT113_TARGET_160_RESULT.md](TENSORRT113_TARGET_160_RESULT.md)。下次以 160 请求、15 秒稳态功耗结果为准，不要重复逐 MLP/逐 attention 小 engine 实验：

- B1 固定批服务可使用 `configs/sm89_bf16_trt113_target_b1.json`。
- B4 固定批服务可使用 `configs/sm89_bf16_trt113_target_b4.json`。
- B16 固定批服务可使用 `configs/sm89_bf16_trt113_target_b16.json`。
- B8 若优先延迟可使用 `configs/sm89_bf16_trt113_target_b8.json`；若优先能效则保持 current，因为 TRT 的单请求能耗增加 2.62%。
- 每个固定 batch 独立启动进程并只加载对应 engine；不要在一个进程里常驻多个 batch engine。这里的要求是隔离变量和减少显存，不再把“额外 GDDR 必然拖慢执行”当作未经 profile 验证的因果结论。
- 160 请求首 chunk P50：B1/B4/B8/B16 分别降低 9.07%/4.73%/4.56%/4.43%；15 秒稳态吞吐分别增加 7.65%/7.24%/3.57%/3.77%。
- TRT 平均功耗分别增加 5.81%/2.38%/6.29%/0.80%；单请求能耗分别变化 -1.71%/-4.53%/+2.62%/-2.87%。
- B4 的 40 个 device round 中有 1 次 fallback，正式结果保留其真实影响；其余 batch 无 fallback。
- 先前 32 请求下“B8/B16 无收益”的判断已被更大样本复测推翻，不再作为选型依据。
- Draft 旧版系统性退化已确认是实现 bug：变量长度初始 context 走 fallback 时只写主 KV pool，没有写 TRT compact mirror；此前归因为 BF16 tactic 数值差异是错误的。
- 修复后真实状态 KV `max_abs=0`，logits cosine 约 0.99997--0.99999；全新 32 文本 B1/B4 的 rounds 和 accepted_sum 已恢复到当前 Draft 同一水平。
- 修复版 B1 P50 42.414 ms、rounds 7.063、accepted 25.531；B4 P50 73.970 ms、rounds 7.688、accepted 24.375。详见 [TENSORRT113_DRAFT_REINVESTIGATION.md](TENSORRT113_DRAFT_REINVESTIGATION.md)。
- 旧版 B1/B4/B8/稳定累加结果均带错误 cache，不得用于 TRT Draft 选型；下一轮重测修复版 B1/B4/B8/B16 的 160 请求及功耗。
- Target B1 生成完整性已通过 32/32；PCM 非 bitwise，默认上线前仍需 ASR/说话人相似度/主观听测。
