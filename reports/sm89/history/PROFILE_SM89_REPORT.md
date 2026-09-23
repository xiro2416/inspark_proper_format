# RTX 4090 / SM89 Profile 报告

> 本文件是 [`PROFILE_SM89_WHITEBOARD.md`](PROFILE_SM89_WHITEBOARD.md) 的执行结果。
> 原始数据在 `outputs/profile_sm89/`，所有结论都可由该目录下的结构化 JSON/CSV 与本文记录的
> 命令复现。**第 5–9 节依赖仍在采集中的第三层与功耗数据。**

会话日期：2026-09-18。物理 GPU：6（单卡独占）。

---

## 0. 执行摘要

**白板五个问题的答案：**

1. **首 chunk 时间花在哪** — AR 投机解码循环占 B1 的 **65.8%**、B8 的 **50.1%**、B16 的 **50.6%**；
   声学阶段（latent+condition+CFM+vocoder）占 B1 的 26%、B8 的 40%、B16 的 39%；
   前端（text_prepare+prefill）占 7%/9%/9%。完整分解见 §2。
2. **每个模块的 top 算子与调用次数** — 见 §4.4/§4.5。没有单一主导 kernel（最大仅 13.4%）；
   时间散在大量中小 kernel 上。Target 验证一次首 chunk 就要启动 6156 个 kernel，
   其中 864 次 `cutlass wmma bf16` GEMM **每次只用 8 个 CTA**（128 SM 上）。
3. **热点属于哪类瓶颈** — 见 §4.6 与 §5.4。**分三类**：(a) 小 grid + 低占用率上限的
   GEMM/attention（Target/CFM，延迟与 shape 利用率受限，且 roofline 证明已是 memory-bound、
   **没有 kernel 级空间**）；(b) 大 grid 的 BigVGAN 卷积族（吞吐受限）；
   (c) 约 5700 次单 CTA 的 elementwise/copy（launch 受限）。
   **但白板 §9 要求的计数器级标签（DRAM%、L2 hit、stall reason）本环境拿不到，见 §5。**
4. **B1/B8/B16 的吞吐、首 chunk 延迟、功耗与每首 chunk 能耗** — 见 §2.1：
   71.45 / 175.97 / 381.91 ms，9.95 / 4.38 / 4.79 J/request。
5. **下一项优化应按什么顺序** — 见 §9。**排序第一是消除 accept/commit 每轮的 host↔device
   往返与显式同步**（B1 关键路径 19.3 ms = 28.7%，数值风险低）。**明确否决
   Target 小 M GEMM 的 kernel 级优化**（roofline 上限只有 1.3–2.9×）。

**三个可能改变决策的发现：**

- **AR 循环的 GPU 占用率只有 21–50%**（§4.2），而声学排空阶段达 97–99%。这意味着 AR 循环内
  任何「减少 GPU 工作量」的优化在 host 路径修好前都不会转化成墙钟下降——这条前置约束
  重塑了整个候选排序。
- **投影 GEMM 是 memory-bound，实测已达自身 DRAM roofline 的 35–79%**（§5.4），
  且权重流量与 batch 无关（每轮固定 ~950 MiB）。**唯一还有大空间的方向是少搬字节，不是算得更快。**
- **CUDA Graph 捕获，而不是模型，是显存主因**：模型 7,351 MiB，而 graph 捕获在 B8 加
  **+11,120 MiB**、B16 加 **+21,064 MiB**。据此推算 B32 约需 +43 GiB，超过 48 GiB——
  这解释了 SM89.md 记录的 B32 OOM。

**两个未达成的交付：**

- **白板 §9（第三层 ncu）无法完成**：环境是 Docker 容器且内核模块设了
  `RmProfilingAdminOnly=1`，root 也拿不到 CUPTI 计数器权限。已用基于 shape 的 roofline 推算
  作部分替代，并明确标注为推算而非测量。详见 §5。
- **白板 §5 的硬隔离要求无法满足**：共享主机，GPU 0 在 profile 期间持续有其它租户负载。
  数据在同一会话内自洽，但不能作为对外承诺的绝对值。详见 §1.1。

---

## 1. 会话 identity 与前置条件

数据源：`outputs/profile_sm89/identity.json`。

| 项 | 值 |
| --- | --- |
| 设备 | NVIDIA GeForce RTX 4090，SM89，128 SM，48 GiB |
| 驱动 / CUDA runtime | 595.71.05 / 12.8 |
| PyTorch / Triton | 2.8.0+cu128 / 3.5.0（隔离 `.toolchains/triton350`） |
| power limit | **400.00 W**（出厂默认 450.00 W，已被下调） |
| clocks | **未锁定**（auto boost，`clocks.max.sm=3105 MHz`）；按本次决策仅记录不锁 |
| persistence | Enabled |
| git revision | `0039042` + 未提交的 SM89 改动（tracked diff sha256 记录在 identity.json） |
| deployment | `configs/sm89_bf16_triton.json`（schema 1，bf16，`cfm_triton_fusions=["norm","rope"]`） |
| reference audio | sha256 `a0416cfe…` → 命中缓存 `6690e3fb`（prompt sha256 `d75b42f7…`），不触发 VAD |
| 文本 / seed / emotion | `他正在整理文件。` / 0 / 8×0.0 |
| preflight | PASS，`custom_kernel_plan=false`，未加载任何 SM120 计划 |

### 1.1 未满足的前置条件（必须在解读时保留）

白板 §5 要求「对外部进程无法实现硬隔离时…若仍发生外部负载，整组数据作废」。

**本次无法实现硬隔离。** 这是共享主机，profile 期间 GPU 0 持续处于约 68% 利用率 / 约 230 W，
GPU 1 与 GPU 7 常驻显存。每次档位采集前后都记录了 `nvidia-smi --query-compute-apps` 作为证据
（见 `run_layer1.log`）。

因此：

- 本报告所有数字是**同一会话、同一 GPU、同一代码 revision 下自洽的**；
- 但**不能声称已排除外部负载对 host CPU/PCIe 的干扰**。抖动范围已在各表中以 min/max/p90 披露；
- 若需要可用于对外承诺的绝对数字，必须在独占主机上重采。

另有一项与白板文本的偏差：白板 §5 要求「整个 profile 期间保持同一个服务/模型进程，不在
batch 档位之间释放 GPU」。代码上做不到——`config['max_batch']` 决定 CUDA Graph 捕获库存
（B1 只捕 batch 1，B8 捕 1..8，B16 捕 1..8,16），`runtime/graph_policy.py:3`。因此本次采用
**每档位一个常驻进程、依次跑完、档位之间不让 GPU 转入空闲冷态**，并在档位间等待 GPU
utilization 回落到 ≤10% 再启动下一档。

---

## 2. 第一层：端到端与宏观模块时间（白板 §7）

两遍采集：

- **pass A**（`--profile-stages`）：权威延迟基线。CUDA Event 计时，无额外插桩。
- **pass B**（`--profile-stages --trace-ranges`）：把 `speech_codes_d2h` / `pcm_d2h` 也纳入
  `phase()` 包裹，用于闭合归因。带 NVTX/record_function 开销，**不得替代 pass A 的延迟表**。

每档 2 次 warmup + 10 次稳定测量，各自独立常驻进程。

### 2.1 延迟、吞吐与功耗（pass A）

| | B1 | B8 | B16 |
| --- | ---: | ---: | ---: |
| 全组首 chunk 均值 | 71.45 ms | 175.97 ms | 381.91 ms |
| 全组 median / p90 | 71.07 / 72.85 | 175.42 / 180.13 | 383.52 / 384.98 |
| 全组 min–max | 70.84–72.92 | 172.53–180.32 | 368.03–385.47 |
| 单请求首 chunk median | 71.07 ms | 174.93 ms | 382.75 ms |
| 单请求 p95 / max | 72.88 / 72.92 | 180.01 / 180.32 | 384.76 / 385.47 |
| 实测 cfm_batch / vocoder_batch | 1 / 1 | 8 / 8 | 16 / 16 |
| 首 chunk samples | 11264 | 11264 | 11264 |
| 板功耗 mean / peak | 139.4 / 171.7 W | 199.5 / 301.8 W | 200.7 / 361.5 W |
| J/request（实测，非推导） | 9.95 | 4.38 | 4.79 |
| 每档 wall time | 36 s | 47 s | 52 s |

`cfm_batch`/`vocoder_batch` 是从 chunk 事件字段读出的真实批量，不是把并发数 `batch=N` 直接当成
`cfm_batch=N`（白板 §3 明确禁止后者）。

### 2.2 模块 GPU 时间（pass A，每档每次运行）

| 阶段 | B1 GPU ms | B1 占比 | B8 GPU ms | B8 占比 | B16 GPU ms | B16 占比 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `draft_verify_accept`（AR 循环） | 46.99 | 65.8% | 88.20 | 50.1% | 193.39 | 50.6% |
| `vocoder` | 8.33 | 11.7% | 35.74 | 20.3% | 89.97 | 23.6% |
| `latent` | 3.69 | 5.2% | 12.50 | 7.1% | 23.61 | 6.2% |
| `cfm2` | 4.58 | 6.4% | 10.73 | 6.1% | 19.17 | 5.0% |
| `condition` | 1.05 | 1.5% | 9.65 | 5.5% | 15.70 | 4.1% |
| `text_prepare`（CPU） | 2.55 | 3.6% | 8.48 | 4.8% | 21.37 | 5.6% |
| `prefill` | 3.19 | 4.5% | 7.28 | 4.1% | 13.77 | 3.6% |

AR 循环内部（每轮 GPU ms，pass A）：

| span | B1 | B8 | B16 |
| --- | ---: | ---: | ---: |
| `verify`（Target 验证 8 个位置） | 2.486 | 3.162 | 5.341 |
| `accept_commit` | 1.811 | 4.049 | 4.794 |
| `draft`（Draft backbone + RNN Proposal） | 0.855 | 1.470 | 2.597 |
| 每轮合计 | 5.15 | 8.68 | 12.73 |

### 2.3 跨档位伸缩：哪些阶段是延迟受限，哪些是吞吐受限

按「每请求 GPU ms」归一：

| 阶段 | B1 | B8 | B16 | B8/B1 | B16/B1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `draft_verify_accept` | 46.99 | 11.03 | 12.09 | 0.23 | 0.26 |
| `cfm2` | 4.58 | 1.34 | 1.20 | 0.29 | 0.26 |
| `prefill` | 3.19 | 0.91 | 0.86 | 0.29 | 0.27 |
| `latent` | 3.69 | 1.56 | 1.48 | 0.42 | 0.40 |
| `text_prepare` | 2.55 | 1.06 | 1.34 | 0.42 | 0.52 |
| `vocoder` | 8.33 | 4.47 | 5.62 | 0.54 | 0.67 |
| `condition` | 1.05 | 1.21 | 0.98 | **1.15** | **0.94** |

三条可直接使用的结论：

1. **B1 上几乎所有阶段都远离吞吐上限**：并发×8 后每请求成本降到 0.23–0.42。B1 的首 chunk 由
   固定开销（kernel launch、小 M GEMM、同步）主导，不能靠「换更快的 kernel」解决，只能减少
   每轮/每模块的固定动作。
2. **`vocoder` 是受益最小的模块**（0.54× / 0.67×），且 B16 的每请求成本反而高于 B8
   （4.47 → 5.62 ms）。它是最接近吞吐饱和的那个。
3. **`condition` 完全不受益于 batching**（≈1.0×）。它是逐请求的 Python 调用，是并发无法摊薄的
   串行成本。

### 2.4 Barrier：吞吐与等待成本（白板 §4.5 要求）

`head_batch_barrier=true` 时，首段等待整组 rows 全部 ready
（`streaming/engine.py:171-173`），所以 AR 循环次数取组内**最慢成员**，而中位请求需要的轮数远少于此。

| | B1 | B8 | B16 |
| --- | ---: | ---: | ---: |
| AR 组轮数（=最慢成员） | 9 | 10 | 15 |
| 单请求 AR 轮数 median | 9 | 7 | 9 |
| 单请求 AR 轮数 min–max | 9–9 | 5–10 | 5–15 |
| 陪跑的轮数 | 0 | 3 | 6 |
| 折算成本 | 0 ms | ≈26.5 ms（15.0%） | ≈77.4 ms（20.3%） |

即：B16 每一组必然包含一个需要 15 轮的请求，而中位请求只要 9 轮，其余 6 轮全组陪跑。

上表的"折算成本"只算了**组内陪跑的轮数**，是 barrier 成本的一个下界。barrier 对**首包**的真实代价
要大得多，见 §7 的 A/B：按第一个请求就绪算，B8 慢 2.13×、B16 慢 3.08×。

---

### 2.5 CUDA Graph vs 全 eager：这张图值多少

采集条件与 pass A 完全相同（2 warmup + 10 次、同一 GPU、同一 reference、同一进程策略），
唯一差别是用 `outputs/profile_sm89/deployment_nographs_diagnostic.json` 关掉全部 5 类 graph
（target/draft/proposal/prefix/head），即完全 eager。

| 配置 | B1 首包 | B8 首包 |
| --- | ---: | ---: |
| graph-enabled | 71.45 ms | 175.97 ms |
| 全 eager | **210.91 ms** | **341.58 ms** |
| 倍数 | **2.95×** | **1.94×** |

分阶段（B1）：

| 阶段 | graph ms | eager ms | 倍数 |
| --- | ---: | ---: | ---: |
| `draft_verify_accept`（9 轮） | 46.99 | 139.14 | 2.96× |
| `vocoder` | 8.33 | 26.38 | 3.17× |
| `cfm2` | 4.58 | 19.68 | 4.30× |
| `prefill` | 3.19 | 11.01 | 3.45× |
| `latent` | 3.69 | 9.57 | 2.59× |
| `text_prepare`（纯 CPU） | 2.55 | 2.66 | 1.04× |
| `condition` | 1.05 | 1.23 | 1.17× |

AR 单轮拆解（B1）：

| span | graph ms | eager ms | 倍数 |
| --- | ---: | ---: | ---: |
| `verify` | 2.486 | 9.929 | **4.00×** |
| `draft` | 0.855 | 3.453 | **4.04×** |
| `accept_commit` | 1.811 | 1.964 | **1.08×** |

三条结论：

1. **`verify` / `draft` 是 4.0×**——它们是纯 kernel 序列（`verify` 的 GEMM grid 只有 8 个 CTA、
   `_attention` grid=1，单轮上千次 launch），graph 的价值全在摊薄 launch 开销。
2. **`accept_commit` 只有 1.08×**——独立印证了 §4.2 的判断：它本来就是 host/同步受限。
   **graph 能摊薄 launch，摊薄不了 D2H 往返与 generator 状态往返**。这也是候选 C1 与
   CUDA Graph 正交的直接证据：C1 的收益不会因为已经开了 graph 而变小。
3. **`text_prepare` 1.04×**——纯 CPU 文本归一化，与 graph 无关，符合预期。

能耗：eager 的板功耗**更低**（B1 99.9 W vs 139.4 W），因为 launch 受限时 GPU 大量空转；
但 **J/request 更高**（B1 21.07 vs 9.95 = 2.1×；B8 6.60 vs 4.38 = 1.51×）。
即 eager 既慢又费电。

归因缺口的一致性检查：graph-enabled B1 的未归因缺口是 19.8%（§3），**eager B1 只有 0.7%**。
机制一致——eager 下 host 始终是瓶颈、没有异步积压可排空，所以 `pcm_d2h` 的等待窗口几乎消失。
这从反面印证了 §3 对缺口的解释。

---

## 3. pass A 的「20–30% 未归因」是什么（重要修正）

pass A 各阶段 host 窗口之和与端到端实测存在缺口：

| | B1 | B8 | B16 |
| --- | ---: | ---: | ---: |
| pass A 缺口 | 14.12 ms（19.8%） | 50.26 ms（28.6%） | 114.61 ms（30.0%） |
| pass B 缺口 | **1.20 ms（1.8%）** | **3.50 ms（2.1%）** | **6.13 ms（1.6%）** |

缺口来源已确认为 pass A 未包裹的阶段：

| 阶段 | B1 host ms | B8 host ms | B16 host ms |
| --- | ---: | ---: | ---: |
| `pcm_d2h` | 12.64 | 46.62 | 109.56 |
| `speech_codes_d2h` | 0.08 | 0.47 | 0.89 |

**关键点：这不是隐藏的额外开销。** `pcm_d2h` 的 host 时间精确等于此前异步入队的声学工作的排空：

| | `pcm_d2h` host | `cfm2` + `vocoder` GPU |
| --- | ---: | ---: |
| B1 | 12.64 ms | 4.58 + 8.33 = 12.91 ms |
| B8 | 46.62 ms | 10.73 + 35.74 = 46.47 ms |
| B16 | 109.56 ms | 19.17 + 89.97 = 109.14 ms |

即 host 在 `pcm_d2h` 里等待的是已经在 `cfm2`/`vocoder` 计过的 GPU 工作。因此：

- **关键路径应按各阶段 host 窗口求和**（互不重叠、串行）；
- **`gpu_ms` 是 stream 占用，直接求和会重复计算**（声学排空被算了两次）；
- 判读规则：`host ≫ gpu` = 该阶段 GPU 空闲等 host（host/同步受限）；`gpu ≫ host` = 该阶段只入队，
  其 GPU 工作会在后面的阻塞阶段排空。

---

## 4. 第二层：算子归因与瓶颈分类（白板 §8）

### 4.1 采集方式与边界

- **graph-enabled**（生产路径）：`nsys_graph_b1` / `nsys_graph_b8`，`--cuda-graph-trace=node`。
- **graph-disabled**（算子归因）：`nsys_nographs_b1`、`trace_nographs_b1/b8`（PyTorch Profiler）。
  关闭全部 CUDA Graph，否则 replay 会隐藏单个算子。
- **两类数据不得混入同一性能表**（白板 §8）。nsys 自身插桩会膨胀 wall time——实测
  `text_prepare` 在 nsys 下为 18.30 ms 而 CUDA Event 为 1.80 ms，约 10×。nsys 只用于
  **结构、kernel 归属与占比**，不用于延迟。

### 4.2 每个 phase 的 GPU 占用（nsys，graph-enabled）

| phase（B1） | wall ms | kernel ms | GPU busy | 说明 |
| --- | ---: | ---: | ---: | --- |
| `draft_verify_accept`（含子 range，会重复计数） | 78.47 | 30.15 | 38.4% | |
| `verify` | 15.30 | 2.28 | 14.9% | GPU 大量空闲 |
| `draft` | 11.67 | 4.58 | 39.2% | |
| `accept_commit` | 49.45 | 21.95 | 44.4% | GPU 空闲过半 |
| `cfm2`（只入队） | 2.37 | 0.03 | 1.1% | 工作在后面排空 |
| `vocoder`（只入队） | 4.35 | 4.27 | 98.0% | |
| `pcm_d2h` | 9.50 | 9.21 | **96.9%** | 确认为排空阶段 |

（`draft_verify_accept` 是外层 range，其 kernel 数包含子 range，不能与子 range 相加。）

B8 同类数据：`draft_verify_accept` 26.6% busy、`accept_commit` 20.8%、`draft` 22.3%、
`verify` 49.8%、`pcm_d2h` **98.5%**。

**结论**：AR 循环的 GPU 占用率只有 21–50%，而声学排空阶段接近 100%。AR 循环不仅是时间大头，
还是**利用率最低**的部分——这直接指向 host/同步侧优化，而不是 kernel 侧。

### 4.3 单次首 chunk 的 host API 计数（nsys）

| | B1 | B8 |
| --- | ---: | ---: |
| `cudaMemcpyAsync` 次数 | 534 | **1252** |
| `cudaStreamSynchronize` 次数 | 103 | **211** |
| `cudaGraphLaunch` 次数 | 31 | 34 |
| `cudaLaunchKernel` 次数 | 1480 | 4821 |
| 实际 GPU 拷贝时间（D2H+H2D） | 0.06 ms | 0.14 ms |

`cudaGraphLaunch` 次数与代码完全对上：B1 = 9 轮 × 3（Target/Draft/Proposal graph）+ 4（prefix prefill/latent、
CFM、vocoder）= 31；B8 = 10 轮 × 3 + 4 = 34。

注意对比：**graph-disabled 下同样一次首 chunk 需要 15,600 次 kernel launch**（`cudaLaunchKernel` 13734 +
`cuLaunchKernel` 1326 + `cudaLaunchKernelExC` 539），消耗约 112 ms host API 时间。这是 CUDA Graph 在本
工作负载上的价值量级，也说明这个负载本质是 launch/同步受限。

`cudaMemcpyAsync` 的总时长由少数长调用主导（B8 最大单次 45.27 ms = `pcm_d2h` 的排空），因此**不能把总时长
当作额外开销**；可行动的是**次数**——每请求每轮有大量细小的 host↔device 往返。

### 4.4 算子排名（PyTorch Profiler，graph-disabled）

B1 按 CUDA self time（已剔除运行时自身的 `record_function` range）：

| 算子 | self ms | 次数 | 实际 shape |
| --- | ---: | ---: | --- |
| `cutlass::Kernel2<...wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2...>` | 14.694 | 1083 | — |
| `aten::addmm` | 4.030 | 216 | `[1280] [8,5120] [5120,1280]` |
| `aten::addmm` | 3.576 | 216 | `[5120] [8,1280] [1280,5120]` |
| `unrolled_elementwise_kernel<direct_copy_kernel_cuda>` | 3.537 | **1763** | — |
| `aten::addmm` | 2.954 | 216 | `[3840] [8,1280] [1280,3840]` |
| `cutlass::Kernel2<...s16816gemm_relu_bf16_64x64_32x6...>` | 2.136 | 180 | — |
| `aten::addmm` | 2.099 | 222 | `[1280] [8,1280] [1280,1280]` |
| `vectorized_layer_norm_kernel<float,float>` | 1.909 | 552 | — |
| `vectorized_elementwise_kernel<...bfloat16_copy...>` | 1.896 | **1702** | — |
| `fmha_cutlassF_f32_aligned_64x64_rf_sm80` | 1.707 | 74 | — |
| `aten::copy_` | 1.562 | 1101 | `[1,8,1280] [1,8,1280]` |
| `native_layer_norm` | 1.549 | 450 | `[1,8,1280]` |

`aten::addmm` 的 4 种 shape × 216 次 = 24 层 × 4 个投影 × 9 轮，与 Target GPT 的层数精确对上。
M 维只有 **8**（一次验证 8 个位置）。

B8（graph-disabled）：`cutlass relu_bf16_64x64_32x6` 15.334 ms ×896 成为第一，随后是 BigVGAN 的
`dgrad2d_grouped_direct` 8.658 ms ×109、`direct_copy` 4.920 ms ×1670、`cutlass_5x_cudnn fprop 128x128`
4.093 ms ×50、`conv_depthwise2d` 3.544 ms ×109、`sm80_xmma_fprop_implicit_gemm` 3.337 ms ×34、
`fmha_cutlassF_f32` 3.017 ms ×74。

### 4.5 按模块的 kernel 时间（graph-disabled，chrome trace）

| phase | B1 kernel ms | B8 kernel ms | B8 占比 |
| --- | ---: | ---: | ---: |
| `vocoder` | 9.35 | **37.22** | 36.9% |
| `verify` | 21.50 | 22.26 | 22.1% |
| `cfm2` | 4.92 | 11.09 | 11.0% |
| `latent` | 3.10 | 9.06 | 9.0% |
| `prefill` | 3.03 | 6.79 | 6.7% |
| `draft` | 6.59 | 6.88 | 6.8% |
| `accept_commit` | 2.90 | 5.71 | 5.7% |
| `condition` | — | 1.62 | 1.6% |

合计 B1 51.71 ms / 15682 次 kernel；B8 100.78 ms / 18272 次。

**关于 module 层级归因**：本次 torch 2.8.0+cu128 的 `torch.profiler` 即使设置
`with_modules=True` 也不向 chrome trace 写入 `Module Hierarchy` 字段，`FunctionEvent.modules` 同样为空
（实测 `module_events_attributed=0`，`module_attribution_coverage=0.0`）。因此分模块聚合以
**pipeline phase 粒度**给出——本运行时中 draft / verify / accept_commit / condition / cfm2 /
vocoder / latent / prefill 正是模块边界。

### 4.6 每个 kernel 的 launch 几何（chrome trace 自带，无需 ncu replay）

下表的 `occ 上限` 是 **torch 按 launch 资源推算的占用率上限**（trace 字段名 `est. achieved occupancy %`），
**不是实测 achieved occupancy**。实测值需要 ncu，见 §5 的说明。

| kernel | 次数 | ms | occ 上限 | regs | smem | grid / block | grid 覆盖 SM 数 | 瓶颈类别 |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |
| `cutlass wmma bf16 16x16x128`（B1 #1） | 1083 | 14.694 | **5%** | 120 | 16896 | `[8,10,1]` / 32 | 80 / 128 | 延迟 + shape 利用率 |
| `fmha_cutlassF_f32`（attention，**fp32**） | 74 | 1.707 | **2%** | 168 | 36352 | `[1,20,1]` / 128 | 20 / 128 | 延迟 + 资源受限 |
| `direct_copy` | 1763 | 3.537 | 9% | 31 | 0 | `[1,1,1]` / 128 | **1 / 128** | **launch 受限（单 CTA）** |
| `bfloat16_copy` | 1702 | 1.896 | 4% | 30 | 0 | `[1,1,1]` / 128 | **1 / 128** | **launch 受限（单 CTA）** |
| `layer_norm` | 552 | 1.909 | 2% | 37 | 24 | `[8,1,1]` / 128 | 8 / 128 | 延迟受限 |
| `cudnn dgrad2d_grouped_direct`（B8） | 109 | 8.658 | **67%** | 32 | 0 | `[2676,1,1]` / 1024 | 2676 / 128 | 吞吐受限 |
| `conv_depthwise2d`（B8） | 109 | 3.544 | **67%** | 64 | 0 | `[1248,1,1]` / 1024 | 1248 / 128 | 吞吐受限 |
| `cutlass_5x_cudnn fprop 128x128`（B8） | 50 | 4.093 | 14% | 230 | 49152 | `[208,1,1]` / 128 | 208 / 128 | 资源受限 |
| `sm80_xmma_fprop_implicit_gemm`（B8） | 34 | 3.337 | — | **254** | 98304 | `[2,104,1]` / 128 | 208 / 128 | **寄存器/smem 受限** |

**从这里可以确定性地分开三类工作形态**（依据是 launch 几何，不是计数器）：

1. **小 grid + 低占用率上限**：Target/CFM 的 GEMM 与 attention。grid 只有 8–120 个 CTA，在 128 SM 上
   连一个 CTA/SM 都铺不满。这是 `batch × 8 positions` 的工作量太小造成的，不是 kernel 写得差——
   **换更快的 kernel 没有空间，只能减少调用次数或增大每次调用的工作量**。
2. **大 grid + 高占用率上限**：BigVGAN 的卷积族（208–2676 个 CTA，上限 14–67%）。并行度不是瓶颈，
   要更快只能减少 FLOPs/字节数（即融合）。
3. **单 CTA 的 elementwise/copy**：约 5,700 次 `grid=[1,1,1]` 的 kernel（bf16↔fp32 转换链）。
   B1 上这部分合计约 8–9 ms，占 kernel 时间约 17%。它们每个只占 1/128 的 GPU。

**但「compute-bound / memory-bandwidth-bound / stall reason」这类标签仍然需要计数器**，
本环境拿不到（§5）。上表的分类依据是 grid 规模与资源占用，属于结构性推断，不是计数器判定。

### 4.7 算子融合层：与 GitHub SM120 版本的对照

对比对象：本地 `origin/main` = `0039042 "Publish SM120 inference-only runtime"`（已发布的 SM120 版本）
vs 当前工作树。**注意本机无网络，无法核实 GitHub 上是否已有更新的 commit。**

#### 4.7.1 当前 SM89 实际执行的融合 kernel

从 nsys 的 B8 kernel 清单中筛出非厂商、非 aten 的 kernel，逐个核对仓库里的 `@triton.jit` 定义：

| kernel | 融合了什么 | B1 次数 / GPU 占比 | B8 次数 / GPU 占比 |
| --- | --- | ---: | ---: |
| `_attention` | Target 注意力（KV 读取 + 注意力 + mask） | 216 / 2.10% | 240 / 2.20% |
| `_append` | Target KV 原地写入 | 216 / 0.50% | 240 / 0.30% |
| `_draft` | Draft 注意力 | 27 / 0.30% | 30 / 0.30% |
| `_rms_mod` | CFM `AdaptiveLayerNorm+RMSNorm+AdaLN` 调制 | 54 / 0.20% | 54 / 0.20% |
| `_rope_qkv` | CFM `wqkv` 投影 + split/layout + RoPE | 26 / 0.10% | 26 / 0.20% |

合计约 **3.2% GPU 时间**，其余全部是 cuBLAS/cutlass/cuDNN/aten。

#### 4.7.2 完整对照

**「阻塞类型」分三类，它们的工作量差一个数量级，不能混为一谈**：

- **A = 合约阻塞**：配置层写不出这个开关。改动量小（小时级），但必须按 AGENTS.md 规则 5
  「离线准备、显式版本化」走，不得放宽现有校验。
- **B = 离线产物缺失**：这些 kernel 的 tile 参数**不由代码决定，而是从计划文件读出来的**
  （例如 `stage2_gemm.col_linear` 接收 `Tile(**p['tile'])`）。而计划文件的 `identity` 同时绑定
  `DeviceCaps.current()` 的全部设备参数 + torch/triton/cuda 版本 + 源码文件 hash
  （构造方式见 `models/acoustic_kernels.py:8-13`）。
  所以问题**不是"缺一个配置文件"**，而是这个产物**本质上是设备专属的**：在 4090 上要用它，
  必须按 4090 的寄存器/SM 数/shared memory 重新做一遍离线 tile 搜索，也就是 PORTING.md 的完整流程
  （步骤 3–6）。本次范围明确排除自定义 tile（白板 §4.4），因此没有生成，也不计划生成。
- **C = 依赖被否决项**：其实现依赖 FP8 或自定义 GEMM tile，而这两者已被本次 profile 的证据否决
  （§5.4、§9.3），所以补它等于先把被否决的东西加回来。

| 融合 | 作用 | GitHub SM120 | 当前 SM89 | 阻塞类型 | 具体依据 |
| --- | --- | :---: | :---: | :---: | --- |
| `_attention` + `_append` | Target KV 写入 + 注意力 | ✅ | ✅ | — | 无差异：由 SlotTarget 无条件使用，两版共享 |
| `_draft` | Draft 注意力 | ✅ | ✅ | — | 无差异：由 SlotDraft 使用 |
| `_rms_mod` (norm) | CFM AdaLN 调制 | ✅ | ✅ | — | 无差异，但**启用路径不同**（见 §4.7.3） |
| `_rope_qkv` (rope) | CFM QKV 投影 + RoPE | ✅ | ✅ | — | 同上 |
| `_silu_mul` (gate) | FFN `silu(w1x)*w3x` | ✅ | ❌ | **实测否决** | SM89 微基准 0.75×（SM89.md），配置里主动不写 `gate`。**不是不能，是测了更慢** |
| `_up_snake` + `_down` | alias-free：`UpSample1d+Snake+LowPass` 三步 → 2 kernel，覆盖 109 个 `Activation1d` | ✅ | ❌ | **A** | `runtime/deployment.py:11` 只对 schema 2–8 接受 `acoustic_kernels`；`scripts/preflight_sm89.py:37` 又强制 `schema==1`。**不需要任何 tile 搜索**，是唯一"独立且不依赖被否决项"的可移植融合 |
| `acceptance` / `_accept` | PCG 接受判定融合 | ❌ | ❌ | — | **两边都没开**（`fused_acceptance: false`）。且 `dspark/batch_pcg.py:74` 的 `packed.cpu().tolist()` 位于融合分支之外，**开启它也不消除 host 往返** |
| `_gemm_col` + `_epilogue` | 自定义 tile GEMM + epilogue 融合 | ✅ | ❌ | **B** | tile 参数来自 RTX 6000D 的离线计划（见上）；且 §5.4 的 roofline 已证明该方向无空间，即使重搜也不值得 |
| `_bf16_conv` | BF16 conv tile | ✅ | ❌ | **B** | 同上；额外要求 stage1 已经装了 `{'alias','ntc'}`（`models/acoustic_stage2.py:44` 的硬门禁） |
| `_conv_ntc` + `_quant_ntc` | NTC 量化卷积（量化+卷积融合） | ✅ | ❌ | **B + C** | tile 参数来自计划**且**依赖 FP8；两个阻塞独立，任一都足以解释为何不开 |
| `_conv` + `_quant` + `_scale` + `_partial_max` | FP8 卷积（量化/layout/scale 融合） | ✅ | ❌ | **C** | 依赖已被实测否决的 FP8 动态量化路径 |
| `_gemm` + `_quantize` + `_reduce` | FP8 GEMM（动态量化融合） | ✅ | ❌ | **C** | 同上；SM89.md 记录动态量化路径 70.0–87.4 µs vs BF16 22.9–27.4 µs |
| `conv_pipeline` / `down_pair` / `up_pair` / wavenet `update` | acoustic pipeline / refine 的分级融合 | ✅ | ❌ | **B** | tile/分级参数来自 RTX 6000D 的离线计划 |
| Draft QKV 融合（`draft_fusion/`） | Draft QKV | ✅ | ❌ | **C** | 依赖 FP8 packed weight，即依赖已被白板 §4.1 实测否决的精度路径；`runtime/deployment.py:107` 有硬校验 `resolved_precision != 'fp8'` 即抛错 |
| `target_norm_quant` | `LayerNorm→FP8 quant` 精确融合 + residual epilogue | ✅ | ❌ | **C** | 融合写在自定义 GEMM 内部，无法摘出来贴到 cuBLAS 上（白板 §4.3） |

#### 4.7.3 两处结构性要点

**① SM120 的融合分两类，可移植性完全不同。**

- **独立融合**：`_rms_mod`、`_rope_qkv`、`_silu_mul`、`_up_snake`/`_down`、`acceptance` —— 自带
  `BLOCK` 常量，**不依赖任何 tile 计划**，换架构即可用（是否更快另说）。
- **绑定在 GEMM/Conv 内部的融合**：`_gemm_col`+`_epilogue`、`_conv`+`_quant`+`_scale`、
  `_conv_ntc`+`_quant_ntc`、`target_norm_quant` —— 量化与 epilogue **写在自定义 tile kernel 里面**。
  这正是白板 §4.3 说的「不能原样用于 BF16 cuBLAS」：不是不想摘，是**摘不出来**，只能照融合思路重写。

**② 有一处当前比 GitHub 更保守，方向是反的。**

GitHub 上 `norm`/`gate`/`rope` 三个是通过 `acoustic_stage2_plan` 一次性装上的
（`models/acoustic_stage2.py:44` 默认 `parts=('gemm','conv','norm','gate','rope')`，
且被 `_acoustic_prepared=={'alias','ntc'}` 硬门禁）。本地新增了 GitHub 上不存在的
`cfm_triton_fusions` 键，把它**解耦**出来只装 `norm`+`rope`——因为 `gate` 在 4090 上实测更慢。
所以 SM89 不是简单"少开几个开关"，而是在融合选择上**更精确**。

---

## 5. 第三层：Nsight Compute —— 本环境无法完成

**结论：白板 §9 未达成。原因不是配置错误，而是环境权限。**

ncu 启动后立即失败：

```
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU
Performance Counters on the target device 0.
```

取证如下：

| 检查 | 结果 |
| --- | --- |
| `/proc/driver/nvidia/params` | **`RmProfilingAdminOnly: 1`** —— 内核模块把性能计数器限制给管理员 |
| `/.dockerenv` | 存在，**运行在 Docker 容器内** |
| `CapEff` | `0x00000000a80425fb`，解码后**不含 `CAP_SYS_ADMIN`**（也不含 `CAP_PERFMON`） |
| `/etc/modprobe.d/` | 无 `RestrictProfiling` 覆盖 |

因此即使是 root 也不满足 CUPTI **profiling**（计数器）的权限要求。

**为什么 nsys 可以而 ncu 不可以**：nsys 走 CUPTI 的 activity/tracing API，该路径不受
`RmProfilingAdminOnly` 限制；ncu 需要 CUPTI profiling API 读取硬件计数器，受该参数限制。这解释了
本次会话中 nsys 全部成功、ncu 全部失败的现象。

**为什么不去改这个参数**：`RmProfilingAdminOnly` 是 nvidia 内核模块的加载参数，修改需要
`modprobe -r nvidia && modprobe nvidia NVreg_RestrictProfilingToAdminUsers=0`，也就是**重载内核模块**。
本机是共享主机，profile 期间 GPU 0 有其它租户在跑（约 68% 利用率），GPU 7 占着约 45 GB 显存。
重载模块会**终止这些任务**。这不仅超出我的权限（无 `CAP_SYS_ADMIN`、无法访问宿主），也是不可接受
的破坏性操作。

### 5.1 因此缺失的指标

白板 §9 要求逐热点记录的项目中，以下**无法提供**（都需要计数器）：

- SM active / Tensor Core utilization
- DRAM throughput 及相对理论峰值的比例
- L2 hit rate
- active / eligible / issued warps
- top warp stall reasons
- 实测 achieved occupancy（区别于按资源推算的占用率上限）

### 5.2 可以替代到什么程度

已获取的、**不是**计数器但属 CUPTI activity 数据的：

- 每个 kernel 的调用次数、平均/累计时长（nsys `cuda_gpu_kern_sum`）；
- 每个 kernel 的 grid/block 规模、registers/thread、shared memory/CTA（chrome trace）；
- 按资源推算的占用率上限，以及 grid 相对 SM 数的覆盖情况（§4.6）；
- 每个 phase 的 GPU busy 比例（§4.2）。

这些足以**区分工作形态**（小 grid / 大 grid / 单 CTA），也就是足以支撑 §4.6 的三分法，
以及后续候选排序中「换 kernel 有没有空间」的判断。但**不足以**给出白板要求的计数器级瓶颈标签。
本报告中所有瓶颈判断都标注了依据类型；凡未标注「计数器」的，都是结构性推断。

### 5.3 若要补齐，需要什么

三条路径，按代价排序：

1. **宿主管理员设置 `NVreg_RestrictProfilingToAdminUsers=0`**（写入 `/etc/modprobe.d/` 后重载模块）。
   需要停机窗口，会中断该主机上所有租户的 GPU 任务。
2. **给容器加 `CAP_SYS_ADMIN`** 并确认 `RmProfilingAdminOnly=0`；若模块参数仍为 1，仅加 capability 不够。
3. **换一台独占且未限制 profiling 的机器**重采第三层。

在拿到上述任一条之前，第三层结论不应被引用。

### 5.4 替代方案：基于 shape 的 roofline 推算

按本次决策，额外做了 roofline 推算（`scripts/report_sm89_roofline.py`）。**这是推算，不是计数器测量**，
前提是理想缓存、零 launch 开销、完美重叠。它的用处是判断「这个算子还有没有空间」——即「值不值得为它写更快的 kernel」。

参考峰值（RTX 4090 公开规格，dense、无稀疏）：bf16 Tensor Core + fp32 累加 **165.2 TFLOP/s**，
DRAM **1008 GB/s**，L2 **72 MiB**，128 SM。**Ridge point = 164 FLOP/byte**。

| | M | N | K | 调用 | 实测 ms | roofline ms | 达 roofline | AI | 判定 | 权重 MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| B1 | 8 | 1280 | 5120 | 216 | 4.030 | 0.0131 | **70.2%** | 7.9 | memory | 12.50 |
| B1 | 8 | 5120 | 1280 | 216 | 3.576 | 0.0131 | **79.2%** | 7.9 | memory | 12.50 |
| B1 | 8 | 3840 | 1280 | 216 | 2.954 | 0.0098 | **71.9%** | 7.9 | memory | 9.38 |
| B1 | 8 | 1280 | 1280 | 222 | 2.099 | 0.0033 | 34.8% | 7.9 | memory | 3.12 |
| B8 | 64 | 1280 | 5120 | 192 | 3.853 | 0.0138 | 68.8% | 60.2 | memory | 12.50 |
| B8 | 64 | 5120 | 1280 | 192 | 3.664 | 0.0138 | 72.4% | 60.2 | memory | 12.50 |
| B8 | 64 | 3840 | 1280 | 192 | 3.218 | 0.0104 | 62.1% | 60.0 | memory | 9.38 |
| B8 | **584** | 1280 | 5120 | 24 | 1.754 | 0.0463 | 63.4% | **371.9** | **compute** | 12.50 |

三条结论：

1. **Target/CFM 的投影 GEMM 全部是 memory-bound**，算术强度只有 8–60 FLOP/byte，远低于 ridge 164。
   唯一 compute-bound 的是 M=584 那个（预填充路径的稠密 GEMM，AI 372）。
2. **它们的实测时间已达自身 DRAM roofline 的 35–79%**。也就是说，即使换一个能跑满 100% DRAM 带宽的
   完美 kernel，理论上限也只有 **1.3–2.9×**，现实中远达不到。**GEMM kernel 级优化没有空间。**
3. **权重流量与 batch 无关**：每轮必须把 24 层 × 4 个投影的 ~950 MiB 权重重新流一遍。
   - B1：9 轮 → 8556 MiB → **DRAM 下限 8.90 ms**，占首 chunk 的 **13.2%**
   - B8：8 轮 → 8172 MiB → 下限 8.50 ms，占 4.8%
   - 实测投影 GEMM 合计 13.70 ms（B1）→ 即 **65% 的纯 DRAM 下限效率**

**由此得到唯一还有大空间的方向是「少搬字节」，而不是「算得更快」。** 把权重字节减半（例如预量化权重）
在天花板上最多省下 B1 的约 4.4 ms（6.6%）。但白板 §4.1 的微基准已经显示当前 FP8 路径净更慢
（BF16 cuBLAS 22.9–27.4 µs vs `torch._scaled_mm` 预量化 48.5–60.0 µs），说明量化与 scale 处理的开销
超过了字节节省。因此这条路只有「预量化权重 + 无逐次量化」的形态才可能成立，且**必须做整路径 A/B，
不能用 GEMM 微基准判定**——这一点与白板 §4.1 自己写的「若实际 profile 发现新热点，可以针对该精确 shape
重新验证，但不得据硬件能力直接启用 FP8」是一致的。

---

## 6. 功耗与能耗（白板 §10）

三个层次，分别说明「模块边际能耗」「整请求总能」和「benchmark 窗口内的板级能耗」。

### 6.1 模块隔离（graph-disabled，batch 8，2 s 窗口，NVML 10 ms 采样）

| 模块 | 平均 W | 峰值 W | J/call（扣基线） | ms/call | SM clock | 温度 | 降频 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| idle 基线 | 73.0 | — | — | — | 2520 MHz | 39 °C | 否 |
| `ar_round` | 101.9 | 110.8 | **0.556** | 19.22 | 2520 MHz | 41 °C | 否 |
| `cfm`（一次两步 CFM） | 181.0 | 242.4 | **2.036** | 18.86 | 2685 MHz | 48 °C | 否 |
| `vocoder` | **324.5** | **386.1** | **9.402** | 37.39 | 2685 MHz | 56 °C | 否 |

三层全部 `valid=true`，`throttle_reasons` 为空，SM clock 在窗口内稳定（`min` 与 `max` 相差 ≤6.5%），
**未发生降频，数据有效**。

注意：`ar_round` 的 19.22 ms/call 是 150 步循环的平均，KV 长度在其中持续增长，因此**不能**与第一层
首 chunk 内 8.68 ms/轮（B8）直接比较。它是稳态边际能耗，不是首 chunk 归因。

### 6.2 生产路径（graph-enabled，batch 8）

| 项 | 平均 W | 峰值 W | J/call | ms/call |
| --- | ---: | ---: | ---: | ---: |
| idle 基线（已捕获 graph） | 89.7 | — | — | — |
| `ar_round` | 118.4 | 138.5 | 0.574 | 20.1 |
| **`first_chunk`（8 请求全组）** | **201.1** | **247.2** | **17.252** | **154.9** |

→ **17.25 J/全组首 chunk，即 2.157 J/request（batch 8，扣除基线）**。

### 6.3 三种口径的区别（避免误用）

| 口径 | B1 | B8 | B16 | 含义 |
| --- | ---: | ---: | ---: | --- |
| `J/request`（benchmark 窗口内板级总能） | 9.95 | 4.38 | 4.79 | 含基线空载功耗与窗口内间隙 |
| `J/request`（扣基线边际，power 脚本） | — | 2.157 | — | 只算高于空载的那部分 |
| `first_chunk` 全程总能 | 9.95 | 17.25* | — | 整组一次 |

\* B8 的 `first_chunk` 17.25 J/组 ÷ 8 = 2.157 J/request（边际）。

两者差 ≈ 89.7 W × 0.176 s ÷ 8 ≈ 1.97 J/request，正是空载基线的贡献。**报告数字时必须说明用的是哪个口径。**

### 6.4 与能耗相关的环境事实

- power limit **400 W**（出厂 450 W），全程未变；
- clocks **未锁定**（按决策只记录）；实测窗口内 SM clock 稳定在 2520–2685 MHz，无 throttle；
- 温度最高 56 °C（vocoder 层），距降频阈值很远；
- 因此本次功耗数据**没有降频污染**，但因为是共享主机，**不能作为对外承诺的绝对值**。

### 6.5 一个交叉验证：`Pool` 边界的成本

生产路径 in-process 的 `first_chunk` 墙钟为 **154.9 ms**（batch 8），而经 `runtime.Pool` 的
benchmark 全组首 chunk 为 **176.0 ms**。差值约 **21 ms（12%）**。

机制上说得通：worker 每发布一个 chunk 就 `pipe.send` 一个含 11264 样本 int16 PCM（约 22.5 KB）的事件，
8 个 chunk 合计约 180 KB，超过 pipe 默认缓冲区，worker 会在 `send` 上阻塞等待父进程排空；父进程还要
轮询 `wait()` 并反序列化。

但这**不是一次受控 A/B**：in-process 测量同时少了父进程的 `PowerSampler` 子进程（每 20 ms 调一次
`nvidia-smi`）与 `--record-requests` 等开销。因此这 21 ms 应读作「`Pool` 边界 + 父进程记账」的上界，
**要归因到 IPC 本身需要单独做一次受控实验**。这条列在候选 C3。

---

## 7. `head_batch_barrier` A/B（白板 §4.5）

除该标志外，配置与 `configs/sm89_bf16_triton.json` 完全一致（`deployment_barrier_off_experiment.json`）。
每档 2 warmup + 10 次。

**先说明用哪个指标**：本运行时的产品定位是"首包优先"（`README.md`、`streaming/engine.py:1`
的 "first-packet-priority"）。因此**第一个请求拿到首 chunk 的延迟**才是决策指标，
而不是单请求 median——barrier 下所有请求同时就绪，median 与 max 几乎相等，会掩盖真实代价。

**指标定义（下表每一列的口径，避免歧义）**：

- **首个就绪请求** = 每轮 8 个请求中最早就绪者的延迟，取 **10 轮的均值**（不是 P50；同一组的
  median 见括号）。
- **整批就绪** = 每轮最后一个请求就绪的时刻，取 10 轮均值。
- **单请求 median / max** = 全部 10×8 = 80 个请求逐条的 P50 / 最大值。

| | **首个就绪请求**（均值 / median） | 整批就绪 | 单请求 median | 单请求 max | cfm_batch 实测 | J/request |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| B8 barrier **on** | 175.03 / 174.47 ms | **175.97 ms** | 174.93 ms | 180.32 | 8 | 4.38 |
| B8 barrier **off** | **82.19 / 81.09 ms** | 206.92 ms | 146.34 ms | 212.41 | 1–2 | 4.53 |
| B16 barrier **on** | 380.40 / — ms | **381.91 ms** | 382.75 ms | 385.47 | 16 | 4.79 |
| B16 barrier **off** | **123.31 / — ms** | 427.01 ms | 324.02 ms | 441.71 | 1–4 | 5.56 |

**读这张表要注意**：barrier **on** 时三个口径几乎相等（B8：171.58–180.32，±3%），因为 barrier
强制 8 个请求同时就绪——此时"首包"**不是**一个独立可测的指标，它恒等于整批就绪。
只有 barrier **off** 才能把首包与末包分开：B8 从 81.09 拉到 212.41，**相差 2.6 倍**。
所以判断 barrier 取舍必须用 barrier-off 行的首包，与 barrier-on 行的整批对比。

**结论（按首包指标）**：

| | 首包代价 | 整批收益 |
| --- | ---: | ---: |
| B8 | 82.19 → 175.03 ms，**慢 2.13×（+113%）** | 206.92 → 175.97 ms，快 15.0% |
| B16 | 123.31 → 380.40 ms，**慢 3.08×（+208%）** | 427.01 → 381.91 ms，快 10.6% |

**即 barrier 让首包慢 2–3 倍，只换整批 11–15% 的吞吐。** 这比"用 15% 延迟换 15% 吞吐"严重得多——
对一个首包优先的服务，这个取舍几乎肯定是反的。

机制清楚：barrier 让声学阶段能按 8/16 批量执行（`cfm_batch` 8/16），而不开 barrier 时请求各自就绪、
`cfm_batch` 掉到 1–4、反复重跑声学——**所以整批更快**；但代价是**任何一个请求都不许先走**，
于是最快的那个请求（本来 82 ms 就能出）被拖到 175 ms。

同时 J/request 是 barrier 更低（B16：4.79 vs 5.56），即 barrier 更省电。

**这不是一个纯工程优化，而是产品语义决策**：对一个"首包优先"的服务，让每个请求都等最慢的那个
（B16 每组必然包含一个需要 15 轮的请求，中位只需 9 轮）是否可接受，取决于产品对首包延迟的承诺。
本报告给出量化取舍，不替产品决定。

**方法学限制（必须披露）**：barrier-off 时每次 `run_until_first_chunk` 会发起**多次** `_advance`，
而 `Engine.take_profile()` 只保留最后一次的 `profile_spans`（`streaming/engine.py:159` 每次都重置）。
因此 **barrier-off 行的阶段分解不完整**——第 2 章表里 barrier-off 出现的 91.7%/94.4% "缺口"是这个
采样限制造成的假象，不是真实开销。**barrier-off 的性能结论只能引用延迟/吞吐/功耗与 `cfm_batch` 证据，
不能引用阶段分解。**

---

## 8. 正确性门槛（白板 §11）

| 门槛 | 结果 | 证据 |
| --- | --- | :---: |
| 首 chunk 为 11264 samples | 全部通过 | `first_chunk_samples_values` 在 B1/B8/B16 两遍 ×30 次运行中恒为 `[11264]`；`streaming/core.py:179` 内还有断言 |
| B1/B8/B16 都通过 | 是 | 30 次运行全部 exit=0，无 `error` |
| 固定 seed 的 PCM 与基线一致 | **通过** | `bf16` 与 `bf16_triton` 两配置产出**逐字节相同** PCM（46592 samples，sha256 `686d0e8cf86226c188774335e5974dc72f63f7a44e491a706c12bf5fde4654eb`），`pcm_identical=true` |
| token/KV/accepted-prefix/EOS 行为不变 | 未变（见下） | 每请求的 `accepted` 序列与 `total_codes` 已逐次记录在 `layer1_b*.json` |
| 无在线 tuning | 是 | 所有 manifest `online_learning=false`；`runtime/online_guard.py` 可拒绝在线编译/捕获 |
| 无 SM120 plan 泄漏 | 是 | `preflight_sm89.py` PASS，`custom_kernel_plan=false`；三份 SM89 配置均为 `schema 1` |

**关于「token/KV/EOS 行为不变」的说明**：本次会话**没有修改生产路径**。唯一的 `src/` 改动是
`runtime/pool.py` 的 `result` 回包追加 3 个只读键（`rounds`/`accepted`/`kv_head_lengths`），
而 `result` 只在请求完成后由 `Pool.result()` 调用，`run_ready` 调度路径从不触发它。
上面那条**逐字节 PCM 一致性**就是这一点的实证：若该改动影响了流式行为，PCM 不可能完全相同。

**仍未做的门槛**：没有与「修改前的 revision」做 accepted-token 序列的逐项比对（因为生产路径未改，
比对对象不存在）。若后续引入候选优化，必须补做。

---

## 9. 优化候选排序（白板 §12、§13）

排序规则按白板：**优先级 = 首 chunk 关键路径累计时间 × 保守可降低比例**。

但在应用这条规则之前，必须先叠加一个本次 profile 得到的**前置约束**：

> **AR 循环（占 B1 首 chunk 65.8%）的 GPU 占用率只有 21–50%**（§4.2）。也就是说 AR 循环内
> 任何「减少 GPU 工作量」的优化，在 host 路径修好之前**都不会等量转化为墙钟下降**。
> 相反，声学排空阶段（`pcm_d2h`）GPU busy 达 97–99%，那里的 GPU 节省可以近似 1:1 转化。

这条约束把候选分成两类：**host 侧（直接见效）** 与 **AR 内 GPU 侧（需先解 host 瓶颈才见效）**。

### 9.1 host 侧候选（直接作用于关键路径）

| # | 候选 | B1 关键路径 | B8 | B16 | 保守降幅 | 依据 | 数值风险 |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| **C1** | 消除 accept/commit 每轮的 host↔device 往返与显式同步 | **19.3 ms（28.7%）** | 22.5 ms（12.8%） | 26.7 ms（7.0%） | 60% | §4.2 `accept_commit` GPU busy 仅 20.8–44.4%；§4.3 534/1252 次 `cudaMemcpyAsync`、103/211 次 `cudaStreamSynchronize`；代码 `dspark/batch_pcg.py:64-67,74,128,139`、`dspark/core.py:88`、`dspark/runtime.py:45` | **低**——常驻 per-request generator 可保持 draw 顺序、bitwise 不变 |
| **C3** | `Pool` 边界：chunk 事件跨 pipe 传 22.5 KB PCM | ≈0（单请求一次往返） | ~21 ms（12%） | ~21 ms+ | 50% | §6.5 | 低 |
| **C4** | `condition` 逐请求 tensor 构造批量化 | 2.8 ms（4.2%） | 8.3 ms（5.0%） | 17.0 ms（4.4%） | 50% | §2.3 `condition` 每请求成本 1.05→1.21→0.98 ms，**完全不受益于 batching** | 低 |
| **C8** | `head_batch_barrier` 策略 | — | 见 §7 | 见 §7 | — | §7 | 无（策略） |

### 9.2 声学侧候选（GPU busy ≈100%，GPU 节省可近似 1:1 转化）

| # | 候选 | B1 | B8 | B16 | 保守降幅 | 依据 | 数值风险 |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| **C2** | BigVGAN 点式/激活链 + alias-free 融合 | 8.3 ms（12.4%） | **35.7 ms（20.3%）** | **90.0 ms（23.6%）** | 20% | §4.4/§4.5：vocoder 阶段 2716 个 kernel 中 **1721 个是 elementwise（3.28/9.39 = 35%）**、grid 仅 1–156 CTA；`dgrad2d`/`conv_depthwise`/`xmma` 是大 grid 吞吐受限 | 中（FP32 接口需保持） |

### 9.3 AR 内 GPU 侧候选（空间有限或需先解 host）

| # | 候选 | B1 | B8 | B16 | 判定 | 依据 |
| --- | --- | ---: | ---: | ---: | --- | --- |
| **C5** | bf16↔fp32 转换链（约 5700 次单 CTA kernel，B1 约 8–9 ms） | ~8.5 ms（12.6%） | — | — | **暂缓**：先受 §9 前置约束压制；且改转换位置会改数值，需先过数值门槛 | §4.4/§4.6；`models/precision.py:103-104` |
| **C6** | Target 小 M GEMM / fp32 attention 的 kernel 级优化 | 22.4 ms（33%） | 31.6 ms（18%） | 80.1 ms（21%） | **否决**：§5.4 roofline 显示已是 memory-bound 且达自身 roofline 的 35–79%，理论上限仅 1.3–2.9×；grid 只有 8 CTA / attention grid=1 是 M=8 的固有属性，换 kernel 无用 | §4.6、§5.4 |
| **C7** | 权重字节数（预量化 FP8 形态） | 天花板 4.4 ms（6.6%） | 天花板 ~4.2 ms（2.4%） | ~7 ms（1.8%） | **低优先**：天花板明确且不大；白板 §4.1 微基准已显示当前 FP8 路径净慢 2×；只有「预量化权重 + 无逐次量化」形态才可能成立，必须整路径 A/B | §5.4、§4.1 |

### 9.4 容量候选（仅在需要 B32 时才做）

| # | 候选 | 依据 |
| --- | --- | --- |
| **C9** | CUDA Graph 显存策略 / B32 selective graph-eager fallback | §6 之外的显存数据：graph 捕获占 B8 **+11,120 MiB**、B16 **+21,064 MiB**，而模型本身只占 7,351 MiB。按此推算 B32 约需 **+43 GiB** 图谱，叠加权重后超过 48 GiB——这解释了 SM89.md 记录的 B32 在 BigVGAN head-graph 捕获时 OOM。白板 §13 的这条候选现在有了具体数字：graph pool 是显存主因，不是模型。 |

### 9.5 建议的执行顺序

1. **C1**（B1 关键路径 28.7%，数值风险低）——先做，因为它同时解除 AR 循环的 host 瓶颈，
   使 C5/C6/C7 这类 GPU 侧候选才有意义。
2. **C3 的受控 A/B**——先确认那 21 ms 是否真的归属 IPC，再决定是否改造进程边界。
3. **C2**——B8/B16 的最大声学项，且 GPU busy 接近 100%，收益转化可靠。需要先解开
   `acoustic_kernels` 的合约开关（白板 §4.3 的 schema/preflight 问题，见 §10 的说明）。
4. **C4**——小但便宜，可与 C1 同批。
5. **C8**——作为产品决策并行推进，不占工程时间。
6. C5 / C7 在 C1 落地并复测后再评估；**C6 建议直接否决**。

### 9.6 与白板 §13「当前不继续投入」的一致性

本次 profile **没有**推翻白板 §13 的任何一条排除结论，并且为其中两条补上了证据：

- FP8 `torch._scaled_mm` / Triton W8A8 / 动态量化 / LayerNorm+FP8：roofline 显示投影 GEMM
  是 memory-bound，理论上限只有 1.3–2.9×，与微基准「净更慢」的结论方向一致——**继续不投入**。
- SM120 shape/tile plan 直接移植：本次未触及，`custom_kernel_plan=false`。
- SiLU×gate 单独融合：仍未 A/B，§4.5 显示 `bfloat16_copy`/`direct_copy` 这类搬运才是大头。

### 9.7 仍然悬空、需要白板更新的两处

1. **`acoustic_kernels="alias"` 在 SM89 上写不出来**（白板 §4.3 称「可以通过该键单独启用」）：
   `runtime/deployment.py:11` 只对 schema 2–8 接受该键，`scripts/preflight_sm89.py:37` 又强制
   `schema == 1`。三份 SM89 配置都无法表达这个开关。要做 C2 必须先按 AGENTS.md 规则 5
   「离线准备、显式版本化」新增一个 approved status，**不能靠放宽现有校验绕过**。
2. **白板 §3 的「作废数据」需要更新**：本次是全新采集（B1/B8/B16 × 两遍 + nsys × 3 + profiler × 2 +
   功耗 + 显存 + barrier A/B + 门槛），与 09:29–09:31 那组无关，§3 的作废结论对本次数据不适用。

---

## 10. 复现命令

全部通过 `scripts/run.sh` 执行（它负责隔离 Triton 3.5.0、`PYTHONPATH`、`HF_HUB_OFFLINE=1`）。
所有采集都固定 `--gpu 6`。原始驱动脚本在 `outputs/profile_sm89/*.sh`，本节的命令与其等价。

### 前置

```bash
CUDA_VISIBLE_DEVICES=6 bash scripts/run.sh scripts/preflight_sm89.py \
  --deployment configs/sm89_bf16_triton.json
```

### 第一层（pass A：权威延迟；pass B：归因闭合）

```bash
# pass A，每档 36-52 s
bash outputs/profile_sm89/run_layer1.sh
# pass B（加 --trace-ranges），每档 37-52 s
bash outputs/profile_sm89/run_layer1b.sh

# 报告（表 + JSON + CSV）
bash scripts/run.sh scripts/report_sm89_layer1.py \
  outputs/profile_sm89/layer1_b1.json outputs/profile_sm89/layer1_b8.json \
  outputs/profile_sm89/layer1_b16.json \
  --json-out outputs/profile_sm89/layer1_report.json \
  --csv-out  outputs/profile_sm89/layer1_report.csv
```

### 第二层

```bash
# Nsight Systems（graph-enabled B1/B8 + graph-disabled B1）
bash outputs/profile_sm89/run_nsys.sh
bash outputs/profile_sm89/extract_nsys.sh
bash scripts/run.sh scripts/report_sm89_nsys_gaps.py \
  outputs/profile_sm89/nsys_graph_b1.sqlite outputs/profile_sm89/nsys_graph_b8.sqlite \
  outputs/profile_sm89/nsys_nographs_b1.sqlite \
  --json-out outputs/profile_sm89/nsys_gaps.json

# PyTorch Profiler 算子归因（graph-disabled）
for B in 1 8; do
  bash scripts/run.sh scripts/profile_sm89_ops.py --gpu 6 --batch $B \
    --warmups 2 --graphs diagnostic --torch-profiler \
    --deployment outputs/profile_sm89/deployment_nographs_diagnostic.json \
    --ref-audio outputs/profile_sm89/reference.wav \
    --out-dir outputs/profile_sm89 \
    --json-out outputs/profile_sm89/ops_nographs_b${B}.json
done

# 从 chrome trace 抽取每个 kernel 的 launch 几何
for B in 1 8; do
  bash scripts/run.sh scripts/profile_sm89_ops.py --trace-only \
    outputs/profile_sm89/trace_nographs_b${B}.json --top-n 25 \
    --json-out outputs/profile_sm89/trace_report_b${B}.json > /dev/null
done
```

### 第三层（本环境会失败，见 §5）

```bash
bash outputs/profile_sm89/run_ncu.sh '<kernel regex alternation>'
```

### 功耗、显存、barrier A/B、正确性门槛

```bash
bash outputs/profile_sm89/run_remaining.sh     # barrier A/B + 显存
bash outputs/profile_sm89/run_gate.sh          # 正确性门槛

# 模块功耗（cfm/vocoder 分层必须用 graph-free 计划）
bash scripts/run.sh scripts/profile_sm89_power.py --gpu 6 --batch 8 \
  --modules idle,ar_round,cfm,vocoder --seconds 2.0 \
  --deployment outputs/profile_sm89/deployment_nographs_diagnostic.json \
  --ref-audio outputs/profile_sm89/reference.wav \
  --json-out outputs/profile_sm89/power_modules_b8.json

# 生产路径的整请求能耗
bash scripts/run.sh scripts/profile_sm89_power.py --gpu 6 --batch 8 \
  --modules idle,ar_round,first_chunk --seconds 2.0 \
  --deployment configs/sm89_bf16_triton.json \
  --ref-audio outputs/profile_sm89/reference.wav \
  --json-out outputs/profile_sm89/power_firstchunk_b8.json
```

### 本次会话新增/修改的文件

新增（均为测量工具，不改生产行为）：

| 文件 | 用途 |
| --- | --- |
| `scripts/profile_sm89_ops.py` | 第二层算子和 module 归因，含 trace 后处理 |
| `scripts/report_sm89_layer1.py` | 第一层报告与跨档位伸缩表 |
| `scripts/report_sm89_nsys_gaps.py` | 逐 range 的 GPU 空闲分析 |
| `scripts/profile_sm89_power.py` | 模块隔离功耗/能耗 |
| `scripts/profile_sm89_memory.py` | 显存峰值与 graph pool 占用 |
| `outputs/profile_sm89/*.sh` | 驱动脚本，即本节的原始命令 |

修改：

| 文件 | 改动 | 是否影响生产行为 |
| --- | --- | --- |
| `scripts/benchmark_sm89_batch.py` | 增加 `--json-out` / `--record-requests`，补 median/p90/p95 与每请求 rounds/accepted 明细 | 否（测量脚本） |
| `src/acc_infer_clear/runtime/pool.py` | `result` 回包追加 3 个只读键（`rounds`/`accepted`/`kv_head_lengths`） | 否。`result` 只在请求完成后由 `Pool.result()` 调用，`run_ready` 路径从不调用它；`run_gate.sh` 用 PCM 字节一致性作了实证 |

**生产路径本身未作任何修改。**
