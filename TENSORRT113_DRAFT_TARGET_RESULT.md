# SM89 Draft/Target TensorRT 11.3 优化结果

日期：2026-09-22  
设备：GPU6，RTX 4090，SM89，48 GiB，Driver 595.71.05  
源码：`7802054b764895895716a21068b9d21802edf13e` 加本工作区实验改动

## 结论

最终采用按固定 batch 分流的方案，而不是在一个 max-batch 进程中常驻全部 engine：

| Batch | 最终 Target | 最终 Draft |
| ---: | --- | --- |
| 1 | TensorRT 11.3 完整 24 层 Target engine | 当前 Slot Draft + CUDA Graph |
| 4 | TensorRT 11.3 完整 24 层 Target engine | 当前 Slot Draft + CUDA Graph |
| 8 | 当前 Slot Target + CUDA Graph | 当前 Slot Draft + CUDA Graph |
| 16 | 当前 Slot Target + CUDA Graph | 当前 Slot Draft + CUDA Graph |

对应部署配置：

- B1：`configs/sm89_bf16_trt113_target_b1.json`
- B4：`configs/sm89_bf16_trt113_target_b4.json`
- B8/B16：`configs/sm89_bf16_current_fair.json`

不建议在 max_batch=16 的单进程里同时加载 B1/B4 engine。两个 engine 常驻会使显存从约 29.8 GiB 增至 32.7 GiB，并拖慢原本应回退当前路径的 B8/B16。

## 精度与调度合同

- 没有 FP8、INT8 或 W8A8；Linear/KV 仍是现行 BF16 合同。
- Target 的输入、LayerNorm、residual、输出及 LM head 保持 FP32。
- speculative acceptance、Target 验证顺序、device-side control、CFM/Vocoder、随机数策略不变。
- TensorRT engine 固定 batch、固定 Q=8、首段 K=128；长上下文和非紧凑/交错 slot 自动回退当前实现。
- TensorRT 11.3 使用原生 `IAttentionV2` 和 `IKVCacheUpdate`。Builder optimization level 3；8 GiB workspace 只是离线 tactic 搜索上限，不是运行时常驻显存。

## Target 模块结果

完整 24 层 Target（含 QKV、attention、MLP、KV update、LM head）的 CUDA-event 测量：

| Batch | 当前 CUDA Graph | TRT 11.3 | 延迟变化 | speedup | logits cosine |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.183 ms | 1.659 ms | -24.00% | 1.316x | 0.999920 |
| 4 | 2.386 ms | 1.662 ms | -30.35% | 1.436x | 0.999908 |
| 8 | 2.653 ms | 1.894 ms | -28.60% | 1.401x | 0.999914 |
| 16 | 3.149 ms | 2.415 ms | -23.32% | 1.304x | 0.999924 |

模块快不等于端到端应该启用。B8/B16 engine 在真实 32 文本路径分别导致吞吐约 -7.71% 和 -1.77%，所以最终拒绝。

持续调用的模块功耗测试也确认 B1/B4 engine 真正执行：Target verify 从约 459/419 calls/s 提高到 603/602 calls/s。按空闲功耗扣除后的单次能量，B1 从 0.297 J 降至 0.183 J，B4 从 0.383 J 降至 0.221 J。

## 公平端到端结果

共同条件：32 条固定文本、固定随机情绪与 seed、`mingxiang_gao.wav`、2 次 warmup、每档独立固定-batch 进程、5 秒持续功耗窗口。

| Batch | 方案 | 全部首 chunk 中位数 | 测量吞吐均值 | 持续吞吐 | 持续功耗 | 单请求能耗 | 峰值显存 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 当前 | 44.832 ms | 21.900 req/s | 21.562 req/s | 194.79 W | 9.034 J | 8256 MiB |
| 1 | TRT Target | 43.582 ms | 23.279 req/s | 22.815 req/s | 193.37 W | 8.476 J | 9234 MiB |
| 1 | 变化 | **-2.79%** | **+6.29%** | **+5.81%** | **-0.73%** | **-6.18%** | +978 MiB |
| 4 | 当前 | 83.810 ms | 49.392 req/s | 47.620 req/s | 214.62 W | 4.507 J | 12568 MiB |
| 4 | TRT Target | 70.589 ms | 56.237 req/s | 53.142 req/s | 237.90 W | 4.477 J | 13668 MiB |
| 4 | 变化 | **-15.78%** | **+13.86%** | **+11.60%** | +10.84% | **-0.68%** | +1100 MiB |

B4 是明显的延迟/吞吐优化，但以更高平均瓦数换取；能耗仍小幅下降。B1 收益较温和，但功耗和能耗也没有恶化。

原始结果：

- `outputs/current_b1_exact_32_power5s.json`
- `outputs/trt113_target_b1_final_32_power5s.json`
- `outputs/current_b4_exact_32_power5s.json`
- `outputs/trt113_target_b4_final_32_power5s.json`
- `outputs/profile_sm89/current_modules_b{1,4}_power.json`
- `outputs/profile_sm89/trt113_target_b{1,4}_power.json`

## Draft 探索及拒绝原因

另行编译了完整 3 层 Draft backbone：RMSNorm、三个 eager-origin Q/K/V projection、FP32 attention/KV、BF16 MLP/Linear、输出 head。Proposal 的 7-step RNN 保持原实现。

| Batch | 当前 Draft backbone | TRT Draft | 模块延迟变化 | base-logits cosine |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.434 ms | 0.312 ms | -28.14% | 0.997399 |
| 4 | 0.445 ms | 0.359 ms | -19.15% | 0.998681 |
| 8 | 0.475 ms | 0.415 ms | -12.58% | 0.997993 |
| 16 | 0.580 ms | 0.543 ms | -6.31% | 0.997931 |

尽管单模块更快，B1 将它与 Target engine 一起接入后，32 文本首 chunk 中位数恶化到 58.556 ms，持续吞吐只有 16.394 req/s。Draft logits 的小数值偏差改变 proposal/acceptance 轨迹，模块收益被更多 speculative rounds 抵消。因此 Draft TensorRT engine 全部保留为实验产物，不进入部署配置。

这也说明 Draft 的优化目标不能只看 backbone latency：必须同时约束 acceptance rate、AR rounds 和音频轨迹。下一轮若继续，应优先做 bitwise-safe 的 layout/graph-boundary 融合，或把 Draft+Proposal+acceptance 作为一个 execution family 联合优化，而不是只替换 Draft backbone。

## 正确性与质量状态

- Target 模块四档 logits cosine 均不低于 0.99990，32/32 随机情绪音频生成完成，无运行失败。
- 与当前 B1 顺序生成相比，两边 chunk 总数同为 87；音频长度比的中位数 0.986、均值 0.998。
- BF16 tactic/attention 舍入会改变采样轨迹，PCM 不应期待逐样本相同；32 例中只有 1 例长度完全相同，因此波形 SNR 不是有效质量指标。
- 当前只通过数值门和生成完整性门，尚未完成 ASR/说话人相似度/主观听测。若作为默认线上配置，仍需完成感知质量门。

## 实现和已知限制

- 独立环境：`.venv-trt113`，TensorRT 11.3.0.99；原 `.venv` 与 TensorRT 10.12 未被覆盖。
- TRT level 4/5 在带 `IKVCacheUpdate` 的 Target 图上触发 output-aliasing builder 问题；Target 使用可稳定构建的 level 3。Draft 无 KV update，使用 level 5，但若干 Myelin tactic 报内部错误并被 builder 自动跳过。
- `key_value_lengths` 在本机 SM89 构建成功但返回全零 attention，因此使用显式 bool mask。
- free-list 已改为稳定的低 slot 优先分配，保证完整固定 batch 在释放/重用后仍可命中紧凑 engine；交错请求仍安全回退。
- B32 不在本轮候选空间内。

