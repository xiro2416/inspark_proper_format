# TensorRT 11.3 Draft 重新裁决

日期：2026-09-22  
设备：GPU6，RTX 4090 / SM89

## 裁决

**2026-09-22 更正：此前“TRT Draft 导致 acceptance 系统性下降”的裁决无效，根因是 KV cache 镜像实现 bug，不是 BF16 tactic 的正常数值差异。**

变量长度的初始 Draft context（例如 41 token）未命中已捕获的 context bucket 时会走 fallback。该 fallback 只更新 2048-capacity 主 KV pool，没有更新 TensorRT 使用的 128-capacity compact mirror。于是 TRT Draft 首轮读取的历史全为零，后续也永久缺少初始 context，造成稳定的 acceptance 坍塌。`BatchedContextAppend` 已修复为在 fallback 完成后同步对应 slot。

修复后的真实状态逐步对拍：KV `max_abs=0`；Draft logits cosine 约 0.99997--0.99999。32 条全新文本的 B1/B4 acceptance 与当前 Draft 恢复一致，因此 full TRT Draft 重新进入候选；B8/B16 尚需按修复版重测，不能沿用旧裁决。

| Batch | 路径 | P50 (ms) | 平均吞吐 (req/s) | 平均 rounds | accepted_sum |
|---:|:---|---:|---:|---:|---:|
| 1 | 当前 Draft | 42.535 | 24.026 | 7.125 | 25.563 |
| 1 | TRT（bug） | 59.773 | 16.979 | 13.688 | 17.406 |
| 1 | TRT（修复） | 42.414 | 23.426 | 7.063 | 25.531 |
| 4 | 当前 Draft | 76.453 | 52.399 | 7.813 | 24.906 |
| 4 | TRT（bug） | 94.266 | 42.184 | 14.000 | 17.188 |
| 4 | TRT（修复） | 73.970 | 53.105 | 7.688 | 24.375 |

修复版本表没有稳态功耗窗口，只用于 correctness/trajectory 复核；功耗不得引用旧 bug 版本。

## 旧的错误实现结果（仅保留用于回归，不得用于选型）

共同路径为 TensorRT Target；实验组只增加 TensorRT Draft。文本、随机情绪/seed、调度、CFM、Vocoder 和精度策略一致，每档 160 请求，功耗窗口 15 秒。

| Batch | Draft | P50 (ms) | P90 (ms) | 稳态吞吐 (req/s) | 平均功耗 (W) | 平均 AR rounds | accepted_sum |
|---:|:---|---:|---:|---:|---:|---:|---:|
| 1 | 当前 | 41.794 | 47.530 | 23.077 | 207.36 | 7.406 | 25.069 |
| 1 | TRT full | 59.732 | 71.719 | 16.385 | 205.09 | 13.738 | 17.044 |
| 4 | 当前 | 73.531 | 81.040 | 52.029 | 231.07 | 7.794 | 24.306 |
| 4 | TRT full | 97.221 | 105.912 | 40.830 | 237.53 | 14.538 | 16.313 |
| 8 | 当前 | 113.874 | 124.435 | 67.031 | 253.27 | 7.838 | 24.369 |
| 8 | TRT full | 135.860 | 146.476 | 57.864 | 259.58 | 14.338 | 16.838 |

相对当前 Draft：

| Batch | P50 | P90 | 稳态吞吐 | 平均功耗 | AR rounds | accepted_sum |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | +42.92% | +50.89% | -29.00% | -1.09% | +85.49% | -32.01% |
| 4 | +32.22% | +30.69% | -21.53% | +2.79% | +86.53% | -32.89% |
| 8 | +19.31% | +17.71% | -13.68% | +2.49% | +82.94% | -30.90% |

## 旧定位结果及其局限

逐层 B1 对齐表明：

- input RMSNorm 几乎一致；
- Q/K/V 输出逐元素一致；
- attention 输出 cosine 为 0.99999994，mask/KV/attention 语义不是主因；
- 第一处显著偏差出现在 attention `o_proj` 的 TensorRT BF16 GEMM；
- 将同一个 TRT attention 输出交给当前 cuBLAS GEMM，输出 cosine 为 1.0；交给 TRT tactic 后 cosine 为 0.999995，随后被三层 residual/MLP 放大；
- 完整 hidden/base cosine 最终降至 0.997891/0.997405。

原 builder 还存在一个真实合同错误：Draft `lm_head` 在当前路径是 FP32 计算，但 builder 将其强制为 BF16。该错误已经修正；不过 hidden 在 head 之前已经偏离，所以修正 head 本身不足以恢复 acceptance。

先前只修复了把“提交 token bucket”误当成“请求 batch”的条件错误，却漏掉了变量长度 fallback 完全不写 compact mirror 的第二个缓存错误。因此当时“B4 轨迹不变，所以主因是 GEMM 数值路径”的推断错误。真实状态 cache 对拍最终确认第二个错误才是系统性退化主因。

## 稳定累加候选

实验候选显式保留 BF16 输入/权重/输出舍入边界，但对敏感 `o_proj/MLP` 强制 FP32 accumulation：

- hidden/base cosine 提升至 0.999539/0.999530；
- Draft 模块从 0.4314 ms 变为 0.4878 ms，慢 13.1%；
- B1 160 请求 P50 仍为 61.838 ms、AR rounds 14.138、accepted_sum 16.744，未恢复端到端轨迹。

该候选在错误 KV cache 上测得，结果作废；修复后的普通 TRT engine 已无需用它恢复 acceptance。

## 后续边界

完整 Draft backbone 可以继续作为候选。后续要求：

1. 重测修复版 B1/B4/B8/B16 的 160 请求延迟、吞吐、功耗、rounds 和 acceptance；
2. 每次新增 compact cache 路径时，覆盖捕获 bucket 与变量长度 fallback 两种同步合同；
3. 验收必须看真实 cache 状态与端到端统计，不能只看零 cache 的层级 cosine 或 backbone latency。

原始数据：

- `outputs/trt113_draft_160/trt_target_draft_b{1,4,8}_power15s.json`
- `outputs/trt113_draft_160_fixed/trt_target_draft_stable_b1_power15s.json`
- `outputs/trt113_draft_layer_diagnosis_b1_detailed.json`
- `outputs/trt113_draft_layer_diagnosis_b1_stable.json`
- `outputs/trt113_draft_stable_b1_module.json`
- `outputs/trt113_draft_realstate_diag_b1_fixed.json`
- `outputs/trt113_draft_new32/trt_draft_fixed_b{1,4}.json`
