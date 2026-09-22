# SM89 TensorRT 11.3 B8 全链路结果

日期：2026-09-22  
GPU：NVIDIA GeForce RTX 4090（SM89，48 GiB）  
精度：保持原方案，Target/Draft 为 BF16；CFM 的外部接口和 Solver 输出为 FP32；未启用 FP8 或 INT8。

## 已实现

- Target：TensorRT 11.3 原生静态 B8 full engine，保留 device-side speculative AR 调度和 CUDA Graph。
- Draft：TensorRT 11.3 原生静态 B8 full backbone，保留原 proposal RNN、采样、验收和 cache 调度。
- CFM：将完整两步 Solver（两次 estimator、Euler 更新和 mask）导出为一个静态 B8/F310 ONNX，并由 TensorRT 11.3 构建 strongly-typed engine。
- Vocoder：继续使用当前已验证的 BF16/cuDNN + alias 路径，没有用较慢的旧 TensorRT vocoder 替换。
- 三个 TensorRT 模块现在使用同一套 TensorRT 11.3 runtime，避免 10.12/11.3 同进程混载。

部署配置：`configs/sm89_bf16_trt113_all_b8.json`  
CFM plan：`artifacts/trt113_cfm/plan_b8.json`  
CFM engine SHA256：`b31895817517bac840f23c8baf17ffd17b34aecffe7273d886e5c23e1f06df88`

## CFM 独立验证

固定输入 B8/F310/P258，50 次计时：

| 路径 | 延迟 |
|---|---:|
| 未融合 BF16 eager Solver | 37.876 ms |
| TensorRT 11.3 native | 12.182 ms |
| TensorRT 11.3 + CUDA Graph | 12.039 ms |

对未融合 eager 的加速为 3.109x。相对原生产路径已经启用 RMS/AdaLN、RoPE Triton 融合的 CFM，实测 stage 从 16.692 ms 降至 11.787 ms，即减少 4.905 ms（-29.38%）。

数值误差：max abs 0.021838、mean abs 0.000579、cosine 0.999999762。CUDA Graph 与 native engine 输出一致；独立测试 fallback 为 0。

原始结果：

- `outputs/trt113_cfm_b8_validation.json`
- `outputs/profile_sm89/trt113_td_b8_stage2.json`
- `outputs/profile_sm89/trt113_all_b8_stage2.json`

## B8、160 请求端到端测量

每组为 32 条文本重复 5 次、随机情绪、20 个 B8 group；另做至少 15 秒连续功耗窗口。基线已经启用 TRT11.3 Target + Draft，候选仅再替换 CFM。

| 指标 | TRT Target+Draft + 当前 CFM | TRT Target+Draft+CFM | 变化 |
|---|---:|---:|---:|
| 全部首 chunk P50 | 114.691 ms | 116.485 ms | +1.564% |
| 全部首 chunk mean | 116.742 ms | 118.227 ms | +1.272% |
| 持续吞吐 | 63.101 req/s | 66.383 req/s | +5.201% |
| 持续平均板卡功耗 | 245.282 W | 248.518 W | +3.236 W / +1.319% |
| 持续峰值板卡功耗 | 279.850 W | 278.830 W | -1.020 W |
| 单请求板卡能耗 | 3.887 J | 3.744 J | -3.690% |
| 峰值板卡显存（含外部占用） | 43017 MiB | 43421 MiB | +404 MiB |

原始结果：

- `outputs/trt113_td_b8_true160_power15s.json`
- `outputs/trt113_all_b8_true160_power15s.json`

这轮端到端结果支持“吞吐和有效 GPU 工作提高”，但不支持宣称 P50 延迟改善：P50 有约 1.8 ms 回退。两次独立进程运行的离散 AR 采样轨迹不完全相同，因而这张端到端表不是逐轨迹配对测量；CFM 在因果上位于 AR 之后，不会改变此前的 token 轨迹。CFM 本身的配对 stage 测量仍稳定减少约 4.9 ms。后续延迟裁决应固定并回放同一组 speech codes/CFM 输入。

## 路由与 fallback 核验

B8 smoke test 中：Target/Draft device-round 成功 2/2、device fallback 0；CFM backend 为 `TensorRT 11.3 native`，测量窗口新增 CFM fallback 0。

部署时累计的 28 次 CFM fallback 是捕获 B1-B7 通用 head graphs 时产生的预期行为；B8 静态 engine 不接受这些 shape。统计现已拆分为部署累计值和测量窗口 delta，避免把图准备回退误报为 B8 推理回退。

原始结果：`outputs/trt113_all_b8_smoke_stats.json`

## 当前裁决

- B8：保留独立全 TRT 实验配置。CFM 模块级收益和持续吞吐/功耗收益成立。
- 尚不直接覆盖默认生产配置：先完成固定 speech-code 输入的端到端配对质量与延迟门禁，确认音频差异符合预期。
- B1/B4/B16：本次新 CFM engine 没有这些 shape，不做外推；应分别构建、timing 和裁决，不能让 B8 engine 动态回退冒充优化。
