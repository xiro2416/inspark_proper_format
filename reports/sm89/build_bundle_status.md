# SM89 TensorRT 11.3 新布局实测（2026-09-23）

本报告只对应物理 GPU4（NVIDIA GeForce RTX 4090，SM89，49140 MiB）上新构建的 `first_chunk_p258_f52_k128`。构建和测试均在同一张 GPU 上串行运行。其他 SM、其他 GPU 型号、batch 或 shape 未通过本次验证；以下结果不是生产认证。

## 独立构建与首 chunk 路由

一条命令构建了精确 B1/B4/B8 各自的 Target、Draft、CFM、Vocoder TensorRT 11.3.0.99 engine：

```bash
bash scripts/build_trt.sh --gpu 4 --model indextts2 \
  --profile first_chunk_p258_f52_k128 --batches 1,4,8 \
  --ref-audio /workspace/index-tts/data/audio/old/mingxiang_gao.wav
```

| batch | bundle ID | 首 chunk Target / Draft 原生 TRT | CFM / Vocoder | 路由门禁内五次测量的全部首 PCM 中位延迟 |
| --- | --- | --- | --- | --- |
| B1 | `b62db855b4239970035f` | 每次 9/9、9/9 | 每次原生 TRT、零回退 | 56.72 ms |
| B4 | `18c46ef53cada517d52d` | 每次 10/10、10/10 | 每次原生 TRT、零回退 | 89.99 ms |
| B8 | `da930a3e09316f984a28` | 每次 13/13、13/13 | 每次原生 TRT、零回退 | 172.58 ms |

`artifacts/trt113_bundles/sm89/first_chunk_p258_f52_k128/bN/<bundle-id>/` 是本地忽略目录，包含逐组件 engine、plan、ONNX、哈希清单、部署配置及 `route_report.json`。上表时间包括请求接纳、打包、内存操作和启动，不含模型/参考音频准备；预热 2 次、测量 5 次，不能单独证明相对 eager 的优势或功耗表现。每个 bundle 的 `numerical_pass=false`、`certified_for_production=false`。

B1 的 Draft 首 chunk 实际出现 **KV64 六次、KV128 三次，没有超过 KV128**。先前 6 次 eager 回退是 KV64 未被路由到可兼容的 KV128 engine；修复后两种 KV 长度均原生 TRT。B4/B8 在 EOS 提前到达时活动请求数会缩小，现由请求隔离的固定 batch 打包路由到对应 engine；这不等于动态 batch engine。

## 同口径性能与功耗

`benchmarks/benchmark_reference.py` 用相同短文本、参考音频、运行配置和 GPU4，分别测 BF16 eager 与上述 TRT bundle 的真实首 PCM。各项预热 2 次、测量 5 次，取每次 batch 内最后一个请求的首 PCM 延迟中位数；计入接纳、打包、拷贝和启动，不含模型/参考音频准备。原始 JSON 在本地 `artifacts/benchmarks/sm89_gpu4_final/`。

| batch | BF16 eager | TRT 11.3 | 速度比 |
| --- | ---: | ---: | ---: |
| B1 | 246.37 ms | 54.38 ms | 4.53× |
| B4 | 426.57 ms | 91.52 ms | 4.66× |
| B8 | 531.73 ms | 162.81 ms | 3.27× |

`torch.compile` BF16 B1 基线也用同入口实测：首 PCM 中位数 318.21 ms，慢于 eager 的 246.37 ms；首次预热含约 23.93 s 编译/审计成本。该次执行成功，但算子数值门禁未通过，不能把它当作性能已优化或正确性已验证的基线。

`benchmarks/profile_sm89_power.py` 用持续 8 s 首 chunk 循环测板卡功率：TRT B1/B4 的平均值分别为 181.95/196.50 W，空闲基线为 68.38/68.28 W，两次均未报告限速；B8 观测平均值 211.52 W、空闲 71.42 W，但驱动报告限速标记，剖析器判 `valid=false`，不能给 B8 稳态功耗认证。相同脚本测得 BF16 eager B1/B4/B8 平均 94.97/113.00/159.36 W，各自空闲基线 89.26/84.10/90.02 W。TRT 延迟降低但板卡瞬时功率更高，且空闲基线、温度和时钟不匹配；这些短窗样本不能支持更省电或准确的能耗比结论。

## 与 eager 的审计边界

“相同权重”在此仅指捕获时的加载器哈希与构建清单记录一致；engine 常量尚未独立提取验证，清单 `provenance_status=recorded_not_audited`，不能宣称已完成同权重数值认证。

在相同权重、输入记录下，分别对 B1/B4/B8 捕获真实 AR 区域并比较 FP32 与 BF16 eager，随后比较 CFM/Vocoder 声学区域。六组 AR 和六组声学审计均完成，但严格浮点门禁均未通过。审计明确区分计算区域/状态与数值：B1 的原生 graph 对 direct 输出逐位一致，Target/Draft cache 未写区域与审计后恢复检查通过，实际组件覆盖也通过；这些边界检查**不能**替代输出浮点门禁。例：B1 第一 Draft hidden 的 mismatch 为 0，首 Target logits 对 FP32/BF16 分别有 1732/1635 个 mismatch；B1 首 Vocoder 对 FP32/BF16 分别有 2501/1226 个 mismatch。B4/B8 也有非零差异。审计 JSON 分别保存在本地 `artifacts/audits/sm89_gpu4_final/{b1_ar,b4_ar_full2,b8_ar_full2}/`；成功捕获的 B4/B8 比较用了满 batch 的前 2 次 AR 调用，尚不能当作活跃行缩小时的逐调用数值证明。数值失败不能被首 chunk 路由通过所覆盖。

| batch | 第一 Draft base mismatch（FP32 / BF16 eager） | 第一 Target logits mismatch（FP32 / BF16 eager） | CFM mismatch（FP32 / BF16） | Vocoder mismatch（FP32 / BF16） |
| --- | ---: | ---: | ---: | ---: |
| B1 | 302 / 1394 | 1732 / 1635 | 0 / 0 | 2501 / 1226 |
| B4 | 32756 / 42878 | 13624 / 13194 | 1 / 2 | 5118 / 740 |
| B8 | 28516 / 54718 | 18498 / 23700 | 17 / 18 | 2847 / 825 |

这些是按审计器既定容差统计的元素不通过数量，不代表误差幅度、感知质量或相同形状间可直接比较的比例；声学数值取第一 head 捕获。FP32/BF16 是两条分别重算的 eager 参考，不是 TRT 精度模式切换。

完整 EOS 的 **256 对**质量评估已完成。B8 TRT 对 BF16 eager 的 UTMOS 均值为 1.637097 对 1.664038，相对下降 1.6191%（门禁 ≤3%）；CER 为 175/4093=4.2756% 对 147/4093=3.5915%，增加 0.6841 个百分点（门禁 ≤2 个百分点）。两项和 256 对覆盖门禁均通过，评估进程没有初始化 CUDA。原始报告在本地 `artifacts/quality/sm89_gpu4_final/paired_eval_b8_256.json`。先前 32 对诊断的 UTMOS 下降 3.1832%，未通过门禁；完整样本结论不同，故保留该失败诊断但以预先指定的 256 对作为质量门禁结论。质量通过不覆盖前述数值失败，也不证明 B1/B4 的质量。

当前仍需 32/64 并发与长序列持续运行，以及 HF 私有缓存发布/新目录拉取验收。以上计时是首 chunk，不是完整 EOS 吞吐或能耗；旧 GPU6 构建受占用限制的记录已被本次 GPU4 实际构建取代，不再是当前状态。
