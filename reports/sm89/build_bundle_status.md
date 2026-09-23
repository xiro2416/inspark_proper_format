# SM89 TensorRT 11.3 新布局实测（2026-09-23）

## 缓存优先入口的追加验收（同日）

新增 `trt ensure` 后，在物理 GPU4 用 `--mode reuse-only` 命中原有 B1/B4/B8 固定 ID，未重建。随后在隔离的 `.work/trt113_ensure_b1_acceptance/` 目录用 `--mode build-only --allow-experimental` 重新构建 B1 四组件；新 schema 2 bundle ID 为 `08e546dba305d3a8877c`，源码指纹 `336b431b3b15cc8404e9947e447b6107a28c3477e07ad19cd075015c82b71dd1`，builder 参数为 Target level 3、其余 level 5、tiling NONE、WORKSPACE 8 GiB、强类型、TF32 关闭。四组件 plan/哈希/源码身份校验及真实首 chunk 路由通过，五次测量全部首 PCM 中位 50.58 ms；再次执行 `reuse-only` 命中该新 bundle。它未上传 HF，`numerical_pass=false`、`certified_for_production=false`，不能把此次路由通过视为新增浮点或质量认证。过程中一次构建因其他工作并行改动源码而按设计拒绝混合指纹，一次因 GPU4 协作锁被占用而在 Target 前停止；失败 staging 均保留，未发布为完整 bundle。其他 SM 和 24GB 同 SM 卡仍无实际构建验收。

本报告只对应物理 GPU4（NVIDIA GeForce RTX 4090，SM89，49140 MiB）上新构建的 `first_chunk_p258_f52_k128`。engine 构建基于提交 `191c821` 的推理代码，清单源码哈希为 `8d8b1a74b75d08095489b06ea15f0a01270d73f3428e67396f1858056d9098d7`；其后仅更新缓存发布工具、测试和报告，未重建 engine。构建和 GPU 测试均在同一张 GPU 上串行运行。其他 SM、其他 GPU 型号、batch 或 shape 未通过本次验证；以下结果不是生产认证。

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

完整 EOS 的 **256 对**质量评估已完成 B1/B4/B8，分别与同一 BF16 eager 基线（UTMOS 1.664038、CER 147/4093=3.5915%）比较：

| batch | TRT UTMOS / 相对变化 | TRT CER / 较 eager 变化 | 预设质量门禁 |
| --- | ---: | ---: | --- |
| B1 | 1.638357 / −1.5433% | 143/4093=3.4938% / −0.0977 百分点 | 通过 |
| B4 | 1.651413 / −0.7587% | 152/4093=3.7137% / +0.1222 百分点 | 通过 |
| B8 | 1.637097 / −1.6191% | 175/4093=4.2756% / +0.6841 百分点 | 通过 |

三组均满足预设 UTMOS 相对下降 ≤3%、CER 增加 ≤2 个百分点和 256 对覆盖门禁，评估进程没有初始化 CUDA。原始报告在本地 `artifacts/quality/sm89_gpu4_final/paired_eval_b{1,4,8}_256.json`。先前 B8 的 32 对诊断 UTMOS 下降 3.1832%，未通过门禁；完整样本结论不同，故保留该失败诊断但以预先指定的 256 对作为质量门禁结论。质量通过不覆盖前述逐算子数值失败。

B8 TRT 在同一张 GPU4 上依次完成单请求、32、64 并发的完整 EOS 持续运行，三个档位均通过 600 秒合格门禁。完整 EOS 数分别为 520、1310、1339；最长 KV 分别为 520、582、568。请求 RNG 隔离、取消/重用、空输入错误恢复、长序列和内存趋势检查均通过。64 并发末次波次排空后总用时 621 秒。完整数据在本地 `artifacts/benchmarks/sm89_gpu4_final/soak_b8_1_32_64_600s.json`。该测试验证稳定性与执行覆盖，不宣称完整 EOS 数值与 eager 一致；旧 GPU6 构建受占用限制的记录已被本次 GPU4 实际构建取代，不再是当前状态。

## 私有 HF 制品

在权利人确认许可门槛并授权私有存储后，三个完整 bundle 均上传至 [`xirr/inspark_proper_format_trt113_sm89`](https://huggingface.co/xirr/inspark_proper_format_trt113_sm89)，仓库为私有。上传包括四组件 engine/ONNX/plan、哈希清单、分发声明、IndexTTS2 许可证、BigVGAN 许可证和派生声明；不把二进制提交 Git。

| bundle | 不可变 Hub revision |
| --- | --- |
| B1 `b62db855b4239970035f` | `fc17c1a37195cbc237f27e68b794f7452b1a9c48` |
| B4 `18c46ef53cada517d52d` | `504ad7ae3d83c8b1f04bb6b245060173ddac78fd` |
| B8 `da930a3e09316f984a28` | `7441a6dbba261c625886f02208ddb06368cf68c0` |

私有仓库经 hf-mirror 的元数据请求在本环境失败，因此拉取器先试镜像、仅在该类失败时回退官方端点。B1/B4/B8 的上述固定 revision 均已分别在新目录完成完整拉取、逐文件 SHA256 校验和 GPU4 四组件首 chunk 路由重验，均返回 `fetched_route_passed_numerically_experimental`，实际下载端点为 `https://huggingface.co`。Xet 大文件传输在本环境曾停滞，改用 `HF_HUB_DISABLE_XET=1` 的普通 HTTP 后成功；B4 普通 HTTP 首次也曾遇到一次 Hub 元数据瞬时错误，重试后复用已校验缓存并完成验收。这不是其他服务器或其他 SM 的可用性证明。
