# InSpark inference

IndexTTS2 推理加速仓库。源码统一位于 `src/inspark_infer`；模型通过 `ops` 选择 PyTorch eager、离线 `torch.compile`、Triton/CUDA 或 TensorRT 实现，不复制整套模型。当前没有 ZipVoice 实现。架构与兼容边界见 [架构说明](docs/architecture.md)。

## 当前支持与状态

| 硬件与路径 | 支持范围 | 审计状态 |
| --- | --- | --- |
| SM89 / FP32 eager | IndexTTS2 完整 EOS；正确性参考 | 默认路径 |
| SM89 / BF16 eager、离线 compile | 已有实验配置 | 按各组件审计结果使用；不能等同 FP32 |
| SM89 / TRT 11.3 | 固定首 chunk 的独立 B1/B4/B8 Target、Draft、CFM、Vocoder engine；其余步骤仍用现有推理流程 | **数值对 eager 的严格门禁未通过，实验性** |
| SM120 | 历史 Triton/CUDA 配置与报告 | [历史证据](reports/sm120/README.md)，迁移后未重新认证 |
| 其他 SM | 暂无经实测的一键 TRT 构建 | 不宣称支持 |

B1/B4 的请求隔离 TRT 路径、B8 的历史和并发结果不是同一测量口径。SM89 的既有结果见 [历史审计](reports/sm89/stage2/README.md)；新布局已在物理 GPU4 构建精确 B1/B4/B8 四组件，并完成真实首 chunk 路由和 eager 对照，结论见 [本次报告](reports/sm89/build_bundle_status.md)。数值门禁未通过，不能用路由或单算子加速覆盖；B8 的单请求、32 和 64 并发各 600 秒完整 EOS 持续运行已通过。

GPU4 上短文本、同口径 BF16 eager 对 TRT 11.3 的首 PCM 中位延迟（预热 2 次、测量 5 次，包含打包/拷贝/启动，不含模型准备）：B1 246.37→54.38 ms，B4 426.57→91.52 ms，B8 531.73→162.81 ms。B1/B4/B8 完整 EOS 的各 256 对质量门禁均通过，但严格浮点门禁未通过；TRT B8 功耗剖析触发限速标记，不能宣称省电或生产认证。

## 安装与运行

Linux、Python 3.11、CUDA 和一张可用 GPU。权重、ONNX、engine、缓存和环境均保存在仓库根目录的本地忽略目录，不进入 Git。

```bash
bash scripts/bootstrap.sh
bash scripts/download_models.sh
CUDA_VISIBLE_DEVICES='' ACC_TRITON_TOOLCHAIN=default bash scripts/run.sh -m pytest -q tests

# FP32 eager 参考；--gpu 为物理 GPU 编号。
ACC_TRITON_TOOLCHAIN=default bash scripts/run.sh \
  benchmarks/benchmark_reference.py --gpu 4 --batch 1 \
  --deployment configs/hardware/sm89/sm89_eager_fp32.json \
  --json-out outputs/eager_fp32_b1.json
```

固定首 chunk B1/B4/B8 的一键入口为 `bash scripts/build_trt.sh --gpu <物理编号> --model indextts2 --profile first_chunk_p258_f52_k128 --batches 1,4,8 --ref-audio /workspace/reference.wav --allow-experimental`。它先匹配本地完整 bundle，再匹配固定私有 HF revision，最后才在目标 SM 上串行构建；`--mode reuse-only|build-only` 可强制策略。当前只有 48GB RTX 4090 SM89 经构建、首 chunk 路由和私有 HF 新目录拉取验收，其他 SM 是待实测的构建候选；所有现有 engine 均未通过严格浮点门禁，不能生产认证。builder 和量化分别配置，当前 TRT 仅支持未量化路径。用法、身份约束及 Codex 扩展步骤见 [构建说明](docs/trt-build-for-codex.md)。

精度参考路径禁用 TF32；项目 kernel 与 TRT 均为可选优化。构建、数值、质量和性能分别判定；硬件、shape、精度或 plan 身份不匹配时明确回退或报错。测试与性能入口分别在 `tests/`、`benchmarks/`；通用审计代码在 `src/inspark_infer/guardrails/`。第三方来源见 [许可证与声明](THIRD_PARTY_NOTICES.md)。
