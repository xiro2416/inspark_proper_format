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

B1/B4 的请求隔离 TRT 路径、B8 的历史和并发结果不是同一测量口径。SM89 的既有结果见 [历史审计](reports/sm89/stage2/README.md)；新布局已在物理 GPU4 构建精确 B1/B4/B8 四组件，并完成真实首 chunk 路由和部分 eager 对照，结论见 [本次报告](reports/sm89/build_bundle_status.md)。数值门禁未通过，不能用路由或单算子加速覆盖；32/64 并发仍待本次独立验证。

## 安装与运行

Linux、Python 3.11、CUDA 和一张可用 GPU。权重、ONNX、engine、缓存和环境均保存在仓库根目录的本地忽略目录，不进入 Git。

```bash
bash scripts/bootstrap.sh
bash scripts/download_models.sh
CUDA_VISIBLE_DEVICES='' ACC_TRITON_TOOLCHAIN=default bash scripts/run.sh -m pytest -q tests

# FP32 eager 参考；--gpu 为物理 GPU 编号。
ACC_TRITON_TOOLCHAIN=default bash scripts/run.sh \
  benchmarks/benchmark_reference.py --gpu 6 --batch 1 \
  --deployment configs/hardware/sm89/sm89_eager_fp32.json \
  --json-out outputs/eager_fp32_b1.json
```

SM89 精确 B1/B4/B8 首 chunk 的统一离线构建入口为 `bash scripts/build_trt.sh --gpu <物理编号> --model indextts2 --profile first_chunk_p258_f52_k128 --batches 1,4,8 --ref-audio /workspace/reference.wav`。GPU4 实测已完成构建和首 chunk 路由，尚非数值或生产认证。私有 HF 制品的显式发布/拉取及 Codex 扩展步骤见 [构建说明](docs/trt-build-for-codex.md)。

精度参考路径禁用 TF32；项目 kernel 与 TRT 均为可选优化。构建、数值、质量和性能分别判定；硬件、shape、精度或 plan 身份不匹配时明确回退或报错。测试与性能入口分别在 `tests/`、`benchmarks/`；通用审计代码在 `src/inspark_infer/guardrails/`。第三方来源见 [许可证与声明](THIRD_PARTY_NOTICES.md)。
