# TensorRT 11.3 构建与 Codex 扩展说明

## 现有入口与认证边界

从干净 checkout，在 `/workspace` 下运行以下一条命令；它准备 Python、固定版本模型、TRT 11.3，然后在**一张物理 GPU**上串行构建每个精确 batch 的 Target、Draft、CFM、Vocoder：

```bash
bash scripts/build_trt.sh --gpu 4 --model indextts2 \
  --profile first_chunk_p258_f52_k128 --batches 1,4,8 \
  --ref-audio /workspace/your_reference.wav
```

`--gpu` 是本机物理编号；示例中的 4 是当前工作区获授权的卡，新服务器应填它自己的空闲物理 GPU 编号。构建器会拒绝繁忙设备。也可单独传 `--batches 4`。目前只实现 SM89、精确 B1/B4/B8、首 chunk `Target verify=8/KV128`、`Draft proposal=7/KV64 或 KV128（KV64 通过掩码和填充路由到同一个 KV128 engine；数值门禁仍未通过）`、`CFM prompt=258/total=310`、`Vocoder=52`。B4/B8 活动行缩小时会按原请求状态打包到固定 batch engine，不构成动态 batch engine。其他 SM、batch 和形状返回机器可读的 `unsupported` 与 `codex_task`；B3/B7 不会自动填充或拆批。权重从 hf-mirror 固定 revision 下载并按 SHA256 校验，engine 构建不依赖 HF engine 缓存。

入口在 `src/inspark_infer/build/trt113.py`，实际导出和构建由现有 `scripts/build_trt113_*.py`、`scripts/export_trt113_*.py` 执行。每个 batch 写入 `artifacts/trt113_bundles/sm89/first_chunk_p258_f52_k128/bN/<bundle-id>/`；失败停留在 `.staging/`，保留 `logs/` 和 `build_failure.json`，不替换完整 bundle。`manifest.json` 记录 GPU/SDK、源码、四组件 SHA256、文件清单和认证状态。`deployment.json` 只指向同 bundle 的相对 plan。2026-09-23 在物理 GPU4 完成三组实际构建，bundle ID 和首 chunk 路由数据见 [SM89 本次报告](../reports/sm89/build_bundle_status.md)。构建门禁运行真实首 chunk 四组件路由检测，但**不证明完整 EOS、浮点一致、质量、32/64 并发或性能优势**。当前数值状态为 `experimental_existing_gates_failed`，生产认证为 `false`。

本地直接使用包入口需先 `bash scripts/bootstrap.sh`，然后运行 `bash scripts/run.sh -m inspark_infer.command trt build ...`；通过 wheel 安装则在 checkout 根目录使用 `inspark trt build ...`，或设置 `INSPARK_REPO_ROOT=/workspace/<checkout>`，仍需单独准备权重和 TRT 环境。wheel 不内置构建脚本或模型。任何 shape/SM 扩展必须先给出目标 GPU 的构建和审计结果。

## 在新 SM 或 shape 上交给 Codex

将不支持组合的 `unsupported` JSON 原样交给 Codex，并提供 GPU 编号、参考 WAV 和权重访问方式。Codex 应依次检查该 SM 的 TRT/CUDA/插件/显存能力；明确四组件 IO、dtype、KV 和静态形状；扩展对应 ONNX 导出、Target/Draft 直接构图、plan 加载及精确 batch 路由。每个 `profile × batch` 独立构建和反序列化；不得仅改 batch 常量或 plan。只有实测证明单一实现不足时才增加 SM 专用 kernel，否则硬件差异留在能力检查和配置。

验收包括同输入 eager 的计算区域边界与明确浮点容差、完整 EOS 质量、首 PCM/吞吐/功耗、转换/拷贝/启动成本、活跃数下降、未命中 batch、32/64 并发、长序列及持续运行。失败必须记录为失败；首 chunk 路由通过不能当作数值认证。新增 SM 结果放入独立 `reports/smXX/`，不挪用 SM89 证据。

## 显式私有 Hugging Face 缓存

构建与缓存分开；engine、ONNX、权重不进 Git。发布前由权利人逐源确认**当前 bundle 实际嵌入**的原权重和派生 engine 可在目标私有仓库存放，准备 `distribution_attestation.json`：`{"schema":1,"reviewed":true,"redistribution_permitted":true,"sources":{"<嵌入权重的源 repo>":{"permitted":true,"evidence":"许可依据"}}}`。当前四组件 bundle 的源是 `IndexTeam/IndexTTS-2`、`xirr/inspark_marlin`、`nvidia/bigvgan_v2_22khz_80band_256x`；未嵌入的 MaskGCT、W2V 等不应被虚假声明为已审批。缺任一实际来源会拒绝上传，未知来源哈希也会拒绝。IndexTTS2 的原始 `LICENSE` 与派生声明随 bundle 上传和下载；使用者仍需遵守许可证条件。只允许私有模型仓库；token 只从 `HF_TOKEN` 环境读取，不放入命令参数、清单或 Git。

```bash
export HF_TOKEN=...  # 在 shell 安全设置，不粘贴到聊天或提交记录
bash scripts/run.sh -m inspark_infer.command trt publish \
  --bundle /workspace/.../b4/<bundle-id> \
  --repo-id your-namespace/private-trt-cache \
  --attestation /workspace/.../distribution_attestation.json
```

发布结果包含不可变的 40 位 Hub commit SHA 和 `bundle_path`。新机器先运行 `bootstrap.sh`、`download_models.sh`、`bootstrap_trt113.sh`，再显式拉取：

```bash
bash scripts/run.sh -m inspark_infer.command trt fetch \
  --repo-id your-namespace/private-trt-cache --revision <40-char-commit> \
  --bundle-path bundles/sm89/first_chunk_p258_f52_k128/b4/<bundle-id> \
  --gpu 4 --ref-audio /workspace/your_reference.wav
```

`fetch` 默认经 hf-mirror 下载，确认仓库私有、来源声明、设备型号和 SM、全文件哈希，然后在目标 GPU 上重新运行四组件首 chunk 路由。它不自动认证数值或质量；私有仓库经镜像访问也须在目标网络实际验证。CLI 存在不等于制品已经发布。

SM89 既有失败与局部通过证据见 [stage2 报告](../reports/sm89/stage2/README.md)；历史 B1/B4 构建记录见 [归档](../reports/sm89/history/TENSORRT113_B1_B4.md)。
