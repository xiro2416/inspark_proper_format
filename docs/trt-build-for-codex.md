# TensorRT 11.3 构建与 Codex 扩展说明

## 现有入口与认证边界

从干净 checkout，在 `/workspace` 下运行以下一条命令。它准备 Python、固定版本模型和 TRT 11.3，随后按**本地完整 bundle → 精确匹配的私有 HF 固定 revision → 在指定物理 GPU 上串行构建**解析每个 batch：

```bash
bash scripts/build_trt.sh --gpu 4 --model indextts2 \
  --profile first_chunk_p258_f52_k128 --batches 1,4,8 \
  --ref-audio /workspace/your_reference.wav --allow-experimental
```

`--gpu` 是本机物理编号；示例中的 4 是当前工作区获授权的卡，新服务器应填自己的空闲物理 GPU 编号。构建器会拒绝繁忙设备。也可单独传 `--batches 4`。固定形状是首 chunk `Target verify=8/KV128`、`Draft proposal=7/KV64 或 KV128（KV64 通过掩码和填充路由到同一个 KV128 engine；数值门禁仍未通过）`、`CFM prompt=258/total=310`、`Vocoder=52`。B4/B8 活动行缩小时会按原请求状态打包到固定 batch engine，不构成动态 batch engine。SM80+ 可尝试同一 TRT 11.3 导出/构建链路，但**只有 SM89 48GB RTX 4090 B1/B4/B8 有实测证据**；在其他 SM 上成功构建也仍需独立审计。其他 batch 和形状返回机器可读的 `unsupported` 与 `codex_task`；B3/B7 不会自动填充或拆批。权重从 hf-mirror 固定 revision 下载并按 SHA256 校验。

解析入口在 `src/inspark_infer/build/ensure.py`，构建编排在 `src/inspark_infer/build/trt113.py`，实际导出和构建由现有 `scripts/build_trt113_*.py`、`scripts/export_trt113_*.py` 执行。每个 batch 写入 `artifacts/trt113_bundles/smXX/first_chunk_p258_f52_k128/bN/<bundle-id>/`；失败停留在 `.staging/`，保留 `logs/` 和 `build_failure.json`，不替换完整 bundle。`manifest.json` 记录 GPU/SDK、源码、四组件 SHA256、builder/量化策略、文件清单和认证状态；旧 SM89 schema 1 bundle 只通过固定注册 ID 兼容读取，新构建为 schema 2。`deployment.json` 只指向同 bundle 的相对 plan。2026-09-23 在物理 GPU4 完成三组实际构建，bundle ID 和首 chunk 路由数据见 [SM89 本次报告](../reports/sm89/build_bundle_status.md)。构建门禁运行真实首 chunk 四组件路由检测，但**不证明完整 EOS、浮点一致、质量、32/64 并发或性能优势**。当前数值状态为 `experimental_existing_gates_failed`，生产认证为 `false`。

`--mode auto|reuse-only|build-only` 控制解析方式；`auto` 是默认值，`build-only` 显式跳过缓存，`reuse-only` 禁止构建。使用任何实验 bundle 都必须传 `--allow-experimental`。SM89 的固定私有 HF 索引在 `configs/hardware/sm89/trt113_hf_registry.json`；命中该索引但令牌/鉴权失败时**报错，不擅自重建**，新服务器须在环境中安全设置 `HF_TOKEN`。24GB RTX 4090 的显存类别不匹配 48GB 索引，因此会在本机尝试构建，而不会复用 48GB engine。`ensure` 不自动发布到 HF；发布仍走下面的显式、私有且带许可证明的命令。`scripts/build_trt.sh` 是全环境准备入口，并将 `HF_HUB_OFFLINE` 默认为 `0`；若依赖和模型已就绪，也可设置 `HF_HUB_OFFLINE=0` 后直接用 `bash scripts/run.sh -m inspark_infer.command trt ensure ...`。

默认 builder 策略见 `configs/common/trt113_builder.json`：Target optimization level 3，其余组件 5；tiling NONE，WORKSPACE 上限每组件 8 GiB，强类型、禁用 TF32。这些是当前 SM89 的起点，不是跨 SM 性能结论。修改策略后 bundle 身份会改变；只在真实端到端和热点分析显示必要时调参，并分别记录构建时间、首 chunk、质量和功耗。量化意图独立放在 `configs/common/trt113_quantization.json`；当前仅 `none` 可执行，FP8/NVFP4 会明确报错。未来须先校准/转换并导出显式 Q/DQ/scale 图，随后先用相同 builder 基线构建、测 Q/DQ 融合和搬运成本，再决定是否调参；现有 Triton FP8 路径不是 TRT FP8 engine。

本地直接使用包入口需先 `bash scripts/bootstrap.sh`，然后运行 `bash scripts/run.sh -m inspark_infer.command trt build ...`；通过 wheel 安装则在 checkout 根目录使用 `inspark trt build ...`，或设置 `INSPARK_REPO_ROOT=/workspace/<checkout>`，仍需单独准备权重和 TRT 环境。wheel 不内置构建脚本或模型。任何 shape/SM 扩展必须先给出目标 GPU 的构建和审计结果。

## 在新 SM 或 shape 上交给 Codex

对支持的固定 shape，新 SM 应直接运行上述 `ensure --mode build-only --allow-experimental`，让 TensorRT 在目标 GPU 上选 tactic；不要复制整套模型，也不要先造 SM 专用 kernel。若导出、插件、显存或路由失败，将 `.staging/` 的失败日志和 `unsupported` JSON 交给 Codex，并提供 GPU 编号、参考 WAV 和权重访问方式。Codex 应检查该 SM 的 TRT/CUDA/插件能力、四组件 IO、dtype、KV 和静态形状；只修复实际不兼容处。每个 `profile × batch` 独立构建和反序列化；不得仅改 batch 常量或 plan。只有实测证明单一实现不足时才增加 SM 专用 kernel，否则差异留在能力检查和配置。

验收包括同输入 eager 的计算区域边界与明确浮点容差、完整 EOS 质量、首 PCM/吞吐/功耗、转换/拷贝/启动成本、活跃数下降、未命中 batch、32/64 并发、长序列及持续运行。失败必须记录为失败；首 chunk 路由通过不能当作数值认证。新增 SM 结果放入独立 `reports/smXX/`，不挪用 SM89 证据。

## 显式私有 Hugging Face 缓存

构建与缓存分开；engine、ONNX、权重不进 Git。发布前由权利人逐源确认**当前 bundle 实际嵌入**的原权重和派生 engine 可在目标私有仓库存放，准备 `distribution_attestation.json`：`{"schema":1,"reviewed":true,"redistribution_permitted":true,"sources":{"<嵌入权重的源 repo>":{"permitted":true,"evidence":"许可依据"}}}`。当前四组件 bundle 的源是 `IndexTeam/IndexTTS-2`、`xirr/inspark_marlin`、`nvidia/bigvgan_v2_22khz_80band_256x`；未嵌入的 MaskGCT、W2V 等不应被虚假声明为已审批。缺任一实际来源会拒绝上传，未知来源哈希也会拒绝。IndexTTS2 的原始 `LICENSE` 与派生声明随 bundle 上传和下载；使用者仍需遵守许可证条件。只允许私有模型仓库；token 只从 `HF_TOKEN` 环境读取，不放入命令参数、清单或 Git。

```bash
export HF_TOKEN=...  # 在 shell 安全设置，不粘贴到聊天或提交记录
HF_HUB_OFFLINE=0 HF_ENDPOINT=https://huggingface.co \
  bash scripts/run.sh -m inspark_infer.command trt publish \
  --bundle /workspace/.../b4/<bundle-id> \
  --repo-id your-namespace/private-trt-cache \
  --attestation /workspace/.../distribution_attestation.json
```

发布结果包含不可变的 40 位 Hub commit SHA 和 `bundle_path`。新机器先运行 `bootstrap.sh`、`download_models.sh`、`bootstrap_trt113.sh`，再显式拉取：

```bash
HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=0 HF_ENDPOINT=https://huggingface.co \
  bash scripts/run.sh -m inspark_infer.command trt fetch \
  --repo-id your-namespace/private-trt-cache --revision <40-char-commit> \
  --bundle-path bundles/sm89/first_chunk_p258_f52_k128/b4/<bundle-id> \
  --gpu 4 --ref-audio /workspace/your_reference.wav \
  --endpoint https://hf-mirror.com
```

`fetch` 默认先经 hf-mirror 下载；若私有仓库元数据在镜像不可用，仅对该下载失败回退到官方 Hub，并在结果记录实际端点。本环境的 Xet 大文件传输曾停滞，示例用 `HF_HUB_DISABLE_XET=1` 走可续传的普通 HTTP；若新机器的 Xet 正常，可去掉该变量。随后确认仓库私有、来源声明、设备型号和 SM、全文件哈希，在目标 GPU 上重新运行四组件首 chunk 路由。它不自动认证数值或质量；新服务器仍须验证私有仓库认证、网络和对应 GPU 型号。

SM89 既有失败与局部通过证据见 [stage2 报告](../reports/sm89/stage2/README.md)；历史 B1/B4 构建记录见 [归档](../reports/sm89/history/TENSORRT113_B1_B4.md)。
