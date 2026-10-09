# IndexTTS2 本机 INT8 部署

本路线在 SM89 RTX4090 上验证，源码基线为 `510e7c2`；发布合并了 GitHub main 的后续 Index 混合精度与 ZipVoice 更新。可在独立 checkout 中部署。

用户指定物理 GPU1，单卡串行；支持 INT8 batch=1、2、4、8、16、32、64、128。B32 已验收，七档迁移、优化及剩余方向检查均已完成（[最终结果](multibatch/RESULTS.md)）。默认入口检查验收状态；各档配置只在完整验收后更新。原始五档结果保留在 [RESULTS.md](RESULTS.md) 与 validated-matrix.json（私有资产中的 history 记录）。

| 内容 | 位置 |
|---|---|
| 构建环境 | `.venv`，Torch2.11cu130/TRT11.3 |
| 运行环境 | `.venv-native`，Torch2.13cu130/TRTLLM1.3rc28 |
| 环境记录 | `deployment/environment.lock`、`native-environment.lock` |
| 固定权重、配方和原始源码 | `local_assets/weights`、`local_assets/download` |
| 加载容器 | `local_assets/runtime/models` |
| 本机引擎 | `artifacts/sm89/int8_smoothquant/b{B}/` |
| 部署配置 | `configs/hardware/sm89/indextts/int8_b{B}_selected.json` |
| 验收与性能记录 | `deployment/history/` |
| 日志与缓存 | `.cache/` |

```bash
cd /workspace/A_1007/indextts
bash deployment/run.sh --batch 1 \
  --ref-audio /workspace/A_TEST_REF/male_news.wav \
  --text '你好，欢迎使用语音合成。' --output outputs/int8.wav
```

入口接受 B1/2/4/8/16/32/64/128，实际入组情况由输入决定。`--batch` 指定静态引擎档位；单条命令不能当作满 batch 吞吐测试。流式接口沿用原仓库 `--stdin-stream` NDJSON。

重建与验收：

```bash
source deployment/env.sh
ACC_GPU_ALLOW_SHARED=1 .venv/bin/python deployment/build_matrix.py --batches 1 2 4 8 16
source deployment/native_env.sh
ACC_GPU_ALLOW_SHARED=1 .venv-native/bin/python deployment/validate_matrix.py
```

`ACC_GPU_ALLOW_SHARED=1` 来自用户已授权使用 GPU1 既有闲置上下文的任务条件。租约仍拒绝忙碌 GPU；不要与其他推理/构建同时运行。构建可续跑，并校验已有 engine 的 hash、硬件、batch 和构建策略。

沿用发布 SQ alpha=1.0、组件 INT8/BF16 保护策略、FP32 接口与归一化，未重新校准。所有引擎为本机 SM89 独立构建：TRT11.3、optimization level5、FULL tiling、最大可设 tactic 数、aux streams0。原 SM120 序列化引擎未使用。

CFM 四步全图在当前编译器中重复崩溃，采用原仓库已有 estimator 四次调用路径，仍计算四个区间 0/.25/.5/.75/1，复用外层 CUDA Graph。Vocoder 保留源实现 zero_insert_conv/FIR polyphase；本机完整卷积图未融合整数计算，等价 im2col+GEMM 表示恢复原生 INT8，初次部署路径为 `vocoder-gemm/`，B1/2/4/8 各76个、B16 79个实际 INT8 tactic 层，证据见 引擎检查记录（私有资产中的 history 记录）。未改权重或尺度。数值审计对比相同配方参考并报告误差，不使用额外固定 L2 门槛。

静态引擎重点覆盖首包；超出固定形状的尾包使用相同配方 Torch 回退。性能文件记录实际命中与回退，不能声称全长语音全部运行在 TRT。生成文本是功能/性能测试集，未作 ASR/MOS 或生产音质认证。

私有资产固定 revision：weights `96933af87262eec17d685050723d4f768d2f8778`，sources/calibration `2edb087d47eefc95c0c008c71e27c21a1158c896`。HF 镜像无法转发私有认证，用户授权仅这些资产直连官方 HF；其他依赖使用清华镜像。凭据只从环境 `HF_TOKEN` 或父目录忽略的 token 文件读取。可用 `deployment/fetch.py` 重取并校验 manifest。

已有 FP32 eager 语义参考保留，可显式 `--precision fp32` 调用；本次后续只优化 INT8。构建诊断与机制清单见 [WHITEBOARD](WHITEBOARD.md) 和 mechanism-inventory（私有资产中的 history 记录）。

启动入口实测已生成22050Hz、有限且非静音 WAV，记录见 cli-smoke.json（私有资产中的 history 记录），示例音频在 `outputs/int8-cli-smoke.wav`。

B32 已独立完成迁移后优化，见 [B32结果](b32/RESULTS.md) 与 [构建/运行说明](b32/README.md)。B32 通过 `bash deployment/b32/run.sh` 使用最终已验收选择；原 B1/2/4/8/16 引擎及结果保留。

七档先迁移后优化的独立记录与运行入口见 [multibatch/README.md](multibatch/README.md)。优化后首包声码器使用109个完整FIR/Snake/FIR路径与76个真实INT8隐式卷积，按各档实验证据选择tile与调度；不拆大batch、不改校准或保护浮点策略。原五档记录是历史结果。

## 从固定资产版本部署

在 checkout 根目录安装 `environment.lock` / `native-environment.lock` 对应环境后，执行：

```bash
source deployment/native_env.sh
.venv-native/bin/python scripts/download_index_sm89.py --token-file /workspace/A_1007/.cache/huggingface/token
.venv-native/bin/python deployment/fetch.py --token-file /workspace/A_1007/.cache/huggingface/token
```

也可用环境变量 `HF_TOKEN`。凭据路径按本机实际位置指定，禁止提交凭据。
`configs/hardware/sm89/indextts/assets.json` 固定私有 HF 的 engine revision 与 manifest 哈希；下载逐文件验证 SHA256。权重和校准保持原版本。下载包包含八档已验收选择，以及内部 B48 Target/Draft；不含 ONNX 重建输入。
原始构建输入绑定在发布前校验，运行包使用独立的 manifest 验证；重建仍要求原 ONNX 和外部数据哈希，不能用运行包跳过构建输入检查。
公开报告仅包含汇总指标。逐请求记录与验证清单在私有 HF 包中，下载后按原 `deployment/**/history` 路径恢复。

[B64 就绪流水线](ready_pipeline/README.md) 是显式启用的首包实验，默认 C；需要 `on_chunk` 即时发布才能提前交付。它未改变生产 Pool 默认调度。
