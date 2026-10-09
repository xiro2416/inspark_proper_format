# IndexTTS INT8 B32

用户指定先迁移、后优化。源 B1/2/4/8/16 的部署、历史和缓存保留；此目录只记录 B32 的新工作。迁移与优化引擎均已验收，当前状态见 [WHITEBOARD.md](WHITEBOARD.md)，性能见 [RESULTS.md](RESULTS.md)。入口加载已验收 selected 配置。

固定 GPU1 / RTX4090 SM89、原 Draft900/CFM800/Target/BigVGAN、SQ alpha1.0、原组件 BF16/FP32 保护策略和 TRT11.3 后端。本任务只加速 INT8。B32 使用完整静态 B32 引擎，五个其他组件及最终Vocoder路径均为本机SM89构建。

```bash
cd /workspace/A_1007/indextts
source deployment/b32/env.sh
export ACC_TRT_SITE="$index_task_root/.venv/lib/python3.12/site-packages"
.venv/bin/python deployment/build_matrix.py --batches 32
source deployment/b32/env.sh
.venv-native/bin/python deployment/b32/migrate.py
```

`migrate.py` 先检查六组件的本机身份/hash/真实 INT8 计算，再用 plain 条件基线运行并做 target-shaped 数值审计；通过后才比较 worker 和源分组条件/latent prefix 组合，执行完整 EOS、取消重放、清理、未满 batch 以及最终延迟/功耗测试。所有步骤单卡串行。

迁移结束保存 `history/migration-complete.json`、`migration-selected.json`、`migration-final-int8-b32.json`，保持为后续优化的固定基线。后续优化需完成执行成本检查和匹配的无分析器端到端对照；仅降低功耗不算收益。

默认性能边界沿用源测试：入组至该 wave 最后一个首 PCM；5次预热、30波，另测30秒连续首包吞吐（包括取消，不是完整语音吞吐）。完整语音测试作为功能验收。尾包超出静态形状时允许同配方 Torch 回退；数值误差只报告，不设额外固定 L2 门槛，未做 ASR/MOS 认证。

使用 `bash deployment/b32/run.sh --ref-audio /workspace/A_TEST_REF/male_news.wav --text '你好，欢迎使用语音合成。' --output outputs/b32.wav`。入口固定 B32，复用原 WAV/NDJSON API，并使用 B32 独立 CUDA/Triton 缓存。

实际优化路线为：先完成迁移验收，再测阶段/层级成本；109个完整halo tiled FIR激活、76个紧凑NCF输入的signed INT8 MMA卷积，最后仅对六个已测C192/F1664/K11卷积改为64/64/64 tile。原权重QuantizeLinear、激活/权重尺度和保护浮点层保留。所有保留方案都经历完整EOS、取消后同种子字节重放、清理和half batch，以及冻结声学审计和无分析器E2E。AR burst、输入打包和条件投影没有一致证据，默认保留原配置。

重建最终Vocoder（五个其他组件使用已验收迁移引擎）：

```bash
source deployment/b32/env.sh
.venv/bin/python deployment/b32/export_fir_candidate.py
export ACC_TRT_SITE="$index_task_root/.venv/lib/python3.12/site-packages"
.venv/bin/python scripts/build_unified_onnx.py --gpu 1 \
  --onnx artifacts/sm89/int8_smoothquant/b32/vocoder-tiled-fir/model.onnx \
  --engine artifacts/sm89/int8_smoothquant/b32/vocoder-tiled-fir/model.engine \
  --optimization-level 5 --tiling full --aux-streams 0 \
  --small-fir-plugin --small-fir-layout all_tiled
.venv/bin/python deployment/b32/export_implicit_candidate.py
.venv/bin/python scripts/build_unified_onnx.py --gpu 1 \
  --onnx artifacts/sm89/int8_smoothquant/b32/vocoder-implicit-int8/model.onnx \
  --engine artifacts/sm89/int8_smoothquant/b32/vocoder-implicit-int8/model.engine \
  --optimization-level 5 --tiling full --aux-streams 0 \
  --small-fir-plugin --small-fir-layout all_tiled --implicit-int8-conv
.venv/bin/python deployment/b32/export_tuned_candidate.py
.venv/bin/python scripts/build_unified_onnx.py --gpu 1 \
  --onnx artifacts/sm89/int8_smoothquant/b32/vocoder-implicit-tuned/model.onnx \
  --engine artifacts/sm89/int8_smoothquant/b32/vocoder-implicit-tuned/model.engine \
  --optimization-level 5 --tiling full --aux-streams 0 \
  --small-fir-plugin --small-fir-layout all_tiled \
  --implicit-int8-conv --tuned-implicit-int8-conv
source deployment/b32/env.sh
.venv-native/bin/python deployment/b32/accept_candidate.py \
  --candidate configs/hardware/sm89/indextts/int8_b32_tuned_candidate.json \
  --label implicit-tuned --select
```

原始与最佳指标、功耗和限制见 [RESULTS.md](RESULTS.md)。源码操作只涉及本项目，未推送或发布。生产入口所需AOT注册与Triton源码都在本项目内；最终引擎明确包含自定义计算，不能称为纯框架标准ONNX路线。
