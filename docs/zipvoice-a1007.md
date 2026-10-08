# ZipVoice A_1007 INT8

七档迁移、优化、私有权重/引擎发布及全新 GitHub checkout 验证均已完成。旧 ZipVoice 本地目录、云端资产与主分支历史已按确认范围清理，IndexTTS2 保留。下文的首轮迁移与计算候选表保留各自测量条件，不能当作最终发布结果。


## 发布版本

完整延迟/功耗表见[最终结果](../reports/sm89/zipvoice/a1007/FINAL_RESULTS.md)，各档算子与调度选择、长度取舍及停止理由见[最终执行审核](../reports/sm89/zipvoice/a1007/final-execution-cost-review.json)。最终路由为 native/geo3/geo1/wp/wp/wp/normtf32k16wp；全部采用完整模型捕获。原始权重与SmoothQuant尺度不变。760帧延迟依次33.579/50.029/83.373/147.868/282.672/611.984/1314.841ms。

## 完整模型 CUDA Graph 阶段记录

完整模型 CUDA Graph 覆盖 text、8次FM/Euler、Vocos、cuFFT ISTFT和RMS；每个请求的输入H2D、新噪声和有序PCM仍执行。相对已验收迁移方案，同轮交错20次/路线，760帧结果如下。

| Batch | 候选延迟(ms) | 相对迁移版收益 | 持续整卡平均功耗(W) |
|---:|---:|---:|---:|
| 1 | 33.579 | +2.16% | 313.9 |
| 2 | 50.029 | +1.58% | 371.2 |
| 4 | 83.373 | +0.79% | 389.7 |
| 8 | 147.987 | +0.74% | 394.4 |
| 16 | 282.724 | +0.20% | 396.6 |
| 32 | 610.606 | +0.25% | 396.4 |
| 64 | 1346.071 | +0.05%（波动量级） | 396.5 |

已审核候选的11个质量样本WAV逐字节一致，完整state/wave直接执行与捕获执行一致，并通过边界与混合文本检查；质量指标沿用相同音频的既有结果。七档捕获路径已验证；B64普通TF32注意力Q64/K16及权重舍入缓存已通过独立构建和端到端验证，浮点FFN分块保留原实现。上表仅为 CUDA Graph 这一轮的对照，不能累计成最终收益。详见[工作记录](../reports/sm89/zipvoice/a1007/WHITEBOARD.md)及[独立审核](../reports/sm89/zipvoice/a1007/model-graph-review.json)。

B64另已保留普通TF32 Q64/K16注意力与现有TF32权重舍入缓存：最终当前候选760帧1314.84ms、整卡持续平均395.5W。注意力改动相对捕获路径提升1.86%；缓存进一步提升0.42%，33个质量WAV与注意力候选逐字节一致。B8缓存已保留：同轮760帧148.843→147.868ms（+0.655%），918帧+0.569%，608帧延迟增加0.106%（0.134ms）；按760帧主目标保留并披露长度取舍。持续整卡平均394.48W，P95 396.60W。真实权重/完整选中state及33个WAV与原路径一致，质量复用以相同音频为依据。B16缓存也已保留：760帧283.945→282.672ms（+0.448%），608/918帧分别+0.525%/+0.291%，持续平均396.73W、P95 398.99W。B32缓存已保留：760帧613.281→611.984ms（+0.211%），608/918帧分别+0.394%/+0.538%，持续平均396.42W、P95 398.93W。B4估算回退，保留原方案。最终版本已发布并通过独立下载验证。

各stage帧数因子为1/2/4/2/1，760对应760/380/190/380/760。不同长度分别测量，计算与调度决策按batch记录。

发布验证见[全新 checkout 回执](../reports/sm89/zipvoice/a1007/published-fresh-validation.json)。实际推理验证使用 GitHub commit `ebc5e2924caef70e72575de541acb868d6b410f5` 和最终私有 Hub revision `f6da21d25400d1b3e0b9a70de333503b3374df76`；随后仅更新文档、报告和发布状态，运行源码、权重、引擎及形状配置完全相同。

已清理103个旧代码/配置路径、46个旧Hub路径（含FP8）、14个旧LFS对象，以及三个旧本地项目目录。Hub主分支只保留最终版本提交；GitHub保留当前IndexTTS2祖先与新ZipVoice版本。参考音频、测试文本和Codex目录保留。详见[Hub历史审核](../reports/sm89/zipvoice/a1007/hub-final-history-audit.json)、[本地目录清理](../reports/sm89/zipvoice/a1007/local-tree-retirement.json)及[Index保留证据](../reports/sm89/zipvoice/a1007/index-preservation-proof.json)。

<!-- COMPUTE_REVIEW_BEGIN -->
## 计算候选正式审查（历史阶段记录）

每行是在同一轮内交错测量的760帧完整batch中位延迟；`native`是原生INT8，`inherited`是目标形状的继承方案。以下结果不能称为最终优化或发布结果。短/长帧单独保留在每档020报告中。B64沿用已完成首轮验证的继承计算。

| Batch | 同次对照 | 对照(ms) | 候选 | 候选(ms) | 主目标收益 | 计算选择 |
|---:|---|---:|---|---:|---:|---|
| 1 | native | 35.46 | geo3 | 35.67 | -0.59% | 保留对照 |
| 2 | native | 52.82 | geo3 | 51.09 | +3.28% | 保留候选 |
| 4 | inherited | 89.79 | geo1 | 85.87 | +4.37% | 保留候选 |
| 8 | inherited | 150.78 | geo1 | 152.65 | -1.24% | 保留对照 |
| 16 | inherited | 290.03 | geo1 | 291.90 | -0.64% | 保留对照 |
| 32 | inherited | 615.11 | geo1 | 622.57 | -1.21% | 保留对照 |

质量变化见 `reports/sm89/zipvoice/a1007/candidate-quality-review.json`；路径映射通过并不等于CER/UTMOS/SIM-o相同。这些计算选择随后与PCM/H2D/D2H调度整合，七档迁移已验收；完整证据见`migration-baseline-audit.json`与`migration-final-review.json`。
<!-- COMPUTE_REVIEW_END -->

<!-- INITIAL_MIGRATION_BEGIN -->
## 首轮迁移对照（历史阶段记录）

760帧、78tokens、完整模型batch；每路线20次交错测量，中位延迟从准备好的CPU条件整形/H2D计到全部有序PCM。参考音频4秒，原始生成波形4.096秒。继承方案是本轮适配前的对照，不能称为最终最佳方案。

| Batch | 原生INT8中位延迟(ms) | 继承方案中位延迟(ms) | 继承方案收益 | 继承方案持续功耗平均/P95/采样最大(W) |
|---:|---:|---:|---:|---:|
| 1 | 35.15 | 50.34 | -43.22% | 231.7 / 236.9 / 237.2 |
| 2 | 52.52 | 64.53 | -22.86% | 299.3 / 307.2 / 307.8 |
| 4 | 93.55 | 90.05 | +3.75% | 368.1 / 378.8 / 380.4 |
| 8 | 179.04 | 151.90 | +15.16% | 390.1 / 395.3 / 396.0 |
| 16 | 365.81 | 289.60 | +20.83% | 394.8 / 398.8 / 399.8 |
| 32 | 784.06 | 613.21 | +21.79% | 396.2 / 398.6 / 399.6 |
| 64 | 1699.99 | 1350.81 | +20.54% | 395.4 / 399.7 / 400.5 |

功耗配置/执行上限400W。GPU1保留已知4108MiB空闲分配，NVML显示主机PID1820496在当前PID空间之外；空闲连续采样0%/~26W，未触碰外部进程。这不是独占GPU认证。详见`reports/sm89/zipvoice/a1007/idle-context-observation.json`。

B1/B2继承方案明显回退，B4在608帧也有小幅回退；先做实际FM profiling和源几何重查，再选择保留/调整/撤回的机制。600/760/920引擎边界与608/760/918真实音频结果分别记录，未混作一组性能结论。
<!-- INITIAL_MIGRATION_END -->

## 输入约定

支持完整 batch 1、2、4、8、16、32、64，并发度1。每档使用总帧数 min600 / opt760 / max920、文本 token min52 / opt78 / max141 的动态 profile。各帧长的性能结论分别记录。参考音频必须先明确完成连续4秒 VAD 截取及对应 ASR；推理入口不会隐式裁剪音频或文本。

参考375帧包含在总帧数内。生成部分225/385/545帧，经 CENTER ISTFT 后原始波形分别为2.389333/4.096/5.802667秒。最终 WAV 经过原有静音边缘处理及尾部停顿补足，实际时长以输出文件为准。

保留原 token 比例时长算法，完整文本自然落在支持范围内才接受。边界张量测试与真实音频质量测试分开记录。INT8 SmoothQuant alpha=.5、前4层浮点/后12层量化、原权重和尺度保持不变。

## 独立构建与验证

```bash
cd /workspace/A_1007
source scripts/zipvoice_env.sh
.venv-builder/bin/python scripts/prepare_zipvoice_a1007.py
.venv-builder/bin/python scripts/build_zipvoice_matrix.py
.venv-zipvoice/bin/python scripts/run_zipvoice_validation.py
```

构建器采用 TensorRT11.3、level5、FULL tiling、最高支持的 tactic 数量及串行共享工作区所需的 aux0。缓存来自已记录的相同硬件源方案，目标形状仍进行搜索。构建成功与迁移验收分开记录。

源 ONNX、INT8/Eager/Vocos 权重在 `models/zipvoice`。FM 可以直接由保存的 INT8 权重重导出，无需训练数据、重新校准或旧项目：

```bash
CUDA_VISIBLE_DEVICES= .venv-zipvoice/bin/python scripts/export_zipvoice_fm.py \
  --output .work/reexport-new/fm.onnx
.venv-builder/bin/python scripts/prepare_zipvoice_dw_layout.py \
  --source .work/reexport-new/fm.onnx \
  --output .work/reexport-new/layout/fm-inherited.onnx
```

已验证重导出的原生图和 depthwise 布局图，在规范化 batch IO 符号及外部权重文件名后，与构建源图 protobuf 完全一致；外部权重字节一致。重导出结果不会自动替换已验收的 engine。

## 迁移机制与证据

完整继承计算包含120个原生 INT8 模块、24个 INT8 depthwise、12个 INT8 非线性投影/value/gate融合、24个 INT8 投影/残差融合、12个浮点 FFN RNA，以及48个顺序注意力分支。B1至B16以原B16方案为来源；B32和B64分别沿用各自来源。目标验收后，B1选原生、B2选geo3、B4选geo1、B8/16/32/64选继承计算；各机制的移除或调整有目标证据，不把完整继承覆盖误称为所有档最终路线。Depthwise使用INT8存储权重的浮点反量化/卷积。

先验证目标计算，再验收 CUDA Graph、文本复用/混合文本回退、共享缓冲区、cuFFT ISTFT、H2D及有序 PCM 调度。每档白板与详细证据位于 `reports/sm89/zipvoice/a1007`。所有 batch 迁移完成后才开始新一轮优化。

延迟计时为已准备的 CPU 条件、请求整形及H2D到全部有序 PCM；不含文本前端、初始化、预热、Graph capture、WAV写盘。功耗另以至少30秒真实持续推理采集整卡 NVML 平均/P95/采样最大值，同时记录400W配置上限。

## 发布与清理顺序

新方案全部验收后发布代码到 `xiro2416/inspark_proper_format`，必要权重及 engine bundle 到私有 `xirr/zip_pipeline`，随后进行全新缓存下载复验。HF旧版本及废弃对象清理完成后，再将最终 revision 固定到 GitHub 注册表。

GitHub保留IndexTTS2原有内容和历史，移除被替代的ZipVoice提交；使用期望旧head的 force-with-lease。最后删除旧ZipVoice本地副本、旧profile引擎及FP8产物。原始用户数据、IndexTTS2及共享Codex资源保留。
