# SM89 审计运行与复现

这里只说明如何产生证据，不表示任何后端已通过审计。已发布结果及局限见
[SM89 stage2 报告](../../reports/sm89/stage2/README.md)；
[compile 数值说明](compile_numerics.md) 单独记录编译路径的失败和有限通过范围。
其他架构、形状、语料不能继承 SM89 结论。

## 环境与素材

以下命令从仓库根目录运行。根目录 `scripts/run.sh` 使用项目 `.venv`，
并把工作目录切到 `inference/`，因此参数中的 `configs/`、`outputs/` 均相对
`inference/`；`artifacts/`、`models/`、`.toolchains/` 留在仓库根目录。
各 GPU 命令顺序执行，只使用物理 GPU 6；不要另行启动并行 GPU 测试。

```bash
cd /workspace/A_inspark_marlin
export ACC_TRT113_SITE=/workspace/A_inspark_marlin/.venv-trt113/lib/python3.11/site-packages
export ACC_TRITON_TOOLCHAIN=custom
```

主环境与隔离 TensorRT 的准备入口分别是 `scripts/bootstrap.sh`、
`inference/scripts/bootstrap_trt113.sh`；模型来源见
[model_sources.json](../configs/model_sources.json)，实际文件身份由每次运行的
`model_provenance` 记录，不以历史清单代替当前文件哈希。
TensorRT 11.3 从 `ACC_TRT113_SITE` 选择性加载，**不要将整个 site-packages
放进 PYTHONPATH**，否则可能替换主环境 NumPy/Torch。
`custom` 使用隔离 Triton 3.5.0，供项目自定义 kernel/TRT 配置使用；
`default` 使用主 Torch 配套 Triton，供纯 eager 和 `torch.compile` 使用。
不要同时把两套 Triton 路径手工加入 PYTHONPATH。

GPU lease 默认拒绝已有负载。仅在明确获准共享 GPU 6 时设置
`ACC_GPU_ALLOW_SHARED=1`；报告必须保留预先占用的显存及共享标记，不能宣称独占性能，
也不要终止外部进程。下面不默认开启这一例外。

[sm89_quality_256.json](../configs/sm89_quality_256.json) 含 256 条固定文本、
情绪、种子及 9 个参考音频的绝对路径，音频位于
`/workspace/index-tts/data/audio/old/`。这些私有/外部素材不随仓库上传，
需要用户提供同一素材；`generation_summary.json.references` 记录全部 9 个
音频的 SHA256，较小 capture 的 `references` 只记录其实际使用的音频。
复现须逐个核对文件哈希和完整 corpus 哈希，不能只核对文件名。
改路径会改变 corpus 哈希；改文本、情绪、种子或参考音频内容都需要重新生成、
配对评估，不能沿用旧质量结论。重新抽取语料还需要 corpus 中记录的外部原始文本。
全新 clone 不具备上述素材、模型、TensorRT engines 或下述外部评测器。

## 构建有来源证据的 B1/B4 engines

保留已有历史 engine/build 文件；以下脚本写入 `audited_source` 路径，
重复运行前先保全该路径中的旧证据，不要覆盖一份已发布报告所引用的 artifact。

```bash
for batch in 1 4; do
  bash scripts/build_trt113_ar.sh 6 "$batch"
  bash scripts/build_trt113_acoustic.sh 6 "$batch" audited_source
done
```

AR 脚本构建 Target/Draft。声学脚本依次导出及构建 CFM/Vocoder：CFM 为
310 frames、258 prompt frames；Vocoder 为 52 frames，使用
`--regular-conv native`、`--strongly-typed`。当前 B1/B4
`sm89_trt113_safe_b*.json` 引用对应的来源绑定 plan。
核对导出 `.export.json`、build JSON 和 plan 中的实际 checkpoint、源码、ONNX
及外部 tensor、engine 哈希、软件和硬件信息；公开示例见
[声学构建证据](../../reports/sm89/stage2/builds/)。
`recorded_not_audited` 仅说明有来源记录，不代表数值通过。
旧 ONNX/engine 缺失来源证据时仍是 `legacy_unverified`，
不能事后用当前权重哈希或新 build metadata 回填历史身份。

## 冻结真实调用，再独立 eager 重放

[audit_real_ar.py](../scripts/audit_real_ar.py) 在完整 EOS 生成中记录有界的
Target/Draft 调用，同时记录声学调用。下例 B1；B4 另用全新目录，
把 deployment、`--batch`、`--cases` 分别改为 B4、4、4。
`--ar-max-calls 4` 是有界采样，不是全部 AR 步骤覆盖。

```bash
bash scripts/run.sh scripts/audit_real_ar.py --gpu 6 \
  --config configs/runtime_reference.yaml capture \
  --deployment configs/sm89_trt113_safe_b1.json \
  --corpus configs/sm89_quality_256.json --batch 1 --cases 1 \
  --ar-max-calls 4 --output-dir outputs/reproduce/ar_b1

ACC_TRITON_TOOLCHAIN=default bash scripts/run.sh scripts/audit_real_ar.py --gpu 6 \
  --config configs/runtime_reference.yaml reference \
  --capture-dir outputs/reproduce/ar_b1 --precision fp32
ACC_TRITON_TOOLCHAIN=default bash scripts/run.sh scripts/audit_real_ar.py --gpu 6 \
  --config configs/runtime_reference.yaml reference \
  --capture-dir outputs/reproduce/ar_b1 --precision bf16
```

这两个 reference 命令需分别执行；FP32 失败也要独立完成 BF16。
输出为 `ar_reference_fp32/report.json` 与 `ar_reference_bf16/report.json`。
以冻结的真实输入/KV 为共同边界，FP32 重放提升的是已冻结的输入精度，
不是声称恢复一条从起点开始的未舍入 FP32 采样轨迹。

声学独立 capture/replay 的准确入口如下；同样对 B4 使用新目录及匹配参数。
也可直接对上述 AR capture 目录运行声学 reference，复用它记录的声学输入。

```bash
bash scripts/run.sh scripts/audit_real_acoustics.py --gpu 6 \
  --config configs/runtime_reference.yaml capture \
  --deployment configs/sm89_trt113_safe_b1.json \
  --corpus configs/sm89_quality_256.json --batch 1 --cases 1 \
  --output-dir outputs/reproduce/acoustic_b1

ACC_TRITON_TOOLCHAIN=default bash scripts/run.sh scripts/audit_real_acoustics.py --gpu 6 \
  --config configs/runtime_reference.yaml reference \
  --capture-dir outputs/reproduce/acoustic_b1 --precision fp32
ACC_TRITON_TOOLCHAIN=default bash scripts/run.sh scripts/audit_real_acoustics.py --gpu 6 \
  --config configs/runtime_reference.yaml reference \
  --capture-dir outputs/reproduce/acoustic_b1 --precision bf16
```

声学输出为 `reference_fp32/report.json` 与 `reference_bf16/report.json`。
重放会先写报告，再以退出码 **2** 表示数值/覆盖门槛失败；不得删除失败报告、
放宽容差或把退出码吞掉后标记通过。已有非空重放目录拒绝覆盖。
先运行 FP32，再运行 BF16，可另外记录两个 reference 之间的精度误差。
容差来自 [numerics.py](../src/acc_infer_clear/guardrails/numerics.py)：
FP32 为 `atol=1e-5, rtol=1e-4`，BF16 为 `atol=rtol=1e-2`，
按候选计算精度选择，不按输出存储 dtype 选择；BF16 对 FP32 参考仍用 BF16 门槛。
Graph/direct 一致不等于 eager 一致，重放插桩也不是性能基准。

冻结的输入、输出、KV 等 `.pt` bundle 只留在本地；公开 JSON 中保留其
SHA256、形状、调用路径和来源字段。不要上传 `.pt`、模型权重、ONNX、engine 或音频。
没有同一组本地 bundle 时可以重新 capture，但不能声称复用了原报告的冻结输入。

## 请求 RNG 与 acceptance 元数据

```bash
bash scripts/run.sh scripts/validate_request_rng_isolation.py \
  --device cuda --gpu 6 --rounds 4 \
  --output outputs/reproduce/request_rng_cuda.json
bash scripts/run.sh scripts/validate_acceptance_metadata.py \
  --gpu 6 --batches 1 4 8 --eos 8193 --max-tokens 1500 \
  --asg /workspace/A_inspark_marlin/models/asg/target_asg_threshold_0p49.safetensors \
  --output outputs/reproduce/acceptance_metadata.json
```

前者使用受控的相同概率输入检查请求 RNG 隔离，不保证浮点神经网络在不同 batch
下产生相同轨迹；后者用真实 ASG 文件比较 fused acceptance/prefix 与参考结果，
核对 flags、tokens、EOS 等离散元数据。CPU 调试分别可用 `--device cpu` 和
`--reference-only`，但 CPU 结果不是 CUDA kernel 通过证据。

## 256 条完整 EOS 质量评测

两臂独立生成完整音频，保存请求种子、emotion、codes、chunk/EOS 和音频哈希；
首包、截断音频或不足 256 对不能替代发布门槛。下面以 FP32 eager 对 B1 TRT 为例：

```bash
ACC_TRITON_TOOLCHAIN=default bash scripts/run.sh scripts/generate_quality_corpus.py \
  --gpu 6 --config configs/runtime_reference.yaml \
  --deployment configs/sm89_eager_fp32.json --corpus configs/sm89_quality_256.json \
  --cases 256 --batch 1 --output-dir outputs/reproduce/quality_fp32
bash scripts/run.sh scripts/generate_quality_corpus.py \
  --gpu 6 --config configs/runtime_reference.yaml \
  --deployment configs/sm89_trt113_safe_b1.json --corpus configs/sm89_quality_256.json \
  --cases 256 --batch 1 --output-dir outputs/reproduce/quality_trt_b1
```

评测使用独立的现有 CPU 环境 `/workspace/.venv/bin/python`，不是项目推理环境。
还依赖 `/workspace/ZipVoice` 中的 UTMOS/音频预处理实现，以及
`/workspace/models/TTS_eval_models` 中的 Paraformer 中文 ASR 和 UTMOS 权重。
这些外部源码、依赖和模型均需另行提供；不要把下面命令当作 clean-clone 安装教程。
评测器离线运行并禁止网络，运行时记录软件版本、评测源码与模型文件哈希，
以实际报告为准，不从未完成评测推断版本或通过结果。

```bash
CUDA_VISIBLE_DEVICES='' INSPARK_PYTHON=/workspace/.venv/bin/python \
  ACC_TRITON_TOOLCHAIN=default bash scripts/run.sh scripts/evaluate_quality_corpus.py \
  --corpus configs/sm89_quality_256.json --expected-cases 256 \
  --baseline-dir outputs/reproduce/quality_fp32 \
  --candidate-dir outputs/reproduce/quality_trt_b1 \
  --zipvoice /workspace/ZipVoice --models /workspace/models/TTS_eval_models \
  --cache /workspace/A_inspark_marlin/.work/quality_eval_cache \
  --report outputs/reproduce/quality_report.json
```

评测核对覆盖、EOS、音频哈希及双方实际 checkpoint 身份；质量策略见
[quality.py](../src/acc_infer_clear/guardrails/quality.py)：平均 UTMOS 相对下降
不超过 3%，字符 CER 绝对增加不超过 0.02。质量不通过同样返回 2 并保留报告。
质量通过不能抵销浮点一致性失败，生成成功也不等于评测完成。

## 持续运行与长序列

```bash
bash scripts/run.sh benchmarks/soak_requests.py --gpu 6 \
  --config configs/runtime_reference.yaml --corpus configs/sm89_quality_256.json \
  --reference /workspace/index-tts/data/audio/old/mingxiang_gao.wav \
  --concurrency 1 4 8 16 --batch 8 --seconds 600 --strict-isolation \
  --min-kv-length 129 --allow-oom-skip-concurrency 16 \
  --output outputs/reproduce/soak.json
```

这是请求并发 1/4/8/16，各 10 分钟；`--batch 8` 是最大模型 microbatch，
不是并发数。默认随并发选择 `safe_b1/b4/b8`：B8 仍引用 legacy engines，
B1/B4 的来源审计不能替它背书；仅测新 B1/B4 engines 时另行显式选择 `--batch 4`，
并在报告中保留这一不同的运行配置。
测试检查完成 EOS、取消/槽位回收、实际 KV>128 和显存/RSS 增长及趋势预算。
允许的 B16 OOM 必须记录为跳过，`completed_with_allowed_skip` 不是全档通过。
32/64 是工具支持的额外场景，不是这里已运行的证据。持续运行通过也不等于
数值、模型质量或无期限无泄漏保证；所有结论以对应的原始报告为准。
