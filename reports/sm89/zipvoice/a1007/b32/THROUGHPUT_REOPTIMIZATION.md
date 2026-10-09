# B32 / 760 帧吞吐优化结果（GPU6）

本轮保留固定 B32/760 的 FM 构建：最终同进程 ABBA 对照吞吐提高 **0.863%**。这是小幅提升，未解决 B32 吞吐扩展问题，不能据此宣称优于 B16 或接近全局最优。

| 方案 | 请求数 | 平均延迟 ms | P99 ms | 吞吐 条音频/s | 平均整卡功耗 W | NVML 采样最大 W |
|---|---:|---:|---:|---:|---:|---:|
| 原发布版 | 200 | 624.464 | 630.647 | 51.1831 | 396.821 | 400.868 |
| 固定 B32/760 候选 | 200 | 619.153 | 625.889 | 51.6246 | 396.516 | 401.827 |

每条请求均完成完整 B32，无拆成 B16。GPU6 单卡，400W 上限保持；同 CUDA context 下 control/candidate/candidate/control，每块预热30、测量100，种子9102–9201。两个候选块吞吐51.7195/51.5301，两个原版块51.2522/51.1141。之前独立进程 ABBA40 得到+0.556%，当时一组持平，因此追加本轮稳态对照。最终仍有约0.3%的块间漂移，不将本轮小收益外推到其他GPU或服务时长。

延迟从预备 CPU 条件整形/H2D开始，到全部32条有序PCM完成，包含8步FM、Vocos、ISTFT、PCM；不含ASR/G2P、初始化、预热、capture及WAV落盘。吞吐按各块首请求开始至末请求完成的墙钟合并，计入请求间检查间隔；条音频/s不等于批请求/s。参考4秒，375帧；生成原始4.096秒，385帧。PCM静音裁剪后时长会不同。

功耗来自50ms NVML整卡采样，平均按请求窗口有效采样数加权；400W是配置上限，400.868/401.827W是未截断的传感器采样最大值。既有空闲分配16206MiB保持，不能将整卡占用全部归因于本任务。峰值原版18754.31MiB、候选18672.31MiB；这些是同进程块峰值，不是隔离显存增量测量。没有更改时钟或功耗上限。

## 保留的改变

仅 FM ONNX 输入/输出 batch32、frame760 静态化并重新执行 TensorRT builder level5/FULL tactic搜索；图算子、外部权重、SmoothQuant alpha=.5、前4浮点/后12 INT8、180个量化模块的参数尺度和原版 b32_wp 算子均保持。构建导致tactic/调度与舍入结果可能改变，收益应归于整个静态构建，不能单独归于某个kernel。文本token仍支持原profile52–141；仅FM帧数锁定760。

本地候选bundle：`artifacts/trt113_bundles/zipvoice/sm89/int8/b32/270f2c741d495edac7b0`；FM SHA256 `f28711d5ebfe7bad0422dffb74c9dd71af4e657255e5cb826e50df63eb71b799`。保留原发布bundle `d1542a79cd19d660c8fa`。默认registry及GitHub/Hugging Face本轮均未更新，运行候选须显式`--bundle`；尚无生产认证。

## 正确性与实现审查

七个完整B32案例含四个自然参考、token52/141边界、混合完整文本：噪声/条件/mask/时间网格与原版逐项一致，全部32行输入及有序PCM独立验证，direct/graph state及wave逐位一致。输出非原版逐位相同；相对L2仅作记录，未擅加门槛。公开worker入口已实测，哈希绑定和非760拒绝验证通过。

四个参考、每route12条WAV的辅助质量：CER 3.419%→2.991%，UTMOS 2.4916→2.4011，SIM-o 0.6525→0.6393。不是质量改进宣称；轻微下降已披露，小样本不代表所有文本。主参考CER均0。

## 未保留的尝试

- FFN GEMM tile/warps/stages探针：没有有价值的微基准收益。
- 普通attention K16：微基准较好、完整E2E较慢，保持K32。
- Nonlinear QK IEEE→TF32：微基准更慢。
- Nonlinear Q64/3×128拆分：初次AOT生成漏改launch，缺写2/3输出；原先+1.70%无效并撤销。修正后真实TRT sentinel验证380/760全部384通道完整、AOT/JIT一致；实际E2E相对static候选-0.186%，不保留。
- FFN两层投影/activation融合：真实TRT reference、完整512维输出，相对L2=0；融合kernel在380/760明显较慢，最佳仍慢约79%，无需整模型构建。
- PCM chunk4、LUT量化、更多CPU线程：微基准收益未转为E2E，chunk4 -0.257%、LUT -0.997%；保持chunk6/workers4，选中WAV逐位相同。

原有CUDA Graph、共享context/输出arena、ISTFT、紧凑条件传输、有序PCM与D2H重叠、静态浮点权重预打包保留。当前测过的tile/fusion/work分配/CPU调度路径未显示进一步有价值的净收益，停止本轮独立探索；功耗接近上限不等于已证明唯一瓶颈。需新机制或新profile证据再开启后续实验。

## 复现与证据

下面路径请使用新的output目录以保留原始测量。原版对照只需替换bundle为`d1542a79cd19d660c8fa`。完整ABBA脚本`run_steady_final.py`及每块command/report均保留；脚本使用固定输出目录，重跑须在副本中替换该目录。

```bash
cd /workspace/A_1007
env PYTHONPATH=src INSPARK_REPO_ROOT=/workspace/A_1007 \
  ACC_GPU_ALLOW_SHARED=1 XDG_CACHE_HOME=/workspace/A_1007/.cache \
  TMPDIR=/workspace/A_1007/.cache/tmp HF_HOME=/workspace/A_1007/.cache/huggingface \
  .venv-zipvoice/bin/python -m inspark_infer.runtime.zipvoice.worker infer \
  --batch 32 --gpu 6 \
  --bundle artifacts/trt113_bundles/zipvoice/sm89/int8/b32/270f2c741d495edac7b0 \
  --inputs outputs/zipvoice-validation/cases/condition-r0-s0.safetensors \
  --workload outputs/zipvoice-benchmarks/b32/throughput-reoptimization/retained-workload.json \
  --output outputs/zipvoice-benchmarks/b32/throughput-reoptimization/reproduce-static-new \
  --warmup 30 --repetitions 100 --save-indices 0 1 31
```

- [最终统计](reoptimization/final-summary.json)、[完成审计](reoptimization/final-audit.json)
- 完整原始ABBA脚本与逐请求测量在本地实验目录；每块统计已收录最终统计。
- [七案例路径映射](reoptimization/static-mapping-review.json)、[参数审查](reoptimization/static-parameter-review.json)、[质量指标](reoptimization/static-corpus-quality.json)
- 公开部署入口测试通过，见完成审计的`public_worker_entry_verified`。
- 详细方案和否决证据在本目录history043–050及实验目录。被否决的新增public源码已移至实验目录archived-src，同名.work构建源码保留；需要复现失败方案可将archived-src中的相对路径复制回src，均不参与候选bundle。

## 后续发布更新

本轮优化完成时的“默认registry及云端未更新”是历史状态。用户随后要求发布：现在默认B32选择固定760帧bundle `270f2c741d495edac7b0`，旧动态bundle保留在Hub对照，其他batch配置不变。新revision与上传/全新下载证据见 [发布回执](reoptimization/publication.json)。上述收益和质量测量未重新测试。
