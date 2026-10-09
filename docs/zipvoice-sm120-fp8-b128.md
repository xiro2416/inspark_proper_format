# ZipVoice SM120 FP8 B128

按“先迁移、后优化”完成，完整128行模型batch、并发1，始终只用GPU3（RTX6000D/SM120/85651MiB，现有600W上限）。原七档源码闭包、资产及HF revision保留。

模型／checkpoint、216个E4M3FN W8A8投影、per-tensor static-max尺度及128条原校准样例冻结不变，不重新校准。FM前4层浮点，后12层的24个depthwise conv、敏感运算、Text/Vocos/ISTFT保持浮点。Torch2.11+cu130、TensorRT11.3.0.99、Triton3.6。总帧600/760/920、padded tokens52/78/141，参考375帧、8Eulersteps、t_shift=.5、guidance1、feat_scale.1。

## 同轮性能

计时包含prepared CPU condition整形／H2D到所有有序PCM，性能测量始终启用完整文本编码；排除前端、初始化、warmup、capture和WAV写盘。每轮20样本ABBA，中位数；迁移和优化增益采用各自同轮比较，不能把不同时段的微小差异计为新收益。

| 阶段 | 对照ms | 结果ms | 改善 |
|---|---:|---:|---:|
| 迁移：原生FP8→继承在线注意力 | 2325.255 | 1813.091 | 22.026% |
| 优化：继承路线→选择性浮点FFN融合 | 1813.043 | 1811.866 | 0.065% |
| 最终整体验收：原生FP8→最终路线 | 2324.803 | 1811.089 | 22.097% |

最终760帧持续推理P95 **1813.052ms**，吞吐 **70.66请求/s**；NVML整卡峰值显存 **4871.1MiB**，包括运行时／capture缓存。至少30秒连续有效推理的整卡功耗：平均 **463.4W**、P95 **478.1W**、采样峰值 **485.4W**。600W是设置上限，不是达到的功耗。

| 长度 | 总帧 | 中位数ms | P95ms |
|---|---:|---:|---:|
| short | 615 | 1400.150 | 1401.583 |
| primary | 760 | 1811.114 | 1813.052 |
| long | 917 | 2274.878 | 2276.641 |

## 验收与优化取舍

16条独立中英真实样例，完整原始B128 FP32审计；同一初始噪声、speech/text、mask和timegrid。候选只改变FM中选择的四个浮点FFN区域，原始FP32音频在条件／初始噪声／文本／speech／mask／时间／行身份及音频哈希逐项一致时复用。候选数值变化对照继承FP8单独标注，未冒充新的原始FP32数值差异。6项极值与混合文本检查、直接／CUDA Graph状态及波形逐位一致、128行有序交付通过。质量使用符合条件的同文本复用；一般性能不使用复用。

| 语言/指标 | 原始FP32 | 最终FP8 | ΔUTMOS | ΔSIM-o |
|---|---:|---:|---:|---:|
| zh/CER | 0.03529 | 0.03137 | -0.0392 | -0.0029 |
| en/WER | 0.11348 | 0.10638 | +0.0855 | -0.0009 |

指标报告，不设固定L2门槛，不声称感知等价。源模型和新引擎均执行完整128batch，不拆模型batch。

迁移继承了16个在线注意力边界／48个AV输出、IEEE非线性QK／RNA AV、普通TF32RNA及未舍入共享行统计；216个FP8投影映射到207个真实E4M3 GEMM tactic，剩余模块保持浮点。TensorRT level5／FULL tiling搜索、独立缓存副本，原生FM scratch约11.22GiB，继承约3.22GiB。

后续几何变体慢4.455%，全36区域残差融合慢1.507%，FFN1-only12区域慢.938%；局部35–40%加速不能当作整请求收益。六种PCM策略筛选后同轮确认没有净收益，保持chunk32/workers4。保护浮点FFN在760/380及三种宽度做源机制复测，只保留两处全长FFN1及两处半长FFN3；额外760收益仅约1ms，属于小收益。短／长句同轮验证通过。相关源机制已复查，其他选择性毛收益低于.1%，剩余原生GEMM和必要转换保留，不声称全局最优。

## 部署

```bash
bash scripts/bootstrap_zipvoice_fp8.sh
INSPARK_REPO_ROOT="$PWD" PYTHONPATH="$PWD/src" .venv-zipvoice-fp8/bin/python -m inspark_infer.command zipvoice ensure --precision fp8 --batches 128 --gpu 3
```

在已安装项目的环境可使用`inspark zipvoice infer --precision fp8 --batch 128 --gpu 3 ...`；提供`--inputs`和`--workload`，或连续4秒参考WAV／准确转写／完整目标文字。参考375帧及自然文本时长必须落在profile，不隐式裁剪。旧七档默认ensure不变；显式`--batches 64,128`可分别验证其独立注册表。

B128注册表`configs/hardware/sm120/zipvoice_fp8_b128_registry.json`，私有HF revision `a1b6f97ee4ef58221b4c1062516f66de51602dd1`；`HF_TOKEN`通过环境注入。新增路径只有`bundles/zipvoice/sm120/fp8/b128/`及整合README，原权重／尺度／ONNX复用，旧资产／LFS规则／历史保留。验证每个源码／插件／binary SHA，不兼容时不静默重建。

公开证据在`reports/sm120/zipvoice/fp8/b128/history/`：003迁移、004候选、005最终取舍、002最终性能、007私有发布。原始文字／转写／音频／状态／profile dump保持本机忽略目录。新源码命名空间`zipvoice_fp8_b128`保持原七档源码哈希；GPU始终单张GPU3、串行lease。

最终GitHub新checkout及独立空缓存的7项128行/形状/混合文本复验通过，PCM哈希一致。证据`reports/sm120/zipvoice/fp8/b128/history/008-fresh-github-checkout.json`；源码验证提交`9ca4691c545c8c71212056ca993ddf441a88359f`。
