# ZipVoice SM120 FP8

七档最终路线已在 GPU3（RTX6000D／SM120，现有600W上限）完成本机验收，并发布私有 HF 资产。GitHub 以普通 main 提交发布；最终新 checkout 复验证据单独记录。

完整模型 batch、并发1；动态总帧600/760/920、padded tokens52/78/141。参考375帧来自连续4秒 VAD 窗口及准确转写；时长按完整原文自然推导，保留8步Euler、t_shift=.5、guidance=1、feat_scale=.1。

原始 FP32 模型用于语义和配对质量参考。FM前4层浮点，后12层216个线性投影使用冻结 W8A8 E4M3FN 静态 per-tensor max 校准（128条中英真实请求、全部8步）。后12层的24个depthwise conv、敏感运算、Text、Vocos、ISTFT保持浮点。各 batch 共用配方，不重校准；原SM89 INT8与Index资产、代码闭包和历史保留。

## 性能

以下为自然760帧／78tokens，prepared CPU condition整形、H2D至所有有序PCM的同轮20样本ABBA中位数，始终启用完整batch文本编码。排除前端、初始化、warmup、capture和WAV写盘。原生 FP8 作为优化基线；B1/B2使用相同路线，其微小测量波动不计为收益。

| Batch | 最终路线 | 原生FP8 ms | 最终ms | 改善 | P95 ms | 请求/s | 峰值显存MiB | 持续功耗均值/最大W |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | native | 28.005 | 27.984 | 相同路线 | 28.021 | 35.75 | 1683.1 | 254.3/266.0 |
| 2 | native | 40.991 | 41.000 | 相同路线 | 41.168 | 48.76 | 1743.1 | 312.2/322.4 |
| 4 | attentiongeo | 65.114 | 63.427 | 2.59% | 63.686 | 63.26 | 1781.1 | 351.8/367.0 |
| 8 | attentiongeo | 121.630 | 102.788 | 15.49% | 104.423 | 77.59 | 1883.1 | 398.2/422.5 |
| 16 | attention | 234.996 | 192.178 | 18.22% | 192.848 | 83.53 | 2105.1 | 419.3/439.2 |
| 32 | attention | 494.763 | 380.437 | 23.11% | 383.116 | 83.95 | 2545.1 | 452.3/468.8 |
| 64 | attention | 1093.112 | 840.861 | 23.08% | 843.534 | 76.10 | 3405.1 | 468.7/497.2 |

P95、吞吐、显存和功耗来自各档至少30秒连续有效推理；显存为NVML整卡峰值，包含运行时与capture缓存；功耗为整个板卡的原始NVML采样，保留超过设置上限的实际读数。短句、长句、context scratch内存和采样量见每档`002-final-performance.json`。

## 配对质量

16条独立测试（10中文、6英文），完整目标batch、同一调用方噪声，检查所有行的运算/交付逻辑。质量语料使用同文本batch及符合条件的文本复用；一般性能测量关闭复用，另有完整文本编码和混合文本控制检查。表内CER/WER为比例，UTMOS和SIM-o是相对原始FP32的变化。指标仅报告，不设置固定L2门槛，不声称感知等价。

| Batch | 语言/指标 | 原始FP32 | FP8 | ΔUTMOS | ΔSIM-o |
|---:|---|---:|---:|---:|---:|
| 1 | zh/CER | 0.02941 | 0.03529 | -0.1067 | -0.0079 |
| 1 | en/WER | 0.10638 | 0.10638 | +0.0127 | +0.0046 |
| 2 | zh/CER | 0.03235 | 0.02941 | -0.0123 | -0.0038 |
| 2 | en/WER | 0.10638 | 0.10638 | +0.0536 | -0.0011 |
| 4 | zh/CER | 0.03137 | 0.03137 | -0.0115 | +0.0014 |
| 4 | en/WER | 0.10638 | 0.10638 | +0.0029 | -0.0022 |
| 8 | zh/CER | 0.03137 | 0.02941 | -0.0501 | -0.0023 |
| 8 | en/WER | 0.10638 | 0.10638 | -0.0145 | +0.0070 |
| 16 | zh/CER | 0.03137 | 0.03137 | -0.0253 | +0.0006 |
| 16 | en/WER | 0.10638 | 0.10638 | +0.0355 | +0.0028 |
| 32 | zh/CER | 0.02941 | 0.03137 | -0.0586 | -0.0016 |
| 32 | en/WER | 0.10638 | 0.10638 | +0.0278 | -0.0067 |
| 64 | zh/CER | 0.03333 | 0.04510 | -0.0213 | -0.0025 |
| 64 | en/WER | 0.10638 | 0.10638 | -0.0111 | +0.0054 |

## 安装和部署

```bash
bash scripts/bootstrap_zipvoice_fp8.sh
INSPARK_REPO_ROOT="$PWD" PYTHONPATH="$PWD/src" .venv-zipvoice-fp8/bin/python -m inspark_infer.command zipvoice ensure --precision fp8 --gpu 3
```

也可在已有环境安装项目后使用`inspark zipvoice ensure/prepare/infer --precision fp8 --gpu 3`。支持完整batch1/2/4/8/16/32/64；infer使用`--batch`、`--output`以及参考WAV/准确prompt text/完整target text，或准备好的`--inputs`和`--workload`。前端不裁剪文本或参考音频，超出profile明确报错。

私有资产需要`HF_TOKEN`，请通过环境注入，勿写入仓库。固定HF revision：`45883af1d711ef065354e0069177ecb4e5898e27`。注册表`configs/hardware/sm120/zipvoice_fp8_registry.json`；硬件配置限定RTX6000D、SM120、85651MiB、TRT11.3.0.99，Torch2.11+cu130/Triton3.6。加载核对每个引擎、插件和应用源码SHA；不兼容时不静默重建。

## 复现和证据

GPU测试始终单张GPU3、串行cooperative lease。推理环境锁位于`configs/common/zipvoice_fp8.lock`，CPU质量评估使用独立`zipvoice_fp8_evaluation.lock`。参考数据和评估模型是只读数据依赖；项目源码不跨工作区导入。

原生migration：`scripts/run_zipvoice_fp8_validation_pipeline.py`。显式候选比较：`scripts/compare_zipvoice_fp8_attention.py --variant attentiongeo --baseline attention --batches 8 16 32 64`。最终路线复验：`scripts/finalize_zipvoice_fp8.py --review <review.json>`；该命令需要原始本机私有准备数据。新缓存部署验证：`scripts/validate_zipvoice_fp8_download.py`。

TensorRT level5/FULL tiling搜索和207个FP8 GEMM tactics覆盖全部216投影。相对位置/softmax中间物化由源浮点在线注意力消除；各档按实际E2E选择原生或融合。几何调整在B4/B8有效，在B16/B32/B64退化；PCM调度单独ABBA复验。受保护浮点FFN融合多数退化，选择性760帧FF1的整请求毛上限约.05%，380帧相关形状也已检查，因此停止该方向；不声称全局最优。

公开证据：`reports/sm120/zipvoice/fp8/history/007-all-target-migration.json`、`012-retained-compute-profiles.json`、`013-protected-ffn-review.json`、`014-final-route-review.json`、`015-private-publication.json`及各batch历史。原始音频、文本、转写、状态、profile dump留在忽略的`outputs/fp8`，不发布。

HF使用新增`bundles/zipvoice/sm120/fp8/`、`fp8/sm120/`、`onnx/sm120/fp8/`，保留旧SM89文件与历史。每个bundle附许可证和来源说明。
