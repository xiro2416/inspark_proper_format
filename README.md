# InSpark current inference

当前仓库只发布选定实现：**Draft：150k 在线 step900；CFM：双语40k第一阶段 step800、四步推理**。Target 和 BigVGAN 沿用原始权重。历史优化路线、淘汰候选及旧实验报告从当前目录移除，旧版本可通过 Git 历史查看。

ZipVoice-Distill 的 A_1007 INT8 B1/B2/B4/B8/B16/B32/B64 使用独立环境和私有 `xirr/zip_pipeline`，七档本地迁移和优化验证已完成，云端发布与独立下载验证正在推进。输入约定、首轮对照和验收状态见 [ZipVoice A_1007](docs/zipvoice-a1007.md)。下表及 Index 命令属于 IndexTTS2 当前发布。

| 精度 | 发布 batch |
|---|---|
| FP8 E4M3FN，1:3保护策略 | 1、8、64、128 |
| INT8 SmoothQuant，alpha=1.0，同一保护策略 | 1、8、64 |

## 执行方案

- AR：NVIDIA DSparkWorker/RNN 桥接、普通 TensorRT Draft/Target、GPU PCG和KV状态；不是完整 TRT-LLM Executor/AutoDeploy。
- 沿用各 batch 已选定 static shape、KV layout、调度、CUDA Graph、融合及 builder/tactic 参数。RNN 每轮初始状态为0，只提交接受的 Target hidden。
- CFM：四个区间 `[0,.25]`、`[.25,.5]`、`[.5,.75]`、`[.75,1]`；B128保留B64×2微批和已选定CFM plugin。
- Vocoder：小 batch普通TRT；B64保留TRT分段＋NVIDIA activation；B128保留已选定FIR实现和B64×2微批。
- 不跨请求复用相同文本的计算结果。输出PCM按请求独立交付；CUDA Graph在新进程内捕获。

## 权重与引擎

资产在私有 Hugging Face 仓库 **[xirr/index_pipeline](https://huggingface.co/xirr/index_pipeline)**。GitHub只存代码、配置和当前验证摘要；不存权重、ONNX、engine或缓存。

下载需要对该HF仓库有访问权限；使用本机HF登录或环境变量 `HF_TOKEN`，不要把凭据写进配置文件。

安装与实际验证环境见 `configs/current/environment.json`。引擎针对 **NVIDIA RTX 6000D／SM120、TensorRT 11.3.0.99**。NVIDIA activation二进制与Torch/CUDA ABI相关，需要匹配记录的运行环境。推荐在已配齐环境中安装本项目：

```bash
python -m pip install -e . --no-deps
inspark fetch --asset-dir ./local_assets --precision fp8 --batch 1
inspark infer --asset-dir ./local_assets --precision fp8 --batch 1 \
  --gpu 0 --ref-audio reference.wav --text '你好，欢迎使用。' --output output.wav
```

批量场景使用已有Pool/NDJSON接口，按所选batch组批。静态首chunk优化覆盖指定profile；超出profile明确走同权重、同量化配方的一般路径，不截断历史KV。性能测试记录实际路径与回退次数。

## 版本身份与验证

发布清单固定HF revision，记录模型张量、量化配方、引擎及依赖文件哈希。Target/Vocoder仅在权重和组件配方一致时复用；Draft/CFM按新权重重建。下载和加载会核对身份，拒绝旧权重与新引擎混用。

当前迁移验证汇总于 `reports/current/VALIDATION.md`。浮点差异单独记录，不以固定L2阈值替代运算逻辑检查；短测不等同长期稳定性或完整音质认证。

许可证与来源见 `LICENSE`、`THIRD_PARTY_NOTICES.md`、`licenses/`。

量化与权重导出入口：`scripts/calibrate_weights.py`、`scripts/export_weights.py`。运行前提供下载后的模型目录、推理用student checkpoint与完整组件校准清单；INT8使用SmoothQuant alpha=1.0。

## 独立的 ZipVoice INT8 实现

ZipVoice-Distill INT8 已在独立的 `/workspace/A_1007` 中完成 batch 1/2/4/8/16/32/64 的本地迁移与优化验证，主优化点为 760 帧，动态 profile 为 600/760/920。使用独立的 SM89 资产、Python 环境和私有 `xirr/zip_pipeline` 仓库。当前验证状态和发布流程见 [ZipVoice A_1007](docs/zipvoice-a1007.md)。旧 B16/24/32/64 资产将在新发布完成下载验证后清理；上面的 IndexTTS2 发布保留。
