# IndexTTS INT8 SM89 部署

用户范围：只做INT8，B1/2/4/8/16；物理GPU1，单卡串行。保留ZipVoice及既有FP32参考，不扩展其他精度。独立GitHub main checkout510e7c2。

硬件：[RTX4090 SM89 48GB](/workspace/.codex/skills/gpu_parameters/rtx4090-sm89-48g.md)。原引擎SM120不复用；已授权GPU1既有闲置约4GB上下文、仅私有xirr/index_pipeline直连HF。其他请求默认国内镜像。

固定weights revision96933af，sources/calibration2edb087；Draft900/CFM800，SQalpha1.0、组件保护BF16/FP32策略不变，不重新校准。manifest/SHA/tensor identities已验证。

构建 `.venv` Torch2.11cu130/TRT11.3；运行 `.venv-native` Torch2.13cu130/TRTLLM1.3rc28，依赖检查通过；27 CPU测试通过。相关NVIDIA Worker/RNN方法与固定源AST相同。

强构建：level5/FULL/max tactics2147483646/aux0、默认workspace。原始表示30引擎已构建。CFM四步全图 modern/legacy、CUDA13.0/13.2均SIGSEGV，采用已有estimator四次enqueue+外层Graph，四区间/掩码/累加保持；history/compiler-failures.json。

关键修复：Vocoder原Conv1d完整图只有FP32/BF16计算和INT8 Q/DQ。小Conv1d/Conv2d/强类型/axis探针均可INT8，完整Conv2d仍未融合。等价im2col+GEMM保持尺度、几何和保护策略，B1/2/4/8恢复76 INT8 tactic，B16为79。沿用zero_insert_conv/FIR polyphase。旧引擎/对照保留，history/baseline-conv1d保存旧基准记录。

B1新GEMM路径完整EOS、有序非静音PCM、运行中取消、逐字节重放、状态清理、权重身份、AR/四波声学数值报告均通过；30波/5预热/30秒持续实测：framework+Graph p50=39.67ms、p95=49.40ms，无首包回退。旧约39.77ms，不能宣称显著差异；真正新增的是整数计算覆盖。Graph相对INT8 direct的匹配收益CI39–54ms；compare-graph-b1.json。

通用机制：请求RNG/KV所有权、首包优先、PCM交付、CG保留；B>1 grouped conditions/latent prefix已逐档对比plain，均实测获益并保留。清单history/mechanism-inventory.json。静态优化首包；超出shape的尾包允许同配方Torch回退。数值仅报告，无固定L2门槛，未作ASR/MOS认证。

当前：30/30 INT8引擎和B1/2/4/8/16全部验收完成，队列会话90064退出0。最终p50=39.67/50.48/66.93/106.51/178.26ms；p95=49.40/59.70/77.05/113.14/192.43ms；30秒首包requests/s=24.22/39.17/59.52/74.87/86.72。B2选native，其余framework，全部Graph；B>1保留分组条件+latent prefix，匹配收益CI依次1.12–3.20/0.22–3.23/6.63–11.76/18.92–28.37ms。完整EOS/取消后字节重放/清理、B>1partial、全部数值审计通过；首包零回退和30/30声学Graph命中。results.json状态validated_complete。源码未推送。

复现：source deployment/env.sh; ACC_GPU_ALLOW_SHARED=1 .venv/bin/python deployment/build_matrix.py --batches 1 2 4 8 16。默认vocoder-gemm/CFM-estimator。验收source deployment/native_env.sh; ACC_GPU_ALLOW_SHARED=1 .venv-native/bin/python deployment/validate_matrix.py。运行bash deployment/run.sh默认INT8/GPU1。

收尾完成：真实deployment/run.sh默认INT8/GPU1调用退出0，WAV22050Hz、3.11秒、有限且非静音；history/cli-smoke.json。两层git diff --check通过，27 CPU测试在最后代码修改后已通过，环境检查通过。最终报告deployment/RESULTS.md，history/results.json状态validated_complete。无活动构建/验收队列；源码未推送。
