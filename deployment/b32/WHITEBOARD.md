# IndexTTS INT8 B32：迁移与优化完成

状态：validated_complete。先完成最小迁移与冻结审计，再迁移调度，最后独立优化；全部GPU任务串行使用物理GPU1，交付无活动构建/试验。源B1/2/4/8/16及ZipVoice保留，无推送/发布。

固定身份：GitHub main 510e7c2624fbd3b3693f4a2b690291948f6ceae8；weights96933af87262eec17d685050723d4f768d2f8778、calibration/source2edb087d47eefc95c0c008c71e27c21a1158c896。Draft900/CFM800/Target/BigVGAN、SmoothQuant alpha1.0、原BF16/FP32保护策略保持；无重新校准。TargetQ8/KV80、DraftQ7、Prefill48/Latent80、CFM310/prompt258四季度step、Vocoder52。

硬件：[RTX4090 SM89 48GiB](/workspace/.codex/skills/gpu_parameters/rtx4090-sm89-48g.md)，128SM/72MiB L2；profile条件I8roof635.7TOPS/BW938GB/s。构建.venv Torch2.11cu130/TRT11.3；运行.venv-native Torch2.13cu130/TRTLLM1.3rc28。实际后端DSpark框架桥接+TRT11.3+Vocoder自定义AOT/Triton，非完整Executor。基线与候选level5/FULL/max tactics2147483646/aux0，兼容源timingcache复制后独立搜索。

迁移：六组件B32完整强构建；原Vocoder99个真实INT8 tactic。先plain compute及冻结AR/声学审计，再比较framework/native和调度；保留framework、grouped conditions+cached latent prefix（配对平均收益CI36.88–50.09ms）。固定迁移记录history/migration-complete.json及migration-selected.json；migrate.py重跑只核验已完成六组件身份，不覆盖当前best。

优化：Vocoder基线148ms，暴露列展开/Slice/FIR成本。early-Q-only探针1.09%且编译器融合掉预期数据缩减，撤回主路径。109 tiled FIR激活替换，原实际六档局部收益70–80%，完整E2E及生命周期后保留。再将76量化卷积改为compact NCF输入+signed INT8 MMA隐式卷积，原权重QuantizeLinear/尺度/zero-point/保护不变；四离散dilation/stride/deconv/output-padding GPU案例精确一致。六个C192/F1664/K11卷积调度64/64/64局部约快50%且输出逐元素一致，匹配E2E验收后保留。最终Vocoder为vocoder-implicit-tuned，109 FIR+76真实INT8卷积（六个tuned独立插件名，旧引擎ABI保留）。

交付匹配复测：5预热/30波，迁移基线首包p50/p95=342.72/367.50ms，最终261.79/285.00ms，p50改善23.61%；30秒连续首包吞吐90.57→119.61请求/s（+32.05%），包括取消，不能当完整语音吞吐。配对平均收益95%CI72.00–87.19ms。整卡平均342.96→328.18W；最佳采样峰435.12W、显存14134MiB。首包device/CFM/Vocoder回退0，CFM/Vocoder Graph30/30。

完整EOS、finite/nonzero有序PCM、主动取消后同种子逐字节重放、无状态泄漏、16-row半满batch通过；冻结AR/声学、原权重配方及候选ONNX/external-data/engine身份已审计。CPU18测试通过；CLI输出4.33秒/22050Hz、finite/nonzero通过。PCM相对迁移约5.92%L2包含浮点/编译变化，只报告误差，不宣称ASR/MOS质量认证。

残余与停止依据：CFM确认13个BF16 gemm_mha_v2融合注意力区域、59个INT8 tactic，保留四个顺序季度区间；原full-solver SIGSEGV前提未变，不重复。保护浮点卷积等按原配方保持。AR burst1/4、packing+latent vector、flat projection未有一致匹配E2E收益，保留burst2；未证明全局最优，进一步搜索需新机制证据。静态首包范围外的尾包仍用原同配方Torch回退。

默认配置configs/hardware/sm89/indextts/int8_b32_selected.json。运行bash deployment/b32/run.sh；重建与选择流程见README.md。最终RESULTS.md/history/results.json，配对compare-delivery-b32.json，生命周期/冻结审计链接在current-best.json。机制清单history/mechanism-inventory.json；各阶段权重/引擎/验证和WAV档案保留。旧白板归档history/whiteboard-before-delivery.md。
