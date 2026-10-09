# IndexTTS INT8 七档迁移与优化

目标B1/2/4/8/16/64/128，先全部迁移，再优化；源[已验收B32](../b32/WHITEBOARD.md)。GPU1单卡串行，保持weights96933af、source/calibration2edb087、SQalpha1.0和原BF16/FP32保护；无重新校准。源引擎、历史与缓存保持，新记录独立history/bN。目标首包p50/吞吐，功耗不作惩罚。

[GPU画像](/workspace/.codex/skills/gpu_parameters/rtx4090-sm89-48g.md)：SM89/128SM/72MiB L2/47.37GiB；400W条件I8roof635.7TOPS、BW938GB/s。构建TRT11.3/Torch2.11cu130，运行Torch2.13cu130/TRTLLM1.3rc28；实际DSpark桥接+TRT+AOT/Triton，非完整Executor。level5/FULL/max tactics/aux0，无人为workspace限制。

源覆盖109完整FIR/Snake/FIR、76 compact signed-I8卷积、六处64/64/64 tile；原四顺序CFM、protected浮点保持。TargetQ8/KV80、DraftQ7、Prefill48/Latent80、CFM310/prompt258、Vocoder52。先plain真实计算/冻结审计，再迁移conditions/prefix并核验调度；B64/B128不拆批。

当前七档全部完成，指标见[RESULTS](RESULTS.md)，配置configs/hardware/sm89/indextts/int8_bN_selected.json。新增保留burst1(B1/2/4/16)、本档tile(B2/128)、grouped条件+prefix组合(B4)、72条原FIR→INT8融合(B4/64/128)。实际每档76 I8卷积、CFM13 attention融合；融合档37原FIR+72融合，其他109原FIR。完整EOS/cancel字节重放/清理/partial、同配方审计和CLI均通过；30波匹配E2E/30秒功耗。24项CPU检查通过。

B8无额外候选通过最终复测，保留迁移方案；其他tile/runtime/融合的接受或拒绝见history/bN/*decision.json和retained-execution-review.json。B64旧32条清单不足已归档，大档改为128条独立验证清单，非校准。B4未选候选曾被重写ONNX，现独立强构建复查并保留故障诊断；恢复检查完整hash/相对路径，禁止覆盖已构建候选。

停止依据：真实保留profile及覆盖核验完成，相关tile、物化、调度疑问已有局部与匹配E2E判断；高成本保护浮点卷积/四步CFM按固定配方保持。完整solver源编译崩溃条件未变，不重复无依据构建；无必做未决项，不保证全局最优。边界：生成smoke清单/四音色/短首段可重复，无文本复用；不同档清单不宜作单因素比较。首包吞吐含取消，不代表全长吞吐；尾包原同配方Torch，无ASR/MOS。

复现见[README](README.md)：source deployment/multibatch/env.sh；matrix.py --phase all，再fir_quant_matrix.py与report.py。运行bash deployment/multibatch/run.sh --batch 16 --ref-audio /workspace/A_TEST_REF/male_news.wav --text '你好。' --output outputs/b16.wav。原B32及ZipVoice保持；无提交/上传/发布。
