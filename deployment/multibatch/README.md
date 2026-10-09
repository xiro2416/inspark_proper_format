# IndexTTS INT8 多batch迁移与优化

目标：B1、B2、B4、B8、B16、B64、B128。源为已经验收的B32优化实现；所有迁移先完成，再进行各档优化。原B32不变，原B1–B16的配置与性能快照保存在history/bN/original-selected.json及source-*.json。权重、量化尺度、校准版本、保护浮点策略保持。

只使用物理GPU1，构建/运行串行。各档真实静态batch，引擎不拆批。约48GiB显存；若资源不足则报告该档未完成，不改变precision或偷偷切换GPU。

```bash
cd /workspace/A_1007/indextts
source deployment/multibatch/env.sh
.venv-native/bin/python deployment/multibatch/matrix.py --phase all
.venv-native/bin/python deployment/multibatch/fir_quant_matrix.py
.venv-native/bin/python deployment/multibatch/report.py
```

可用`--batches 1 2 4 8 16 64 128`指定目标，`--phase migrate`仅迁移，`--phase optimize`仅处理已经完成迁移的目标。构建使用.venv，运行使用.venv-native；源B32缓存复制到独立timingcache，CUDA/Triton缓存也独立。

matrix.py完成初轮迁移/优化；fir_quant_matrix.py随后对尚有收益可能的tile做30波复测、检查调度组合，并测量独占FIR→原INT8量化的读写融合。局部探针不足0.1%则记录并停止该方向；完整候选必须强构建、匹配E2E获益及全验收后才保留。最终状态须同时具备七档optimization-complete.json和fir-quant-matrix-decisions.json，各档实际覆盖与停止依据见retained-execution-review.json。支持续跑，已构建候选不重写；改变候选配置应使用新目录，禁止旧engine绑定被重写的ONNX。

六组件覆盖：Target/Draft/Prefill/Latent/四季度CFM估计器/BigVGAN。继承109个完整FIR/Snake/FIR插件、76个真实signed INT8隐式卷积以及六处大卷积的源tile。先minimal plain真实调用和冻结AR/声学审计，然后比较分组条件+latentprefix、完整EOS/取消重放/清理/partialbatch，再最终测量。迁移结果固定为migration-complete.json和migration-selected.json。

后续优化先检查实际层级成本/融合，测试大卷积tile及burst/packing/projection等batch相关运行选项，候选需匹配E2E获益、完整生命周期及冻结审计通过。最终验收才更新configs/hardware/sm89/indextts/int8_bN_selected.json。旧引擎仍保留，源B32插件ABI不变，新shape/newtile使用独立注册名。

各档进展和证据在history/bN/WHITEBOARD.md；活动日志见history/active.json。没有optimization-complete.json则不能称为已完成优化。性能边界为入组到该满batch最后一个首包PCM，5次预热/30波；另测30秒连续首包吞吐与GPU1整卡功耗，不能当完整语音吞吐。数值误差报告，无ASR/MOS认证；静态形状外尾包仍使用同配方Torch。

验收后使用：`bash deployment/multibatch/run.sh --batch 16 --ref-audio /workspace/A_TEST_REF/male_news.wav --text "你好。" --output outputs/b16.wav`。每档默认选择前另做实际CLI合成及有限/非静音WAV检查，失败则恢复此前默认配置。

B64/B128使用独立的history/validation-manifest-128.json（每split128条、四音色、全文长度三档，生成smoke数据）。这不用于量化校准；B1–B16保留原32条清单。不同档位的横向数值不能单独归因于batch，优化收益始终用各档同清单对照。B64最初32条清单审计不足的记录归档于history/b64/manifest32-incomplete。
