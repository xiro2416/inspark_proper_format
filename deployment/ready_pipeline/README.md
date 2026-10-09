# B64 readiness pipeline

已实现并验证按请求首包就绪推进的实验调度，详见 [RESULTS.md](RESULTS.md)。[selected.json](selected.json) 记录所选 **C：提前16行声学交付＋AR缩批**；它是实验侧配置，不是生产schema9部署文件。生产选择器保持原样。

GPU1单卡，复用原INT8 SmoothQuant alpha1.0、模型和framework DSPARK adapter+TensorRT backend。计算档位为64/48/32/16，声学固定B16；默认两轮一次检查。就绪请求冻结语音码/metadata，剩余请求连同KV、接受历史、预分配随机数和轮数一起迁移；不重做prefill或重新播种。

## 运行与复现

在 `indextts/` 下执行，所有GPU命令按顺序运行，一次一个作业：

```bash
source deployment/multibatch/env.sh
export TRITON_CACHE_DIR="$index_task_root/.cache/triton_ready_pipeline"
export INSPARK_TRT_TIMING_CACHE_DIR="$index_task_root/.cache/trt113_sm89_ready_pipeline"
# 五组完整对照，输出用新文件名，保护现有历史。
.venv-native/bin/python deployment/ready_pipeline/run.py --waves 30 --warmups 5 --power-seconds 30 --out deployment/ready_pipeline/history/repeat-matrix.json
# 保留C与原基线的匹配、颠倒顺序复测及边界验收。
.venv-native/bin/python deployment/ready_pipeline/run.py --modes baseline C --reference deployment/ready_pipeline/history/confirmation.json --waves 30 --warmups 5 --power-seconds 30 --lifecycle --out deployment/ready_pipeline/history/repeat-retained.json
```

`--reference` 复用已验证B48审计并要求相同部署/清单/输出重放。用原五组 `confirmation.json` 作为参考，包含baseline/A/C的同波次输出。`--modes C D` 可复测串行/重叠，参考存在时交替颠倒顺序。`--acoustic-priority -1` 是已测试但未选择的声学优先级；所选C为0。

B48 Target/Draft已构建，首次迁移命令记录如下。完整B48服务未新增：

```bash
.venv/bin/python deployment/export.py --component target --batch 48
.venv/bin/python scripts/build_unified_onnx.py --gpu 1 --onnx artifacts/sm89/int8_smoothquant/b48/target/model.onnx --engine artifacts/sm89/int8_smoothquant/b48/target/model.engine --optimization-level 5 --tiling full --aux-streams 0
.venv/bin/python deployment/export.py --component draft --batch 48
.venv/bin/python scripts/build_unified_onnx.py --gpu 1 --onnx artifacts/sm89/int8_smoothquant/b48/draft/model.onnx --engine artifacts/sm89/int8_smoothquant/b48/draft/model.engine --optimization-level 5 --tiling full --aux-streams 0
```

## 接入现有 Engine

在GPU1 lease内，先按原接口创建B64 Engine、注册参考音频并完成原B64 `prepare_deployment`，然后显式接入：

```python
from deployment.ready_pipeline.scheduler import ReadyPipeline
engine.head_ready_pipeline = ReadyPipeline(engine, mode="C", acoustic_priority=0)
# 按原接口admit_batch。回调在owner线程执行，第一组PCM可提前送出。
engine.run_ready(on_chunk=send_pcm)
# cancel仍在run_ready返回的推理边界执行。
engine.close()  # 先join pipeline，再释放模型和请求状态
```

`send_pcm` 应将PCM及时送到消费者或IO队列。若仅等待 `run_ready()` 的返回列表，应用仍会等待整组完成，无法享受提前交付。原Pool已有逐chunk IPC，但本轮没有将实验策略接入生产Pool配置。

A/B/C/D分别为：提前16串行固定AR64、提前16重叠固定64、提前16串行缩批、提前16重叠缩批。`ReadyPipeline` 只创建一个共享模型的阶段视图；各TRT engine/context、KV/Graph与可写输出独立，额外档位确实增加显存。每次最多一个声学任务在途，完成event保护D2H和请求释放；D在AR检查等待中轮询可交付输出，回调保持owner线程语义。

## 验证与边界

- 显式首包profile：F310/P258、Latent80、TargetQ8/DraftQ7/KV80；不兼容引用或上下文拒绝进入实验路径。普通生产路径沿用原行为。
- 就绪队列按检查点和入组顺序组成16行；全体AR就绪后尾组padding16，只发布真实请求。试验覆盖1/15/17请求、重放/取消/清理、提前EOS状态的交接与短PCM裁剪。
- 缩批改变浮点计算形状，生成序列可与B64不同；按原规则报告数值审计，无新增固定L2门槛。同一路径串行/重叠及重放要求PCM/code/接受序列一致。
- 实验每轮64请求全部收到首包后取消；没有完整句子质量、持续到达/动态补槽或服务公平性验收。
- `sustained` 功耗是各模式独立30秒实际服务窗口，加载/预热/哈希排除，派发/取消计入。baseline也驻留所有候选engine，不能把实验整卡显存当原生产占用。

CPU检查：

```bash
.venv-native/bin/python -m pytest -q tests/test_ready_pipeline_state.py tests/test_sm89_deployment.py tests/test_current_release_identity.py
```

GPU时间线（诊断用途，性能取无剖析测量）：

```bash
/usr/local/cuda/bin/nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end=stop --output deployment/ready_pipeline/history/repeat-trace .venv-native/bin/python deployment/ready_pipeline/run.py --reference deployment/ready_pipeline/history/confirmation.json --modes C D --waves 2 --warmups 5 --power-seconds 0 --trace-mode D --out deployment/ready_pipeline/history/repeat-trace.json
/usr/local/cuda/bin/nsys export --type sqlite --output deployment/ready_pipeline/history/repeat-trace.sqlite deployment/ready_pipeline/history/repeat-trace.nsys-rep
python deployment/ready_pipeline/analyze_trace.py deployment/ready_pipeline/history/repeat-trace.sqlite
```

[白板](WHITEBOARD.md) · 最终测量（私有资产中的 history 记录） · 最后交付检查（私有资产中的 history 记录） · 五组对照（私有资产中的 history 记录） · GPU重叠证据（私有资产中的 history 记录）
