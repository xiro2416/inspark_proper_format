# IndexTTS 独立批次重叠检查

检查同一 GPU1、同一 CUDA primary context 中，两份独立请求状态能否利用不同 stream 重叠计算。使用现有最佳 INT8 engine，不改量化/权重/数学；`probe.py` 是显式实验入口，未修改生产 Pool 或默认配置。

从 `indextts/` 执行：

```bash
source deployment/multibatch/env.sh
.venv-native/bin/python deployment/overlap/probe.py --batch 16 --waves 30 --warmups 5 --paired --out deployment/overlap/history/b16-confirmation.json
.venv-native/bin/python deployment/overlap/probe.py --batch 32 --single-only --waves 30 --warmups 5 --out deployment/overlap/history/b32-control.json
```

以上按顺序执行，GPU1 一次只运行一个实验作业。探针只测首包：每轮32个独立请求，以最后一个请求的第一段PCM为完成点，然后取消并检查状态清理；不代表完整句子服务、持续到达或公平性验收。两个 bank 独立模型/执行 context/KV/RNG/Graph/输出缓冲，但共享CUDA context。额外一份模型增加显存，未实现权重共享。

未剖析结果决定性能；时间线用于判断是否实际重叠：

```bash
/usr/local/cuda/bin/nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end=stop --output deployment/overlap/history/b16-concurrent-trace .venv-native/bin/python deployment/overlap/probe.py --batch 16 --waves 2 --warmups 2 --trace --out deployment/overlap/history/b16-trace-probe.json
/usr/local/cuda/bin/nsys export --type sqlite --output deployment/overlap/history/b16-concurrent-trace.sqlite deployment/overlap/history/b16-concurrent-trace.nsys-rep
python deployment/overlap/analyze_trace.py deployment/overlap/history/b16-concurrent-trace.sqlite
```

输出文件如已存在，使用新路径。分析按同一线程的 NVTX 阶段归属 CUDA launch，以 GPU kernel 区间并集计算跨 bank 重叠，避免把CPU enqueue重叠当GPU重叠。`synchronization_overlap` 还记录同步等待期间另一 bank 的实际 kernel 执行时间；此时间与跨kernel重叠不能相加当收益。

`one_bank_serial` 是只装载一份B16引擎、串行完成32请求的原始比较。`resident_one_bank_serial` 控制第二份模型驻留成本，交替测试与并行执行采用相同驻留配置。`two_bank_serial` 还包含两次run_wave之间的结果哈希/清理，故只作诊断，不作为主要加速分母。`board_lifecycle` 包含加载、预热、清理，不能作为持续服务功耗。

[结果](RESULTS.md)、[白板](WHITEBOARD.md)、原始记录在 `history/`。

补测上述三种调度的持续首包功耗（每种30秒，预热后、无哈希开销；计入真实派发和取消）：

```bash
.venv-native/bin/python deployment/overlap/probe.py --batch 16 --waves 5 --warmups 5 --power-seconds 30 --out deployment/overlap/history/b16-power.json
.venv-native/bin/python deployment/overlap/probe.py --batch 32 --single-only --waves 5 --warmups 5 --power-seconds 30 --out deployment/overlap/history/b32-power.json
```

`--power-seconds` 单独统计 `resident_one_bank_serial`、`two_bank_concurrent` 或 `single_bank` 的 `sustained` 窗口。GPU1整卡瞬时功耗约每20ms采样，平均/峰值均来自指定窗口；不使用包含加载阶段的 `board_lifecycle`。原始传感器样本和原生Graph/无回退计数保留于报告。
