# IndexTTS 独立请求批次重叠检查

目标：查明检查边界/声学阶段能否执行另一批请求，并验证净E2E收益；GPU1单卡、一个实验作业，源为[七档最佳](../multibatch/WHITEBOARD.md)。权重96933af、校准2edb087/SQalpha1.0、TRT11.3/Torch2.13cu130与BF16/FP32保护保持；旧配置/引擎保持。硬件[SM89/128SM/72MiB L2/47.37GiB](/workspace/.codex/skills/gpu_parameters/rtx4090-sm89-48g.md)。功耗仅观察，不作为收益。

已知依赖：FrameworkRoundRuntime在每burst Graph后status.item()等待就绪；首包run()等待全批就绪，现overlap_acoustics分支需要remaining行，因此首包路径不能靠该开关重叠。输出D2H已有独立stream，但CPU等待copy_done后才返回，尚未推进下一组。Pool已有多进程并行派发，独立CUDA context通常不能实现GPU kernel并发（无MPS时），需与同context多stream区别；不更改系统MPS服务。

实验：先B16。比较一份B16引擎处理同32请求、两份独立状态/执行context串行、同进程两份引擎各自stream并行，以及最佳单B32。批量/输入/种子/完整PCM计数一致；两个请求银行分别拥有KV/RNG/Graph/输出缓冲，禁止复用可写指针。额外resident模型成本单独控制。多stream不保证实际并发，用Nsight CUDA活动/NVTX确认；有完整结束依赖，不能把enqueue时间当完成。

参考与验收：同B16串行PCM/code哈希和接受轮序列，取消后清理和重放；首包无计算回退、CFM/Voco Graph计数完整。时延从32请求组开始到最后首PCM；统计以整个32请求wave为单位，未剖析matched E2E决定保留，power另测。不同batch编译浮点差异报告，不新增误差阈值。准备/捕获/额外profiling不计入生产时延。

状态：检查完成。Nsight确认同context双stream跨bank和跨阶段实际kernel重叠，两轮区间并集约158.06ms。B16串行32请求P50 269.32ms→并行248.59ms；驻留相同交替复测281.54→259.19ms。全部同B16输出位相同、工作计数完整、无回退和残留状态。B32参考P50244.03ms，但不同batch随机序列/AR工作量不同，不能严格归因。双份模型显存额外约7GiB，未显示优于单B32优势；仅保留显式实验probe.py，不修改生产默认。首包后取消，不宣称完整EOS/持续服务验证或长期功耗。详见[RESULTS](RESULTS.md)与history/results.json；下一步若继续，测试共享只读权重+独立写状态的受控交错。

功耗补测完成：同调度GPU1预热后连续30秒、20ms整卡瞬时采样；加载/预热/哈希排除，真实派发和取消计入。B16串行两批（第二bank驻留）平均327.13W/峰值390.86W；双B16并行平均360.95W/峰值449.93W；单B32平均349.78W/峰值446.09W。原生工作完整、无回退或残留请求；见history/power-results.json，非原始时延窗口。
