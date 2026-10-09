# IndexTTS INT8 SM89 本机结果

历史首轮部署：B1/2/4/8/16 全部通过本机验收。后续七档先迁移后优化结果在 [多batch报告](multibatch/RESULTS.md)，B32 在 [B32报告](b32/RESULTS.md)。以下表格保留原始测量，不代表更新后的默认性能。

| Batch | 后端 | 首包 p50 ms | p95 ms | 首包 requests/s |
|---|---|---:|---:|---:|
| 1 | framework + Graph | 39.67 | 49.40 | 24.22 |
| 2 | native + Graph | 50.48 | 59.70 | 39.17 |
| 4 | framework + Graph | 66.93 | 77.05 | 59.52 |
| 8 | framework + Graph | 106.51 | 113.14 | 74.87 |
| 16 | framework + Graph | 178.26 | 192.43 | 86.72 |

延迟从入组到接收该 wave 最后一个首 PCM；5次预热、30波。吞吐为另行连续30秒首包测试，包含取消，不能当作完整语音吞吐。

完整EOS、运行中取消、逐字节重放与状态清理见 `history/lifecycle-selected-b*.json`；未满batch检查覆盖B>1。
权重身份与同配方数值审计见 `history/audit-*-b*.json`；误差仅报告，未作ASR/MOS音质认证。
首包使用本机TRT INT8；超出静态shape的尾包允许同配方Torch回退。CFM四步全图编译失败，使用既有四次estimator调用和外层Graph。

构建、验收与调用方式见 [部署说明](README.md)。
