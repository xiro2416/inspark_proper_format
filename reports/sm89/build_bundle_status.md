# SM89 TRT11.3 新布局构建状态（2026-09-23）

本报告只记录本次根项目迁移及统一 bundle 构建入口的实测，不把历史 engine 冒充为新布局生成物。

| 检查 | 结果 | 证据 / 限制 |
| --- | --- | --- |
| 根项目 CPU 回归 | 217 passed, 4 skipped | `CUDA_VISIBLE_DEVICES='' ACC_TRITON_TOOLCHAIN=default bash scripts/run.sh -m pytest -q tests` |
| wheel | 构建成功 | `uv build --wheel --out-dir .work/wheels/trt_build`；包名 `inspark-infer`，含 `inspark` 入口 |
| B3 预检 | `unsupported`，不下载权重 | 精确批次仅 B1/B4/B8，输出 Codex 交接 JSON |
| B1 新 bundle 构建 | **未开始构图** | 唯一授权的物理 GPU6 在租约检查时已有 21854 MiB 占用，默认安全门禁拒绝；日志保存在本地 `.staging`，未产生完整 bundle |
| B4/B8 新 bundle 构建 | 未运行 | 待 GPU6 可用后串行执行，不能引用旧 engine 当新构建通过 |
| 四组件首 chunk 路由 | 未在新 bundle 上验证 | 旧 SM89 测量见 [stage2](stage2/README.md) |
| eager 数值、完整 EOS 质量、32/64 并发、长序列、持续运行、功耗 | **未通过本次新构建验收** | 历史浮点严格门禁未通过；无新 bundle 就无法宣称新路径通过 |
| 私有 HF publish/fetch | 仅离线入口与拒绝测试完成 | 尚无逐源再分发许可声明、环境变量 token、完整 bundle 和新服务器复验；未上传 |

GPU6 的既有进程没有被停止，也没有将 `ACC_GPU_ALLOW_SHARED` 当作绕过条件。一次完整构建必须在目标卡可用后重新运行 `bash scripts/build_trt.sh --gpu 6 --batches 1,4,8 --ref-audio /workspace/index-tts/data/audio/old/mingxiang_gao.wav`，再按 [构建说明](../../docs/trt-build-for-codex.md) 做数值、质量和性能验收。本报告不构成生产认证。
