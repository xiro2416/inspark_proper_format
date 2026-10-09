#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/../native_env.sh"
export INDEX_HISTORY_DIR="$index_task_root/deployment/multibatch/history"
export INSPARK_TRT_TIMING_CACHE_DIR="$index_task_root/.cache/trt113_sm89_multibatch"
export TRITON_CACHE_DIR="$index_task_root/.cache/triton_multibatch"
export CUDA_CACHE_PATH="$index_task_root/.cache/cuda_multibatch"
export ACC_GPU_ALLOW_SHARED=1
mkdir -p "$INDEX_HISTORY_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$INSPARK_TRT_TIMING_CACHE_DIR"
