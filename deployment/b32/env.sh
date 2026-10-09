#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/../native_env.sh"
export INDEX_HISTORY_DIR="$index_task_root/deployment/b32/history"
export INSPARK_TRT_TIMING_CACHE_DIR="$index_task_root/.cache/trt113_sm89_b32"
export TRITON_CACHE_DIR="$index_task_root/.cache/triton_b32"
export CUDA_CACHE_PATH="$index_task_root/.cache/cuda_b32"
export ACC_GPU_ALLOW_SHARED=1
mkdir -p "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"
