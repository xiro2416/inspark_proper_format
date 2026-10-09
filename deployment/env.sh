#!/usr/bin/env bash
index_task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export CODEX_HOME=/workspace/.codex
export PYTHONPATH="$index_task_root/src:$index_task_root/scripts:$index_task_root"
export CUDA_VISIBLE_DEVICES=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME="$index_task_root/.cache/huggingface"
export HF_HUB_DISABLE_XET=1
export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export UV_CACHE_DIR="$index_task_root/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$index_task_root/.cache/python"
export TMPDIR="$index_task_root/.cache/tmp"
export XDG_CACHE_HOME="$index_task_root/.cache"
export TORCH_HOME="$index_task_root/.cache/torch"
export TRITON_CACHE_DIR="$index_task_root/.cache/triton"
export CUDA_CACHE_PATH="$index_task_root/.cache/cuda"
export NLTK_DATA="$index_task_root/.cache/nltk_data"
export MPLCONFIGDIR="$index_task_root/.cache/matplotlib"
export ACC_TRT_SITE="$index_task_root/.venv/lib/python3.12/site-packages"
export INSPARK_TRT_TIMING_CACHE_DIR="$index_task_root/.cache/trt113_sm89"
mkdir -p "$TMPDIR" "$HF_HOME" "$TORCH_HOME" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$NLTK_DATA" "$MPLCONFIGDIR"

if [[ -d "$index_task_root/deployment/publication/references" ]]; then
  export INSPARK_REFERENCE_ROOT="$index_task_root/deployment/publication/references"
fi
