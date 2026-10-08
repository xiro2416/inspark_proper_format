#!/usr/bin/env bash
# Source before isolated ZipVoice build or validation commands.
zipvoice_task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export CODEX_HOME=/workspace/.codex
export INSPARK_REPO_ROOT="$zipvoice_task_root"
export PYTHONPATH="$zipvoice_task_root/src"
export CUDA_VISIBLE_DEVICES=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME="$zipvoice_task_root/.cache/huggingface"
export HF_HUB_DISABLE_XET=1
export UV_CACHE_DIR="$zipvoice_task_root/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$zipvoice_task_root/.cache/python"
export TMPDIR="$zipvoice_task_root/.cache/tmp"
export XDG_CACHE_HOME="$zipvoice_task_root/.cache"
export TRITON_CACHE_DIR="$zipvoice_task_root/.cache/triton"
export CUDA_CACHE_PATH="$zipvoice_task_root/.cache/cuda"
export TORCH_HOME="$zipvoice_task_root/.cache/torch"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$TORCH_HOME"
