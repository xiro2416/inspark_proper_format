#!/usr/bin/env bash
set -euo pipefail
zipvoice_task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export CODEX_HOME=/workspace/.codex
export UV_CACHE_DIR="$zipvoice_task_root/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$zipvoice_task_root/.cache/python"
export TMPDIR="$zipvoice_task_root/.cache/tmp"
export XDG_CACHE_HOME="$zipvoice_task_root/.cache"
export UV_INDEX_URL="${UV_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
mkdir -p "$TMPDIR"
zipvoice_environment="${1:-builder}"
case "$zipvoice_environment" in
  builder) zipvoice_cuda_index=https://download.pytorch.org/whl/cu132 ;;
  evaluation) zipvoice_cuda_index=https://download.pytorch.org/whl/cu130 ;;
  *) echo 'Usage: bootstrap_zipvoice_build.sh [builder|evaluation]' >&2; exit 2 ;;
esac
uv venv --allow-existing --python 3.12 "$zipvoice_task_root/.venv-$zipvoice_environment"
uv pip sync --python "$zipvoice_task_root/.venv-$zipvoice_environment/bin/python" \
  "$zipvoice_task_root/configs/common/zipvoice_$zipvoice_environment.lock" \
  --extra-index-url "$zipvoice_cuda_index"
