#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
repo_root="$project_dir"
python_bin="${INSPARK_PYTHON:-$repo_root/.venv/bin/python}"
toolchain_dir="$repo_root/.toolchains/triton350"
[[ -x "$python_bin" ]] || { echo "Run scripts/bootstrap.sh first" >&2; exit 1; }
export TMPDIR="${TMPDIR:-$repo_root/.cache/tmp}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$repo_root/.cache}"
export HF_HOME="${HF_HOME:-$repo_root/.cache/huggingface}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$repo_root/.cache/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$repo_root/.cache/torchinductor}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$repo_root/.cache/torch_extensions}"
export TORCH_HOME="${TORCH_HOME:-$repo_root/.cache/torch}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-$repo_root/.cache/cuda}"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$CUDA_CACHE_PATH"
case "${ACC_TRITON_TOOLCHAIN:-custom}" in
  custom)
    [[ -f "$toolchain_dir/triton/__init__.py" ]] || { echo "Missing isolated Triton 3.5.0; run scripts/bootstrap.sh" >&2; exit 1; }
    export PYTHONPATH="$toolchain_dir:$project_dir/src:$project_dir${PYTHONPATH:+:$PYTHONPATH}"
    export ACC_CLEAR_TRITON=native ;;
  default)
    export PYTHONPATH="$project_dir/src:$project_dir${PYTHONPATH:+:$PYTHONPATH}"
    export ACC_CLEAR_TRITON=default ;;
  *) echo "ACC_TRITON_TOOLCHAIN must be custom or default" >&2; exit 2 ;;
esac
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
case "${1:-}" in
  scripts/benchmark_*|scripts/profile_*|scripts/report_sm89_*|scripts/planner_v2_device_probe.py)
    set -- "benchmarks/${1#scripts/}" "${@:2}" ;;
esac
cd "$project_dir"
exec "$python_bin" "$@"
