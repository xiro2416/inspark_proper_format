#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${INSPARK_PYTHON:-$project_dir/.venv/bin/python}"
toolchain_dir="$project_dir/.toolchains/triton350"
[[ -x "$python_bin" ]] || { echo "Run scripts/bootstrap.sh first" >&2; exit 1; }
[[ -f "$toolchain_dir/triton/__init__.py" ]] || { echo "Missing isolated Triton 3.5.0; run scripts/bootstrap.sh" >&2; exit 1; }
export PYTHONPATH="$toolchain_dir:$project_dir/src${PYTHONPATH:+:$PYTHONPATH}"
export ACC_CLEAR_TRITON=native
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
exec "$python_bin" "$@"

