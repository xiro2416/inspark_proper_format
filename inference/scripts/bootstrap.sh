#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
repo_root="$(cd -- "$project_dir/.." && pwd)"
command -v uv >/dev/null 2>&1 || { echo "Install uv first: https://docs.astral.sh/uv/" >&2; exit 1; }
cd "$repo_root"
export UV_INDEX_URL="${UV_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$repo_root/.cache/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$repo_root/.cache/uv-python}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
if [[ ! -x .venv/bin/python ]]; then uv venv --python 3.11 .venv; fi
uv pip install --python .venv/bin/python torch==2.8.0 torchaudio==2.8.0
uv pip install --python .venv/bin/python -e "$project_dir"
uv pip install --python .venv/bin/python ninja==1.13.0
uv pip install --python .venv/bin/python pytest==8.4.2
mkdir -p .toolchains/triton350
uv pip install --python .venv/bin/python --target .toolchains/triton350 --no-deps triton==3.5.0
echo "Environment ready. Next: scripts/download_models.sh"
