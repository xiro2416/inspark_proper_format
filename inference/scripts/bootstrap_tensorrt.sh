#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
repo_root="$(cd -- "$project_dir/.." && pwd)"
cd "$repo_root"
[[ -x .venv/bin/python ]] || { echo "Run scripts/bootstrap.sh first" >&2; exit 1; }
uv pip install --python .venv/bin/python \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  torch-tensorrt==2.8.0
.venv/bin/python -c 'import torch_tensorrt,tensorrt; print(torch_tensorrt.__version__,tensorrt.__version__)'
