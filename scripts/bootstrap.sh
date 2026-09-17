#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
command -v uv >/dev/null 2>&1 || { echo "Install uv first: https://docs.astral.sh/uv/" >&2; exit 1; }
cd "$project_dir"
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python -e .
uv pip install --python .venv/bin/python ninja==1.13.0
mkdir -p .toolchains/triton350
uv pip install --python .venv/bin/python --target .toolchains/triton350 --no-deps triton==3.5.0
echo "Environment ready. Next: scripts/download_models.sh"

