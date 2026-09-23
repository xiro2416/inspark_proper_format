#!/usr/bin/env bash
# One command from a checkout: set up dependencies/assets, then build offline.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
PYTHONPATH="$project_dir/src" python3 -m inspark_infer.command trt ensure --preflight-only "$@"
bash scripts/bootstrap.sh
bash scripts/download_models.sh
bash scripts/bootstrap_trt113.sh
export ACC_TRT113_SITE="$project_dir/.venv-trt113/lib/python3.11/site-packages"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
exec bash scripts/run.sh -m inspark_infer.command trt ensure "$@"
