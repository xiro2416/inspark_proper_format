#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
repo_root="$project_dir"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-$repo_root/.cache/huggingface}"
"$repo_root/.venv/bin/python" "$project_dir/scripts/download_models.py" --model-root "$repo_root/models"
if [[ "${INSPARK_INSTALL_HISTORICAL_SM120_PLANS:-0}" == 1 ]]; then
  CUDA_VISIBLE_DEVICES="${INSPARK_PLAN_GPU:?Set physical SM120 GPU explicitly}" bash "$project_dir/scripts/run.sh" "$project_dir/scripts/install_kernel_plans.py"
fi
echo "Model assets are ready. Historical SM120 plan installation is opt-in; SM89 engines must be built separately."
