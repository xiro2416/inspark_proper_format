#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
"$project_dir/.venv/bin/python" "$project_dir/scripts/download_models.py" --model-root "$project_dir/models"
CUDA_VISIBLE_DEVICES="${INSPARK_PLAN_GPU:-0}" bash "$project_dir/scripts/run.sh" "$project_dir/scripts/install_kernel_plans.py"
echo "Models and measured SM120 plans are ready."

