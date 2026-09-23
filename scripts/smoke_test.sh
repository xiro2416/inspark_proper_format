#!/usr/bin/env bash
# Historical SM120 profile smoke, not the current pure eager SM89 audit.
# For the latter use benchmarks/benchmark_reference.py with sm89_eager_fp32.json.
set -euo pipefail
if [[ $# -ne 1 ]]; then echo "usage: $0 /path/to/reference.wav" >&2; exit 2; fi
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
repo_root="$project_dir"
gpu="${INSPARK_GPU:-0}"
CUDA_VISIBLE_DEVICES="$gpu" bash "$project_dir/scripts/run.sh" "$project_dir/scripts/preflight.py"
mkdir -p "$project_dir/outputs"
bash "$project_dir/scripts/run.sh" -m inspark_infer.cli \
  --gpu "$gpu" --workers 1 --batch 1 \
  --config "$project_dir/configs/common/runtime.yaml" \
  --deployment "$project_dir/configs/hardware/sm120/sm120.json" \
  --ref-audio "$1" --text '他正在整理文件。' \
  --output "$project_dir/outputs/smoke.wav"
