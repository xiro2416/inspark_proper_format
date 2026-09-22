#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 /path/to/reference.wav [fp32|bf16|bf16-triton]" >&2
  exit 2
fi
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
repo_root="$(cd -- "$project_dir/.." && pwd)"
gpu="${INSPARK_GPU:-0}"
mode="${2:-bf16}"
case "$mode" in
  fp32|bf16) deployment="$project_dir/configs/sm89_${mode}.json" ;;
  bf16-triton) deployment="$project_dir/configs/sm89_bf16_triton.json" ;;
  *) echo "mode must be fp32, bf16, or bf16-triton" >&2; exit 2 ;;
esac
CUDA_VISIBLE_DEVICES="$gpu" bash "$project_dir/scripts/run.sh" \
  "$project_dir/scripts/preflight_sm89.py" --deployment "$deployment"
mkdir -p "$project_dir/outputs"
bash "$project_dir/scripts/run.sh" -m acc_infer_clear.cli \
  --gpu "$gpu" --workers 1 --batch 1 \
  --config "$project_dir/configs/runtime.yaml" \
  --deployment "$deployment" \
  --ref-audio "$1" --text '他正在整理文件。' \
  --output "$project_dir/outputs/sm89_${mode}_smoke.wav"
