#!/usr/bin/env bash
# Source-attested AR engines; legacy engine files are retained for comparison.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
repo_root="$project_dir"
gpu="${1:?Usage: bash scripts/build_trt113_ar.sh PHYSICAL_GPU BATCH}"
batch="${2:?Specify batch 1, 4 or 8}"
case "$batch" in 1|4|8) ;; *) echo "Supported batches: 1, 4, 8" >&2; exit 2 ;; esac
[[ "$gpu" =~ ^[0-9]+$ ]] || { echo "PHYSICAL_GPU must be an integer" >&2; exit 2; }
: "${ACC_TRT113_SITE:?Point ACC_TRT113_SITE at the isolated TensorRT 11.3 site-packages}"
cd "$project_dir"
mkdir -p outputs/trt113_stage1_build
for component in target draft; do
  build_log="outputs/trt113_stage1_build/${component}_b${batch}.log"
  echo "Building source-attested ${component} B${batch} on GPU${gpu}"
  if ! bash scripts/run.sh "scripts/build_trt113_${component}_full.py" \
      --gpu "$gpu" --batch "$batch" \
      --out-dir "artifacts/trt113_${component}_full/audited_source" \
      --plan "artifacts/trt113_${component}_full/plan_b${batch}_audited_source.json" \
      > "$build_log" 2>&1; then
    tail -60 "$build_log" >&2
    exit 1
  fi
  echo "Built ${component} B${batch}; log: ${build_log}"
done
