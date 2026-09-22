#!/usr/bin/env bash
# Sequential B1/B4 acoustic build. Existing exporters retain their GPU leases.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
gpu="${1:?Usage: bash scripts/build_trt113_acoustic.sh PHYSICAL_GPU BATCH}"
batch="${2:?Specify batch 1 or 4}"
case "$batch" in 1|4) ;; *) echo "This stage builds B1 or B4 only" >&2; exit 2 ;; esac
[[ "$gpu" =~ ^[0-9]+$ ]] || { echo "PHYSICAL_GPU must be an integer" >&2; exit 2; }
: "${ACC_TRT113_SITE:?Point ACC_TRT113_SITE at the isolated TensorRT 11.3 site-packages}"
cd "$project_dir"
export CUDA_VISIBLE_DEVICES="$gpu"
# run.sh selects the project Python/NumPy/Torch stack. Builders import only TRT
# from ACC_TRT113_SITE; never place the entire isolated site on PYTHONPATH.
bash scripts/run.sh scripts/export_trt113_cfm_onnx.py --gpu "$gpu" --batch "$batch" \
  --frames 310 --prompt-frames 258 --output "artifacts/trt113_cfm/cfm_solver_b${batch}.onnx"
bash scripts/run.sh scripts/build_trt113_cfm_onnx.py --batch "$batch" --frames 310 \
  --onnx "artifacts/trt113_cfm/cfm_solver_b${batch}.onnx" \
  --engine "artifacts/trt113_cfm/cfm_solver_b${batch}.engine" \
  --plan "artifacts/trt113_cfm/plan_b${batch}.json"
bash scripts/run.sh scripts/export_trt113_vocoder_onnx.py --gpu "$gpu" --batch "$batch" \
  --frames 52 --regular-conv native \
  --output "artifacts/trt113_vocoder/vocoder_b${batch}_nativeconv.onnx"
bash scripts/run.sh scripts/build_trt113_vocoder_onnx.py --batch "$batch" --frames 52 --strongly-typed \
  --onnx "artifacts/trt113_vocoder/vocoder_b${batch}_nativeconv.onnx" \
  --engine "artifacts/trt113_vocoder/vocoder_b${batch}_nativeconv_strong.engine" \
  --plan "artifacts/trt113_vocoder/plan_b${batch}_nativeconv_strong.json"
