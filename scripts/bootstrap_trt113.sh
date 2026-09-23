#!/usr/bin/env bash
# Keep native TensorRT 11.3 isolated from optional Torch-TensorRT 2.8/TRT 10.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
repo_root="$project_dir"
cd "$repo_root"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$repo_root/.cache/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$repo_root/.cache/uv-python}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
command -v uv >/dev/null 2>&1 || { echo "Install uv first" >&2; exit 1; }
[[ -x .venv/bin/python ]] || { echo "Run scripts/bootstrap.sh first" >&2; exit 1; }
if [[ ! -x .venv-trt113/bin/python ]]; then
  uv venv --python .venv/bin/python .venv-trt113
fi
uv pip install --python .venv-trt113/bin/python \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  tensorrt-cu12==11.3.0.99 cuda-python==13.4.1 cuda-bindings==13.4.2 \
  cuda-core==1.2.0 cuda-pathfinder==1.8.2
uv pip install --python .venv/bin/python \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  numpy==2.2.6 onnx==1.23.0 ml-dtypes==0.6.0 protobuf==7.36.2 pytest==8.4.2
export ACC_TRT113_SITE="$repo_root/.venv-trt113/lib/python3.11/site-packages"
CUDA_VISIBLE_DEVICES='' bash scripts/run.sh -c \
  'from inspark_infer.ops.tensorrt.native113 import _import_trt113; print("TensorRT", _import_trt113().__version__)'
echo "Use ACC_TRT113_SITE=$ACC_TRT113_SITE with scripts/run.sh; do not prepend it to PYTHONPATH."
