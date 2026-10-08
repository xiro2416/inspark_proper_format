#!/usr/bin/env bash
set -euo pipefail
zipvoice_fp8_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export UV_CACHE_DIR="$zipvoice_fp8_root/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$zipvoice_fp8_root/.cache/python"
export TMPDIR="$zipvoice_fp8_root/.cache/tmp"
mkdir -p "$TMPDIR"
uv venv --allow-existing --python 3.12 "$zipvoice_fp8_root/.venv-zipvoice-fp8"
uv pip sync --python "$zipvoice_fp8_root/.venv-zipvoice-fp8/bin/python" \
  "$zipvoice_fp8_root/configs/common/zipvoice_fp8.lock" \
  --index-url https://pypi.org/simple \
  --extra-index-url https://download.pytorch.org/whl/cu130 \
  --index-strategy unsafe-best-match \
  --find-links https://k2-fsa.github.io/icefall/piper_phonemize.html
