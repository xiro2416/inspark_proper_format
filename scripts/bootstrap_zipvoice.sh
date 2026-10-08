#!/usr/bin/env bash
set -euo pipefail
zipvoice_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export UV_CACHE_DIR="$zipvoice_root/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$zipvoice_root/.cache/python"
export TMPDIR="$zipvoice_root/.cache/tmp"
export UV_INDEX_URL="${UV_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
mkdir -p "$TMPDIR"
uv venv --allow-existing --python 3.12 "$zipvoice_root/.venv-zipvoice"
uv pip sync --python "$zipvoice_root/.venv-zipvoice/bin/python" "$zipvoice_root/configs/common/zipvoice_runtime.lock" --extra-index-url https://download.pytorch.org/whl/cu130 --find-links https://k2-fsa.github.io/icefall/piper_phonemize.html
echo 'ZipVoice environment ready; main IndexTTS2 environment is unchanged.'
