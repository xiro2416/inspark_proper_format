#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
cd "$index_task_root"
exec "$index_task_root/.venv-native/bin/python" "$index_task_root/deployment/infer.py" "$@" --batch 32
