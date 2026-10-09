#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/native_env.sh"
cd "$index_task_root"
# This task explicitly uses GPU1 alongside its preserved idle allocation.
export ACC_GPU_ALLOW_SHARED="${ACC_GPU_ALLOW_SHARED:-1}"
exec "$index_task_root/.venv-native/bin/python" "$index_task_root/deployment/infer.py" "$@"
