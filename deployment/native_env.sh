#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
export ACC_TRT_SITE="$index_task_root/.venv-native/lib/python3.12/site-packages"
export LD_LIBRARY_PATH="$index_task_root/.venv-native/lib:$index_task_root/.venv-native/lib/openmpi:${LD_LIBRARY_PATH:-}"
export MPI4PY_MPIABI=openmpi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
