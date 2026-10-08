"""Dispatch into the separate FP8 environment before importing GPU libraries."""
import os
from pathlib import Path
import subprocess
import sys
from .common import environment, root


def main(argv=None):
    args=list(sys.argv[1:] if argv is None else argv)
    python=Path(os.getenv('INSPARK_ZIPVOICE_FP8_PYTHON',str(root()/'.venv-zipvoice-fp8/bin/python')))
    if not python.is_file():raise RuntimeError('Run scripts/bootstrap_zipvoice_fp8.sh')
    env=environment()
    for key in ('ACC_TRT113_SITE','ACC_CLEAR_TRITON','ACC_TRITON_TOOLCHAIN'):
        env.pop(key,None)
    return subprocess.call([str(python),'-m','inspark_infer.runtime.zipvoice_fp8.worker',*args],cwd=root(),env=env)
