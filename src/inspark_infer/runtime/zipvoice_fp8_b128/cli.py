"""Dispatch into the separate FP8 environment before importing GPU libraries."""
import os
from pathlib import Path
import subprocess
import sys
from .common import environment, root


def main(argv=None):
    args=list(sys.argv[1:] if argv is None else argv)
    if args and args[0]=='ensure':
        selected='128'
        for i,value in enumerate(args):
            if value=='--batches':selected=args[i+1]
            elif value.startswith('--batches='):selected=value.split('=',1)[1]
        requested=[int(x) for x in selected.split(',')]
        if not requested or len(set(requested))!=len(requested) or any(b not in (1,2,4,8,16,32,64,128) for b in requested):raise ValueError('Supported FP8 batches1/2/4/8/16/32/64/128')
        old=[b for b in requested if b!=128]
        if old:
            from inspark_infer.runtime.zipvoice_fp8.cli import main as existing
            clean=[];skip=False
            for value in args:
                if skip:skip=False;continue
                if value=='--batches':skip=True;continue
                if value.startswith('--batches='):continue
                clean.append(value)
            result=existing([*clean,'--batches',','.join(map(str,old))])
            if result:return result
            if 128 not in requested:return 0
            args=[*clean,'--batches','128']
    python=Path(os.getenv('INSPARK_ZIPVOICE_FP8_PYTHON',str(root()/'.venv-zipvoice-fp8/bin/python')))
    if not python.is_file():raise RuntimeError('Run scripts/bootstrap_zipvoice_fp8.sh')
    env=environment()
    for key in ('ACC_TRT113_SITE','ACC_CLEAR_TRITON','ACC_TRITON_TOOLCHAIN'):
        env.pop(key,None)
    return subprocess.call([str(python),'-m','inspark_infer.runtime.zipvoice_fp8_b128.worker',*args],cwd=root(),env=env)
