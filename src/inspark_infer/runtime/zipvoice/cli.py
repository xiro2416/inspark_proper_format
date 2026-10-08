"""Launch ZipVoice in its own Python/Torch environment without changing IndexTTS2."""
import argparse,os,subprocess,sys
from pathlib import Path
from inspark_infer.build.zipvoice import root

def main(argv=None):
 args=list(sys.argv[1:] if argv is None else argv)
 python=Path(os.getenv('INSPARK_ZIPVOICE_PYTHON',str(root()/'.venv-zipvoice/bin/python')))
 if not python.is_file():raise RuntimeError('Run scripts/bootstrap_zipvoice.sh or set INSPARK_ZIPVOICE_PYTHON')
 env=dict(os.environ);env['INSPARK_REPO_ROOT']=str(root());env['PYTHONPATH']=str(root()/'src')
 env.pop('ACC_TRT113_SITE',None);env.pop('ACC_CLEAR_TRITON',None);env.pop('ACC_TRITON_TOOLCHAIN',None)
 env['PYTHONDONTWRITEBYTECODE']='1';env['HF_HOME']=env.get('HF_HOME',str(root()/'.cache/huggingface'))
 env['HF_ENDPOINT']=env.get('HF_ENDPOINT','https://hf-mirror.com');env['HF_HUB_DISABLE_XET']='1'
 env['XDG_CACHE_HOME']=str(root()/'.cache/zipvoice');env['TRITON_CACHE_DIR']=str(root()/'.cache/zipvoice/triton');env['CUDA_CACHE_PATH']=str(root()/'.cache/zipvoice/cuda');env['TMPDIR']=str(root()/'.cache/zipvoice/tmp')
 for key in ('XDG_CACHE_HOME','TRITON_CACHE_DIR','CUDA_CACHE_PATH','TMPDIR'):Path(env[key]).mkdir(parents=True,exist_ok=True)
 return subprocess.call([str(python),'-m','inspark_infer.runtime.zipvoice.worker',*args],env=env,cwd=root())
