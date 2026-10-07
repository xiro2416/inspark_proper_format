"""Small shared TensorRT builder policy for the A_0924 static engines."""

from pathlib import Path
import hashlib
import os


_CACHE_PREPARATIONS = {}


def timing_cache_metadata(path):
    """Metadata for this process's last preparation of the requested cache."""
    return dict(_CACHE_PREPARATIONS.get(str(Path(path).resolve()), {}))


def prepare(config, trt, *, component, tiling, workspace_bytes=0, max_aux_streams=0,
            l2_limit_for_tiling=-1, seed_timing_cache=True):
    config.tiling_optimization_level=getattr(trt.TilingOptimizationLevel,tiling.upper())
    config.max_aux_streams=int(max_aux_streams)
    config.clear_flag(trt.BuilderFlag.TF32)
    if workspace_bytes:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,int(workspace_bytes))
    if l2_limit_for_tiling>=0:
        config.l2_limit_for_tiling=int(l2_limit_for_tiling)
    version=trt.__version__.replace('.','_')
    path=Path(os.environ.get('INSPARK_TRT_TIMING_CACHE_DIR', '.cache/trt113_timing'))/f'{component}_{version}_{tiling}_aux{max_aux_streams}_l2{l2_limit_for_tiling}.cache'
    path.parent.mkdir(parents=True,exist_ok=True)
    existing=path.read_bytes() if path.exists() else b''
    metadata=dict(cache=str(path),existing_bytes=len(existing),seed=None,
                  ignore_mismatch=False)
    payload=existing
    if not payload and seed_timing_cache:
        baseline=path.parent/f'{component}_{version}_moderate_aux0_l2-1.cache'
        source=(Path(seed_timing_cache) if isinstance(seed_timing_cache,(str,Path)) else baseline)
        if source.resolve()!=path.resolve() and source.is_file() and source.stat().st_size:
            payload=source.read_bytes()
            metadata['seed']=dict(path=str(source.resolve()),sha256=hashlib.sha256(payload).hexdigest(),
                                  bytes=len(payload),status='pending_compatibility_check')
    try:
        cache=config.create_timing_cache(payload)
        compatible=config.set_timing_cache(cache,False)
    except Exception as error:
        if metadata['seed'] is None:raise
        compatible=False
        metadata['seed']['error']=f'{type(error).__name__}: {error}'
    if not compatible:
        if metadata['seed'] is None:
            raise RuntimeError(f'Incompatible TensorRT timing cache: {path}')
        metadata['seed']['status']='rejected_incompatible'
        cache=config.create_timing_cache(b'')
        if not config.set_timing_cache(cache,False):
            raise RuntimeError(f'Cannot initialize empty TensorRT timing cache: {path}')
    elif metadata['seed'] is not None:
        metadata['seed']['status']='accepted'
    _CACHE_PREPARATIONS[str(path.resolve())]=metadata
    return path,cache


def save(config,path):
    cache=config.get_timing_cache()
    if cache is not None:
        Path(path).write_bytes(bytes(cache.serialize()))
