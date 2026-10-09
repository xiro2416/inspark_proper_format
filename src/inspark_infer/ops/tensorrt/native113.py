"""Current static TensorRT CFM/Vocoder wrappers and IO/identity checks."""
from __future__ import annotations
from inspark_infer.runtime.bundle_paths import read_json

import hashlib
import os
import sys
from copy import deepcopy
from pathlib import Path

import torch
import triton
import triton.language as tl










def _import_trt113():
    expected=os.environ.get('ACC_TRT_VERSION_PREFIX','11.3')
    loaded = sys.modules.get("tensorrt")
    if loaded is not None:
        version = getattr(loaded, "__version__", "")
        if not version.startswith(expected):
            raise RuntimeError(f"TensorRT {version} already loaded; expected {expected}")
        return loaded
    site = os.environ.get("ACC_TRT_SITE") or os.environ.get("ACC_TRT113_SITE")
    if not site:
        raise RuntimeError("ACC_TRT_SITE or ACC_TRT113_SITE is required for the isolated TensorRT backend")
    site = str(Path(site).resolve())
    sys.path.insert(0, site)
    try:
        import tensorrt as trt
    finally:
        sys.path.remove(site)
    if not trt.__version__.startswith(expected):
        raise RuntimeError(f"Expected TensorRT {expected}, got {trt.__version__}")
    return trt












def _plan_engine_identity(plan_file, plan, raw_batch, raw_path):
    path = Path(raw_path)
    path = path if path.is_absolute() else (plan_file.parent / path).resolve()
    with path.open('rb') as handle:
        digest = hashlib.file_digest(handle, 'sha256').hexdigest()
    expected = plan.get('engine_sha256', {}).get(str(raw_batch))
    if expected is not None and digest != expected:
        raise ValueError(f'TensorRT engine hash mismatch: {path}')
    provenance = plan.get('provenance', {}).get(str(raw_batch))
    return path, dict(path=str(path), sha256=digest, plan_hash_verified=expected is not None,
                      provenance_status='recorded_not_audited' if provenance else 'legacy_unverified',
                      numerical_audit_pass=False)




def _validate_engine_io(engine, trt, expected, component, allow_layer_outputs=False):
    """Validate the physical binding contract before allocating/enqueuing buffers."""
    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    extra=set(names)-set(expected)
    if (not set(expected)<=set(names) or
            (extra and not allow_layer_outputs) or
            any(not name.startswith('layer') or engine.get_tensor_mode(name)!=trt.TensorIOMode.OUTPUT
                for name in extra)):
        raise ValueError(f"TensorRT {component} I/O names mismatch: {names}; expected {list(expected)}")
    for name, (shape, dtype, mode) in expected.items():
        actual = (tuple(engine.get_tensor_shape(name)), engine.get_tensor_dtype(name),
                  engine.get_tensor_mode(name))
        if actual != (shape, dtype, mode):
            raise ValueError(f"TensorRT {component} I/O mismatch for {name}: {actual}; "
                             f"expected {(shape, dtype, mode)}")
        if engine.get_tensor_location(name) != trt.TensorLocation.DEVICE:
            raise ValueError(f"TensorRT {component} I/O {name} must use device memory")
        if engine.get_tensor_format(name) != trt.TensorFormat.LINEAR:
            raise ValueError(f"TensorRT {component} I/O {name} must use contiguous LINEAR format")




def _validate_acoustic_io(engine, trt, expected):
    return _validate_engine_io(engine, trt, expected, 'acoustic')


def _acoustic_provenance_metadata(plan_file, plan, component, engine_sha256):
    """Freeze build declarations, without pretending to check loader weights.

    No model files are opened here. Full role-by-role checkpoint verification is
    an offline audit responsibility, separate from deserialization and routing.
    """
    declared = plan.get("provenance")
    recorded = (isinstance(declared, dict) and declared.get("status") == "recorded_not_audited"
                and declared.get("schema") == 1 and declared.get("component") == component)
    return dict(status="recorded_not_audited" if recorded else "legacy_unverified",
                declared_status=declared.get("status") if isinstance(declared, dict) else None,
                declared_plan_status=plan.get("provenance_status"),
                plan_sha256=hashlib.sha256(Path(plan_file).read_bytes()).hexdigest(),
                engine_sha256=engine_sha256, engine_hash_verified=plan.get("sha256") == engine_sha256,
                weight_identity_verified=False, loader_weights_checked=False,
                verification_scope="build declaration only; offline audit must compare actual loader files",
                build_provenance=deepcopy(declared) if isinstance(declared, dict) else None)


def _acoustic_route(wrapper, args):
    """Read tensor metadata only: safe during capture and without a CUDA context."""
    reason = None
    for (name, shape, dtype), value in zip(wrapper.input_signature, args):
        if not isinstance(value, torch.Tensor):
            reason = f"{name}.type"
        elif tuple(value.shape) != shape:
            reason = f"{name}.shape"
        elif value.dtype != dtype:
            reason = f"{name}.dtype"
        elif value.device != wrapper.device or value.device.type != "cuda":
            reason = f"{name}.device"
        elif value.layout != torch.strided or not value.is_contiguous():
            reason = f"{name}.layout"
        if reason is not None:
            break
    if len(args) != len(wrapper.input_signature):
        reason = "argument_count"
    first = args[0] if args else None
    shape = tuple(first.shape) if isinstance(first, torch.Tensor) else ()
    route=dict(kind="tensorrt" if reason is None else "eager",
                backend="tensorrt113" if reason is None else "eager",
                candidate_backend="tensorrt113", component=wrapper.component,
                batch=shape[0] if shape else None, frames=shape[-1] if len(shape) == 3 else None,
                engine_batch=wrapper.batch, engine_frames=wrapper.frames,
                plan=wrapper.plan, sha256=wrapper.engine_sha256, reason=reason,
                provenance_status=wrapper.provenance["status"],
                plan_sha256=wrapper.provenance["plan_sha256"],
                weight_identity_verified=False, loader_weights_checked=False,
                plugins=list(wrapper.plugins))
    if wrapper.component=='cfm':
        route.update(solver_kind='estimator_four_enqueues' if wrapper.estimator_only else 'full_solver_one_enqueue',
                     optimization_level=wrapper.optimization_level,
                     tiling_optimization_level=wrapper.tiling_optimization_level)
    return route


class NativeCFMSolver113:
    """Static F310/P258 four-step CFM on the shared TRT 11.3 runtime."""

    def __init__(self, plan_path, eager):
        import json
        from inspark_infer.runtime.graph_policy import BATCHES
        plan_file=Path(plan_path).resolve();plan=read_json(plan_file)
        if (plan.get("format") != 1 or type(plan.get("batch")) is not int or
                plan["batch"] not in BATCHES or plan.get("frames") != 310 or
                plan.get("prompt_frames") != 258):
            raise ValueError("Expected static TensorRT 11.3 F310/P258 CFM plan")
        self.estimator_only=plan.get('kind')=='estimator'
        self.optimization_level=plan.get('optimization_level')
        self.tiling_optimization_level=plan.get('tiling_optimization_level')
        self.batch=plan["batch"];self.frames=310;self.component="cfm";self.plugins=list(plan.get('plugins',[]))
        self.plan=str(plan_file)
        engine_path=Path(plan["engine"])
        if not engine_path.is_absolute():engine_path=(plan_file.parent/engine_path).resolve()
        serialized=engine_path.read_bytes();self.engine_sha256=hashlib.sha256(serialized).hexdigest()
        if self.engine_sha256!=plan.get("sha256"):
            raise ValueError("TensorRT 11.3 CFM engine hash mismatch")
        self.provenance=_acoustic_provenance_metadata(plan_file,plan,"cfm",self.engine_sha256)
        self.trt=_import_trt113();self.runtime=self.trt.Runtime(self.trt.Logger(self.trt.Logger.ERROR))
        if self.plugins:
            if self.plugins not in (['inspark_custom::fp8_quantize','inspark_custom::fp8_gated_up'],
                                    ['inspark_custom::fp8_quantize','inspark_custom::fp8_gated_up_interleaved']):
                raise ValueError('Unsupported CFM plugin inventory')
            from inspark_infer.ops.tensorrt.cfm_gated_up_plugin import register
            register()
        if plan.get('trt')!=self.trt.__version__:
            raise ValueError('CFM TensorRT plan/runtime version mismatch')
        self.engine=self.runtime.deserialize_cuda_engine(serialized)
        if self.engine is None:raise RuntimeError(f"Failed to deserialize {engine_path}")
        b=self.batch;trt=self.trt
        self.input_signature=(("x",(b,80,310),torch.float32),
                              ("prompt",(b,80,310),torch.float32),
                              ("lengths",(b,),torch.int64),
                              ("style",(b,192),torch.float32),
                              ("mu",(b,310,512),torch.float32),
                              ("mask",(b,1,310),torch.bool))
        dtypes={torch.float32:trt.float32,torch.int64:trt.int64,torch.bool:trt.bool}
        if self.estimator_only:
            expected={name:(shape,dtypes[dtype],trt.TensorIOMode.INPUT)
                      for name,shape,dtype in self.input_signature if name!='mask'}
            expected['times']=((b,2),trt.float32,trt.TensorIOMode.INPUT)
            expected['velocity']=((b,80,310),trt.float32,trt.TensorIOMode.OUTPUT)
        else:
            expected={name:(shape,dtypes[dtype],trt.TensorIOMode.INPUT)
                      for name,shape,dtype in self.input_signature}
            expected["output"]=((b,80,310),trt.float32,trt.TensorIOMode.OUTPUT)
        _validate_acoustic_io(self.engine,trt,expected)
        self.context=self.engine.create_execution_context();self.eager=eager
        if self.context is None:raise RuntimeError("Failed to create TensorRT CFM execution context")
        self.model=eager.model;self.times=eager.times;self.identity=dict(eager.identity)
        self.times_batched=(tuple(value.expand(b,-1).contiguous() for value in self.times)
                            if self.estimator_only else ())
        self.identity.update(backend="TensorRT 11.3 native",batch=b,frames=310,
                             prompt_frames=258,plan=self.plan,engine_sha256=self.engine_sha256,
                             provenance_status=self.provenance["status"],
                             plan_sha256=self.provenance["plan_sha256"],
                             precision=plan.get('precision','BF16 learned matrices; FP32 interfaces and solver accumulation'),
                             interface_precision='FP32')
        self.observer=None;self.calls=0;self.estimator_enqueues=0;self.fallbacks=0;self.fallback_reasons={}
        self.device=next(eager.model.parameters()).device
        if self.device.type != "cuda":raise ValueError("TensorRT CFM model must reside on CUDA")
        self.output=torch.empty(b,80,310,device=self.device,dtype=torch.float32)
        self.velocity=torch.empty_like(self.output) if self.estimator_only else None

    def route_for_signature(self,*args):
        return _acoustic_route(self,args)

    describe_route=route_for_signature

    def __call__(self,x,prompt,lengths,style,mu,mask):
        route=self.route_for_signature(x,prompt,lengths,style,mu,mask)
        if route["kind"] != "tensorrt":
            self.fallbacks+=1
            reason=route["reason"];self.fallback_reasons[reason]=self.fallback_reasons.get(reason,0)+1
            return self.eager(x,prompt,lengths,style,mu,mask)
        common={"prompt":prompt.data_ptr(),"lengths":lengths.data_ptr(),
                "style":style.data_ptr(),"mu":mu.data_ptr()}
        if self.estimator_only:
            current=x.float().masked_fill(mask,0)
            for times in self.times_batched:
                addresses=dict(common,x=current.data_ptr(),times=times.data_ptr(),
                               velocity=self.velocity.data_ptr())
                for name,address in addresses.items():
                    if not self.context.set_tensor_address(name,address):
                        raise RuntimeError(f"TensorRT refused CFM estimator binding {name}")
                if not self.context.execute_async_v3(torch.cuda.current_stream(x.device).cuda_stream):
                    raise RuntimeError("TensorRT CFM estimator enqueue failed")
                self.estimator_enqueues+=1
                current=(current+.25*self.velocity).masked_fill(mask,0)
            self.output.copy_(current)
        else:
            addresses=dict(common,x=x.data_ptr(),mask=mask.data_ptr(),output=self.output.data_ptr())
            for name,address in addresses.items():
                if not self.context.set_tensor_address(name,address):
                    raise RuntimeError(f"TensorRT refused CFM binding {name}")
            if not self.context.execute_async_v3(torch.cuda.current_stream(x.device).cuda_stream):
                raise RuntimeError("TensorRT full CFM enqueue failed")
        self.calls+=1
        return self.output

    def stats(self):
        return dict(backend="TensorRT 11.3 four-step CFM estimator" if self.estimator_only else "TensorRT 11.3 native full four-step CFM Solver",batch=self.batch,frames=310,
                    optimization_level=self.optimization_level,tiling_optimization_level=self.tiling_optimization_level,
                    prompt_frames=258,calls=self.calls,fallbacks=self.fallbacks,
                    estimator_enqueues=self.estimator_enqueues,
                    fallback_reasons=dict(self.fallback_reasons),identity=dict(self.identity),
                    provenance=deepcopy(self.provenance))




class NativeVocoder113:
    """Static F52 BigVGAN; standard graphs need no project plugin registration."""

    def __init__(self, plan_path, eager):
        import json
        plan_file=Path(plan_path).resolve();plan=read_json(plan_file)
        if (plan.get("format") != 1 or type(plan.get("batch")) is not int or
                plan["batch"] not in (1, 4, 8, 16, 24, 32, 64, 128) or plan.get("frames") != 52):
            raise ValueError("Expected static TensorRT 11.3 B1/B4/B8/B16/B24/B64, F52 Vocoder plan")
        self.batch=plan["batch"];self.frames=52;self.component="vocoder"
        self.plugins=plan.get("plugins",["Quick Plugins (inventory unspecified)"])
        if not isinstance(self.plugins,list) or any(not isinstance(p,str) for p in self.plugins):
            raise ValueError("TensorRT Vocoder plugins must be a list of names")
        self.plan=str(plan_file)
        self.precision=plan.get('precision','BF16 learned convolutions; FP32 alias-free activation and interfaces')
        engine_path=Path(plan["engine"])
        if not engine_path.is_absolute():engine_path=(plan_file.parent/engine_path).resolve()
        serialized=engine_path.read_bytes();self.engine_sha256=hashlib.sha256(serialized).hexdigest()
        if self.engine_sha256!=plan.get("sha256"):
            raise ValueError("TensorRT 11.3 Vocoder engine hash mismatch")
        self.provenance=_acoustic_provenance_metadata(plan_file,plan,"vocoder",self.engine_sha256)
        self.trt=_import_trt113()
        if self.plugins:
            if self.plugins==['nvidia_bigvgan::alias_free']:
                from inspark_infer.ops.tensorrt.official_bigvgan_plugin import register
                vendor=plan.get('vendor_plugin_provenance',{})
                if not vendor or vendor.get('math_source_modified',True):
                    raise ValueError('Missing unchanged NVIDIA plugin provenance')
                if hashlib.sha256(Path(vendor['module']).read_bytes()).hexdigest()!=vendor.get('module_sha256'):
                    raise ValueError('NVIDIA activation binary identity changed')
            elif set(self.plugins)<= {'inspark_custom::small_fir_activation','inspark_custom::small_fir_activation_tiled'}:
                from inspark_infer.ops.tensorrt.vocoder_small_fir_plugin import register
                # Legacy flag means custom arithmetic is present, including reuse.
                self.new_gpu_math=True
            else:
                from inspark_infer.ops.tensorrt.vocoder_plugin import register
            register()
        if plan.get('trt')!=self.trt.__version__:
            raise ValueError('Vocoder TensorRT plan/runtime version mismatch')
        self.runtime=self.trt.Runtime(self.trt.Logger(self.trt.Logger.ERROR))
        self.engine=self.runtime.deserialize_cuda_engine(serialized)
        if self.engine is None:raise RuntimeError(f"Failed to deserialize {engine_path}")
        b=self.batch;trt=self.trt
        self.input_signature=(("mel",(b,80,52),torch.float32),)
        expected={"mel":((b,80,52),trt.float32,trt.TensorIOMode.INPUT),
                  "pcm":((b,1,13312),trt.float32,trt.TensorIOMode.OUTPUT)}
        _validate_acoustic_io(self.engine,trt,expected)
        self.context=self.engine.create_execution_context();self.eager=eager
        if self.context is None:raise RuntimeError("Failed to create TensorRT Vocoder execution context")
        self.calls=0;self.fallbacks=0;self.fallback_reasons={}
        # The serving entrypoint is usually BigVGAN.forward, a bound method.
        from itertools import chain
        model=getattr(eager,"__self__",eager)
        sample=next(chain(model.parameters(),model.buffers()),None)
        if sample is None or sample.device.type != "cuda":
            raise ValueError("TensorRT Vocoder reference model must reside on CUDA")
        self.device=sample.device
        self.output=torch.empty(b,1,13312,device=self.device,dtype=torch.float32)

    def route_for_signature(self,*args):
        return _acoustic_route(self,args)

    describe_route=route_for_signature

    def __call__(self,mel):
        route=self.route_for_signature(mel)
        if route["kind"] != "tensorrt":
            self.fallbacks+=1
            reason=route["reason"];self.fallback_reasons[reason]=self.fallback_reasons.get(reason,0)+1
            return self.eager(mel)
        for name,address in (("mel",mel.data_ptr()),("pcm",self.output.data_ptr())):
            if not self.context.set_tensor_address(name,address):
                raise RuntimeError(f"TensorRT refused Vocoder binding {name}")
        if not self.context.execute_async_v3(torch.cuda.current_stream(mel.device).cuda_stream):
            raise RuntimeError("TensorRT full Vocoder enqueue failed")
        self.calls+=1;return self.output

    def stats(self):
        return dict(backend=("TensorRT 11.3 BigVGAN with in-engine Quick Plugins" if self.plugins else
                             "TensorRT 11.3 BigVGAN standard ONNX operators"),
                    batch=self.batch,frames=52,calls=self.calls,fallbacks=self.fallbacks,
                    fallback_reasons=dict(self.fallback_reasons),plan=self.plan,
                    sha256=self.engine_sha256,plugins=list(self.plugins),
                    precision=self.precision,interface_precision='FP32',
                    provenance=deepcopy(self.provenance))
