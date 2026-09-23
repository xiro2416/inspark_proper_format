"""Explicit offline compilation of existing PyTorch callables.

No serving-time compilation, CUDA graph capture, custom kernels, or model copies.
Unknown shapes use the unchanged eager callable. Compiler guard failures are not
silently ignored: the caller chooses an eager fallback or an exception.
"""
from contextlib import contextmanager
import time


def signature(value):
    import torch
    if isinstance(value, torch.Tensor):
        return ("tensor", tuple(value.shape), tuple(value.stride()), str(value.dtype),
                str(value.device), bool(value.requires_grad))
    if isinstance(value, (tuple, list)):
        return (type(value).__name__, tuple(signature(v) for v in value))
    if isinstance(value, dict):
        return ("dict", tuple((k, signature(v)) for k, v in sorted(value.items())))
    if value is None or isinstance(value, (str, bool, int, float)):
        return (type(value).__name__, value)
    raise TypeError(f"Unsupported compile boundary value: {type(value).__name__}")


class CompiledOp:
    def __init__(self, eager, name, *, max_signatures=1, error_fallback=False,
                 compiler=None, pattern_matcher=False):
        if max_signatures < 1:
            raise ValueError("max_signatures must be positive")
        self.eager, self.name = eager, name
        self.max_signatures, self.error_fallback = max_signatures, error_fallback
        self.compiled = None
        self.compiler = compiler
        self.pattern_matcher = bool(pattern_matcher)
        self.warming = False
        self.known = set()
        self.failed = set()
        self.compiled_calls = self.eager_calls = 0
        self.compile_seconds = 0.0
        self.audit_seconds = 0.0
        self.errors = []
        self.audit_hook = None

    def __call__(self, *args, **kwargs):
        import torch
        key = signature((args, kwargs))
        if key in self.known:
            # Guard changes must not trigger hidden serving-time compilation.
            try:
                with torch.compiler.set_stance("fail_on_recompile"):
                    result = self.compiled(*args, **kwargs)
                self.compiled_calls += 1
                return result
            except Exception as exc:
                if not self.error_fallback:
                    raise
                self.errors.append(dict(kind="guard_or_execution", error=repr(exc)))
                self.known.remove(key)
                self.failed.add(key)
        elif self.warming and key not in self.failed and len(self.known)+len(self.failed) < self.max_signatures:
            start = time.perf_counter()
            compile_end = None
            try:
                if self.compiled is None:
                    compile_fn = self.compiler or torch.compile
                    self.compiled = compile_fn(self.eager, backend="inductor", fullgraph=True,
                        dynamic=False, options={"triton.cudagraphs": False,
                                              "emulate_precision_casts": True,
                                              "pattern_matcher": self.pattern_matcher})
                result = self.compiled(*args, **kwargs)
                if torch.cuda.is_available():
                    torch.cuda.current_stream().synchronize()
                compile_end = time.perf_counter()
                if self.audit_hook is not None:
                    self.audit_hook(self, args, kwargs, result)
                    self.audit_seconds += time.perf_counter()-compile_end
                self.known.add(key)
                self.compiled_calls += 1
                return result
            except Exception as exc:
                self.failed.add(key)
                self.errors.append(dict(kind="offline_compile", error=repr(exc)))
                if not self.error_fallback:
                    raise
            finally:
                self.compile_seconds += (compile_end or time.perf_counter()) - start
        self.eager_calls += 1
        return self.eager(*args, **kwargs)

    def stats(self):
        return dict(component=self.name, compiled_calls=self.compiled_calls,
                    eager_calls=self.eager_calls, warmed_signatures=len(self.known),
                    signatures=[repr(k) for k in sorted(self.known, key=repr)],
                    failed_signatures=len(self.failed), errors=self.errors,
                    compile_seconds=self.compile_seconds, audit_seconds=self.audit_seconds,
                    emulate_precision_casts=True, online_compile=False,
                    pattern_matcher=self.pattern_matcher,
                    unknown_shape="eager", compiler_error="eager" if self.error_fallback else "raise")


class CompileBank:
    def __init__(self, operators):
        self.operators = operators

    @contextmanager
    def offline_warmup(self):
        if any(op.warming for op in self.operators.values()):
            raise RuntimeError("Nested compiler warmup is not supported")
        for op in self.operators.values():
            op.warming = True
        try:
            yield
        finally:
            for op in self.operators.values():
                op.warming = False

    def stats(self):
        return {name: op.stats() for name, op in self.operators.items()}
