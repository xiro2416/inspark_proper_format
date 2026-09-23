"""One backend-selection boundary for the existing IndexTTS2 inference flow."""

REFERENCE_BACKENDS = ("eager", "compile")
COMPONENTS = ("target", "draft", "cfm", "vocoder")


def validate_reference_plan(plan):
    backend = plan.get("compute_backend", "optimized")
    if backend not in ("optimized", *REFERENCE_BACKENDS):
        raise ValueError("compute_backend must be eager, compile or optimized")
    if backend not in REFERENCE_BACKENDS:
        if any(k.startswith("compile_") for k in plan):
            raise ValueError("compile options require compute_backend=compile")
        return
    if plan["precision"] not in ("fp32", "bf16") or plan["rnn_precision"] not in ("fp32", "bf16"):
        raise ValueError("Pure PyTorch reference supports FP32/BF16, not FP8/auto")
    allowed = {"schema", "status", "precision", "components", "convolutions", "rnn_precision",
               "compute_backend", "strict_request_isolation", "compile_components",
               "compile_max_signatures", "compile_error_fallback", "compile_pattern_matcher"}
    for key, value in plan.items():
        if key not in allowed and value not in (False, None, [], "off"):
            raise ValueError(f"{backend} forbids optimized option {key}={value!r}")
    parts = plan.get("compile_components", list(COMPONENTS))
    if not isinstance(parts, list) or not parts or len(parts) != len(set(parts)) or not set(parts) <= set(COMPONENTS):
        raise ValueError("compile_components must be a nonempty unique component subset")
    count = plan.get("compile_max_signatures", 1)
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 8:
        raise ValueError("compile_max_signatures must be in 1..8")
    if not isinstance(plan.get("compile_error_fallback", False), bool):
        raise ValueError("compile_error_fallback must be boolean")
    if not isinstance(plan.get("compile_pattern_matcher", False), bool):
        raise ValueError("compile_pattern_matcher must be boolean")
    if backend == "eager" and any(k.startswith("compile_") for k in plan):
        raise ValueError("eager does not accept compile options")


def prepare_reference(engine, plan, result):
    """Configure arithmetic only; keep raw PyTorch AR/cache/sampling/acoustics."""
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if engine.config["target_tf32"]:
        raise ValueError("Reference requires target_tf32=false before model/reference loading")
    result.update(compute_backend=plan["compute_backend"], custom_kernels=False,
                  tf32=False, same_model_structure=True, shared_batch_rng=False)
    if plan["precision"] != "fp32":
        result["precision"] = engine.prepare_precision(plan["precision"], plan["components"], plan["convolutions"])
    if plan["rnn_precision"] != "fp32":
        result["rnn"] = engine.prepare_rnn_precision(plan["rnn_precision"])
    if plan["compute_backend"] == "compile":
        import triton
        # Pinned, measured stack. The isolated Gluon 3.5 toolchain is not the
        # Triton version bundled with PyTorch 2.8's Inductor.
        if not torch.__version__.startswith("2.8.") or not triton.__version__.startswith("3.4."):
            raise RuntimeError("Compile baseline requires torch 2.8 + bundled Triton 3.4; use ACC_TRITON_TOOLCHAIN=default")
        from inspark_infer.ops.compile import CompiledOp, CompileBank
        boundaries = {"target": (engine.rt.target, "body"),
                      "draft": (engine.rt.backbone, "body"),
                      "cfm": (engine.student.model, "forward"),
                      "vocoder": (engine, "vocoder")}
        operators = {}
        for name in plan.get("compile_components", list(COMPONENTS)):
            owner, attr = boundaries[name]
            eager = getattr(owner, attr)
            if name == "target":
                from inspark_infer.ops.eager.target import TargetCacheAdapter
                eager = TargetCacheAdapter(eager)
            op = CompiledOp(eager, name,
                max_signatures=plan.get("compile_max_signatures", 1),
                error_fallback=plan.get("compile_error_fallback", False),
                pattern_matcher=plan.get("compile_pattern_matcher", False))
            setattr(owner, attr, op)
            operators[name] = op
        engine.compile_bank = CompileBank(operators)
        result["compile"] = dict(components=list(operators), offline_warmup_required=True,
                                 online_compile=False, unknown_shape="eager", fullgraph=True,
                                 dynamic=False, cudagraphs=False, emulate_precision_casts=True,
                                 pattern_matcher=plan.get("compile_pattern_matcher", False),
                                 torch=torch.__version__, triton=triton.__version__)
    if hasattr(engine, "validate_request_isolation"):
        result["request_isolation"] = engine.validate_request_isolation(strict=True)
    engine.deployment_state = "ready"
    engine.deployment_manifest = result
    return result
