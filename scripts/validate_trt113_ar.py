#!/usr/bin/env python3
"""Same-input eager/direct/graph audit of existing SM89 Target and Draft engines.

The prefix caches come from a real reference and text prefill. Verification uses
the sampled first anchor followed by seven fixed diagnostic tokens; this is not
a complete sampled speech trajectory or a perceptual quality audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from trt113_validation import TOLERANCES, compare
from trt113_provenance import file_record, source_identity


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def engine_evidence(path, batch, component, current_paths):
    """Verify build declarations against the files actually used by the loader.

    Build metadata paths are descriptive only: never follow them to choose the
    reference weights. Checkpoint and config contents must both match by role.
    """
    path = Path(path).resolve()
    metadata_path = path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
    actual_sources = {role: file_record(source, role) for role, source in current_paths.items()}
    expected_roles = {f"{component}_checkpoint", f"{component}_config"}
    if component not in ("target", "draft") or set(actual_sources) != expected_roles:
        raise ValueError(f"Reference model source roles do not match {component}")
    sha256 = digest(path)
    if metadata.get("batch", batch) != batch:
        raise ValueError(f"Engine metadata batch mismatch: {path}")
    if metadata.get("sha256", sha256) != sha256:
        raise ValueError(f"Engine SHA256 differs from its build metadata: {path}")
    checkpoint_hash = actual_sources[f"{component}_checkpoint"]["sha256"]
    if metadata.get("model_sha256", checkpoint_hash) != checkpoint_hash:
        raise ValueError(f"Engine build checkpoint SHA256 mismatch: {path}")
    provenance = metadata.get("provenance")
    verified = False
    source_checks = []
    constant_hash = None
    if isinstance(provenance, dict) and provenance.get("status") == "recorded_not_audited":
        if provenance.get("schema") != 1 or provenance.get("component") != component:
            raise ValueError(f"Engine build provenance schema/component mismatch: {path}")
        if metadata.get("sha256") != sha256:
            raise ValueError(f"Recorded provenance lacks a matching engine SHA256 binding: {path}")
        declared = provenance.get("model_sources")
        if not isinstance(declared, list) or any(not isinstance(row, dict) for row in declared):
            raise ValueError(f"Recorded provenance has no valid model_sources list: {path}")
        by_role = {row.get("role"): row for row in declared}
        if len(by_role) != len(declared) or set(by_role) != expected_roles:
            raise ValueError(f"Recorded model source roles are missing, duplicated or unexpected: {path}")
        for role, current in actual_sources.items():
            build = by_role[role]
            if build.get("sha256") != current["sha256"]:
                raise ValueError(f"Engine build model source SHA256 mismatch for {role}: {path}")
            source_checks.append(dict(role=role, current=current, build=build, sha256_matches=True,
                                      path_matches=build.get("path") == current["path"]))
        constant_hash = provenance.get("constant_data_sha256")
        if (not isinstance(constant_hash, str) or len(constant_hash) != 64
                or any(character not in "0123456789abcdef" for character in constant_hash)):
            raise ValueError(f"Recorded provenance lacks a valid constant_data_sha256: {path}")
        verified = True
    return dict(path=str(path), sha256=sha256,
                build_metadata_path=str(metadata_path) if metadata_path.is_file() else None,
                build_metadata_sha256=digest(metadata_path) if metadata_path.is_file() else None,
                build_metadata=metadata, reference_checkpoint_sha256=checkpoint_hash,
                actual_model_sources=list(actual_sources.values()), model_source_checks=source_checks,
                constant_data_sha256=constant_hash, constant_data_independently_verified=False,
                weight_identity_verified=verified,
                weight_identity_note=("Actual loader checkpoint/config hashes match every build model_sources role; engine hash binding verified. Constant hash is a build attestation, not independently extracted from the engine."
                                      if verified else
                                      "Legacy metadata lacks complete hash-bound model-source provenance; numerical results do not attest engine weight provenance"))


def uniform_length(values, name):
    if not values or len(set(values)) != 1:
        raise ValueError(f"{name} requires nonempty, equal prefix lengths for this diagnostic")
    length = int(values[0])
    if length <= 0 or length + 8 > 128:
        raise ValueError(f"{name} prefix length {length} must be in [1,120] for K128 plus eight positions")
    return length


def replay_inputs(engine, rows):
    """Copy real prefill state without retaining request-owned cache views."""
    import torch

    target_length = uniform_length([row.past_length for row in rows], "Target")
    draft_length = uniform_length([row.cache.length for row in rows], "Draft")
    target_cache = torch.stack([
        torch.stack([torch.cat([row.kv[layer][bank] for row in rows], dim=0)
                     for bank in range(2)])
        for layer in range(len(rows[0].kv))
    ]).clone()
    draft_cache = torch.stack([
        torch.stack([torch.cat([getattr(row.cache, name)[layer] for row in rows], dim=0)
                     for name in ("keys", "values")])
        for layer in range(len(rows[0].cache.keys))
    ]).clone()
    if target_cache.shape[-2] != target_length or draft_cache.shape[-2] != draft_length:
        raise ValueError("Visible cache extents differ from recorded prefix lengths")
    if target_cache.dtype != torch.float32 or draft_cache.dtype != torch.float32:
        raise ValueError("Raw reference prefill must produce FP32 Target and Draft cache snapshots")
    anchors = torch.cat([row.codes[-1] for row in rows]).long().clone()
    device, batch = anchors.device, len(rows)
    first = torch.tensor([row.past_length + 1 - row.mel_length for row in rows], device=device)
    target_positions = first[:, None] + torch.arange(8, device=device)[None]
    draft_positions = first[:, None] + torch.arange(7, device=device)[None]
    fixed = torch.arange(1, 8, device=device).expand(batch, -1)
    tokens = torch.cat((anchors[:, None], fixed), dim=1)
    target_model = engine.rt.engine.target.model
    target_x = (target_model.embeddings(tokens)
                + target_model.text_pos_embedding.emb(target_positions)).clone()
    prefix_mask = torch.cat([row.mask for row in rows], dim=0).clone()
    if prefix_mask.shape != (batch, target_length):
        raise ValueError("Target prefix mask shape mismatch")
    target_mask = torch.cat((prefix_mask, prefix_mask.new_ones(batch, 8)), dim=1)
    keep = torch.zeros(batch, 128, device=device, dtype=torch.int32)
    keep[:, :target_length + 8] = target_mask.to(torch.int32)
    return dict(target_cache=target_cache, draft_cache=draft_cache, anchors=anchors,
                tokens=tokens, target_x=target_x, target_mask=target_mask, keep=keep,
                draft_positions=draft_positions, target_length=target_length,
                draft_length=draft_length, slots=torch.arange(batch, device=device, dtype=torch.int32),
                target_lengths=torch.full((batch,), target_length, device=device, dtype=torch.int32),
                draft_lengths=torch.full((batch,), draft_length, device=device, dtype=torch.int32))


def target_reference(target, inputs, cache):
    import torch

    past = tuple((cache[layer, 0], cache[layer, 1]) for layer in range(cache.shape[0]))
    logits, present, selected, final = target._block_forward_with_hidden_states(
        inputs["target_x"], past, inputs["target_mask"], None)
    new_cache = torch.stack([torch.stack((key[:, :, -8:], value[:, :, -8:]))
                             for key, value in present])
    return dict(logits=logits, selected=selected, final=final, new_kv=new_cache)


def draft_reference(backbone, inputs):
    import torch

    cache = inputs["draft_cache"]
    # Pad to the same K128 context arena as native; padding is masked exactly.
    padded = torch.nn.functional.pad(cache, (0, 0, 0, 128 - inputs["draft_length"]))
    keep = (torch.arange(128, device=cache.device)[None]
            < inputs["draft_lengths"][:, None])
    context_positions = inputs["draft_lengths"][:, None] + torch.arange(7, device=cache.device)[None]
    hidden, base = backbone.forward(
        inputs["anchors"], inputs["draft_positions"],
        tuple(padded[layer, 0] for layer in range(padded.shape[0])),
        tuple(padded[layer, 1] for layer in range(padded.shape[0])), keep,
        context_positions if backbone.context_uses_positions else None)
    return dict(hidden=hidden, base=base)


def snapshot(outputs):
    return {name: value.detach().clone() for name, value in outputs.items()}


def compare_outputs(reference, candidate, precision):
    if not reference or set(reference) != set(candidate):
        raise ValueError("Nonempty reference/candidate output names must match")
    rows = {name: compare(reference[name], candidate[name], precision) for name in reference}
    return dict(pass_gate=all(value["pass_gate"] for value in rows.values()), outputs=rows)


def check_native_bindings(native, component, batch):
    """Reject a wrong engine before any enqueue can use fixed-size buffers."""
    trt, engine = native.trt, native.engine
    i, o = trt.TensorIOMode.INPUT, trt.TensorIOMode.OUTPUT
    if component == "target":
        expected = {"x": ((batch, 8, 1280), trt.float32, i),
                    "mask": ((batch, 1, 8, 128), trt.bool, i),
                    "write_indices": ((batch,), trt.int32, i),
                    "logits": ((batch, 8, 8194), trt.float32, o),
                    "selected": ((batch, 8, 6400), trt.float32, o),
                    "final": ((batch, 8, 1280), trt.float32, o)}
        for layer in range(24):
            for letter in ("k", "v"):
                for direction, mode in (("in", i), ("out", o)):
                    expected[f"{letter}_cache_{direction}_{layer}"] = (
                        (batch, 20, 128, 64), trt.bfloat16, mode)
    elif component == "draft":
        expected = {"x": ((batch, 7, 1280), trt.float32, i),
                    "mask": ((batch, 1, 7, 135), trt.bool, i),
                    "hidden": ((batch, 7, 1280), trt.float32, o),
                    "base": ((batch, 7, 8194), trt.float32, o)}
        for layer in range(3):
            for letter in ("k", "v"):
                expected[f"{letter}_cache_{layer}"] = ((batch, 20, 128, 64), trt.float32, i)
    else:
        raise ValueError(f"Unknown component {component}")
    names = [engine.get_tensor_name(index) for index in range(engine.num_io_tensors)]
    if len(names) != len(expected) or set(names) != set(expected):
        raise ValueError(f"{component} engine tensor names differ from the supported contract")
    for name, signature in expected.items():
        actual = (tuple(engine.get_tensor_shape(name)), engine.get_tensor_dtype(name),
                  engine.get_tensor_mode(name))
        if actual != signature:
            raise ValueError(f"{component} engine binding {name}: {actual}; expected {signature}")
        if (engine.get_tensor_location(name) != trt.TensorLocation.DEVICE
                or engine.get_tensor_format(name) != trt.TensorFormat.LINEAR):
            raise ValueError(f"{component} engine binding {name} must be contiguous device memory")


def timed(torch, call, iterations):
    for _ in range(3):
        call()
    torch.cuda.current_stream().synchronize()
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    started = time.perf_counter()
    begin.record()
    for _ in range(iterations):
        call()
    end.record()
    end.synchronize()
    return dict(gpu_ms=begin.elapsed_time(end) / iterations,
                wall_ms=(time.perf_counter() - started) * 1000 / iterations,
                iterations=iterations, warmups=3)


def audit_component(torch, native, direct, capture_call, capture_args, reset,
                    collect, references, iterations, invariants):
    from acc_infer_clear.runtime.graphs import capture

    start_calls = native.calls
    reset()
    direct()
    actual = snapshot(collect())
    direct_calls = native.calls - start_calls
    direct_invariants = invariants()
    before_capture = native.calls
    reset()
    graph = capture(capture_call, capture_args)
    capture_calls = native.calls - before_capture
    reset()
    graph(*capture_args)
    graphed = snapshot(collect())
    graph_invariants = invariants()
    comparisons = {}
    for name, reference in references.items():
        comparisons[f"direct_vs_{name}"] = compare_outputs(reference, actual, "bf16")
        comparisons[f"graph_vs_{name}"] = compare_outputs(reference, graphed, "bf16")
    comparisons["graph_vs_direct"] = compare_outputs(actual, graphed, "fp32")

    def native_timed():
        reset()
        return direct()

    def graph_timed():
        reset()
        return graph(*capture_args)

    timing = {"native": timed(torch, native_timed, iterations),
              "graph": timed(torch, graph_timed, iterations)}
    return dict(comparisons=comparisons, cache_invariants={"direct": direct_invariants,
                    "graph": graph_invariants}, timing=timing,
                execution=dict(direct_validation_calls=direct_calls,
                               capture_wrapper_calls=capture_calls,
                               graph_validation_replays=1,
                               graph_timing_replays=iterations + 3,
                               wrapper_calls_total=native.calls, fallback_supported=False),
                pass_gate=(direct_calls == 1 and capture_calls > 0
                           and all(comparisons[key]["pass_gate"] for key in comparisons)
                           and all(direct_invariants.values()) and all(graph_invariants.values())))


def validate(args, report):
    import torch
    from acc_infer_clear.config import load
    from acc_infer_clear.streaming.engine import Engine
    from acc_infer_clear.tensorrt_backend.native113 import NativeTargetFull113, NativeDraftFull113

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    report["hardware"] = dict(device=torch.cuda.get_device_name(), sm=list(torch.cuda.get_device_capability()),
                              torch=torch.__version__, cuda=torch.version.cuda)
    if tuple(report["hardware"]["sm"]) != (8, 9):
        raise ValueError("Existing engines in this diagnostic are SM89-specific")
    config = load(args.config)
    config.update(max_batch=args.batch, target_tf32=False)
    report["stage"] = "load_model"
    engine = Engine(config)
    try:
        base = Path(config["weights"])
        checkpoints = {"target": Path(engine.tts.gpt_path),
                       "draft": base / "draft_onpolicy100/model.safetensors"}
        report["checkpoints"] = {name: dict(path=str(path), sha256=digest(path))
                                  for name, path in checkpoints.items()}
        model_paths = {
            "target": {"target_checkpoint": checkpoints["target"],
                       "target_config": Path(engine.tts.model_dir) / "config.yaml"},
            "draft": {"draft_checkpoint": checkpoints["draft"],
                      "draft_config": base / "draft_onpolicy100/config.json"},
        }
        report["engines"] = {}
        for name, path in (("target", args.target_engine), ("draft", args.draft_engine)):
            report["engines"][name] = engine_evidence(path, args.batch, name, model_paths[name])
        report["weight_identity_verified"] = all(row["weight_identity_verified"] for row in report["engines"].values())
        report["stage"] = "real_prefill"
        engine.prepare_reference("audit", args.reference)
        identifiers = [f"audit-{index}" for index in range(args.batch)]
        for index, identifier in enumerate(identifiers):
            engine.create_session(identifier, "audit", args.seed + index)
            engine.push_text(identifier, args.text)
            engine.finish_input(identifier)
        with torch.cuda.stream(engine.model.stream), torch.inference_mode():
            rows = engine.prepare_rows([engine.sessions[identifier] for identifier in identifiers], 0)
            inputs = replay_inputs(engine, rows)
            for identifier, row in zip(identifiers, rows):
                engine.sessions[identifier]["_row"] = row
                engine.cancel(identifier)
            del rows
            target, backbone = engine.rt.engine.target, engine.rt.backbone
            report["inputs"] = dict(text=args.text, reference=str(Path(args.reference).resolve()),
                                    reference_sha256=digest(args.reference), seed=args.seed,
                                    target_prefix=inputs["target_length"], draft_prefix=inputs["draft_length"],
                                    diagnostic_tokens=inputs["tokens"].cpu().tolist(),
                                    draft_positions=inputs["draft_positions"].cpu().tolist(),
                                    cache_source="Real FP32 PyTorch frontend/Target/Draft-context prefill; snapshots survive cancellation")
            report["stage"] = "fp32_reference"
            raw_target_cache = inputs["target_cache"].detach().clone()
            matched_target_cache = raw_target_cache.bfloat16().float()
            report["input_precision_boundaries"] = dict(
                target_raw_cache_dtype=str(raw_target_cache.dtype),
                target_native_cache_storage_dtype="torch.bfloat16",
                target_matched_reference_cache_dtype=str(matched_target_cache.dtype),
                target_matched_cache_conversion="raw FP32 -> BF16 rounding -> FP32 reference storage",
                target_embeddings_dtype=str(inputs["target_x"].dtype),
                draft_cache_dtype=str(inputs["draft_cache"].dtype),
                snapshots="Cloned real prefill caches and cloned eager outputs; no request cache views retained")
            fp32_target = snapshot(target_reference(target, inputs, raw_target_cache))
            fp32_matched = snapshot(target_reference(target, inputs, matched_target_cache))
            fp32_draft = snapshot(draft_reference(backbone, inputs))
            report["stage"] = "bf16_reference"
            report["precision_preparation"] = engine.prepare_precision("bf16", ["target", "draft"], False)
            bf16_target = snapshot(target_reference(target, inputs, matched_target_cache))
            bf16_draft = snapshot(draft_reference(backbone, inputs))
            report["precision_comparisons"] = {
                "target_bf16_vs_fp32_raw": compare_outputs(fp32_target, bf16_target, "bf16"),
                "target_fp32_bf16cache_vs_fp32_raw": compare_outputs(fp32_target, fp32_matched, "bf16"),
                "draft_bf16_vs_fp32_raw": compare_outputs(fp32_draft, bf16_draft, "bf16"),
            }
            report["reference"] = dict(
                implementation="Raw Engine; TargetEngine._block_forward_with_hidden_states; BatchedDraftBackbone.forward",
                precision="FP32 outputs saved before BF16-only Target/Draft linear wrappers; FP32 norms/residuals/heads",
                target_cache="Native stores BF16; same-input PyTorch references receive BF16-rounded cache values promoted to FP32",
                target_attention="PyTorch reference attention computes from FP32 Q/K/V tensors carrying BF16-rounded linear/cache values; TensorRT attention has BF16 I/O and engine-selected reduction tactics. This is a reference comparison, not identical arithmetic.",
                draft_cache="Native and all reference arms use identical FP32 K128 padded real-prefix cache",
                prefill_scope="Fixed FP32 reference producer state; this diagnostic does not audit deployed prefill or whole-trajectory cache production",
                tf32=False, deployment_prepared=False, reference_cuda_graphs=False,
                reference_custom_kernels=False, reference_torch_compile=False)
            report["stage"] = "target_native"
            native_target = NativeTargetFull113(args.target_engine, args.batch, args.batch, inputs["target_x"].device)
            check_native_bindings(native_target, "target", args.batch)
            compact = torch.zeros_like(native_target.cache)
            length = inputs["target_length"]
            compact[..., :length, :].copy_(raw_target_cache)

            def reset_target():
                native_target.cache.copy_(compact)

            def collect_target():
                return dict(logits=native_target.logits, selected=native_target.selected,
                            final=native_target.final,
                            new_kv=native_target.cache[..., length:length + 8, :])

            def target_invariants():
                return dict(prefix_unchanged=torch.equal(native_target.cache[..., :length, :], compact[..., :length, :]),
                            unused_tail_unchanged=torch.equal(native_target.cache[..., length + 8:, :], compact[..., length + 8:, :]))

            target_args = (inputs["target_x"], inputs["keep"], inputs["slots"], inputs["target_lengths"])
            report["target"] = audit_component(
                torch, native_target, lambda: native_target.run(*target_args), native_target.run,
                target_args, reset_target, collect_target,
                {"fp32_raw": fp32_target, "fp32_same_cache": fp32_matched,
                 "bf16_same_cache": bf16_target},
                args.iterations, target_invariants)
            report["target"]["timing"]["eager_bf16"] = timed(
                torch, lambda: target_reference(target, inputs, matched_target_cache), args.iterations)
            report["stage"] = "draft_native"
            draft_compact = torch.nn.functional.pad(inputs["draft_cache"], (0, 0, 0, 128 - inputs["draft_length"]))
            native_draft = NativeDraftFull113(args.draft_engine, args.batch, draft_compact.clone(), draft_compact.device)
            check_native_bindings(native_draft, "draft", args.batch)
            draft_x = backbone.model._noise_embeddings(inputs["anchors"], inputs["draft_positions"]).clone()
            draft_args = (draft_x, inputs["slots"], inputs["draft_lengths"])
            report["draft"] = audit_component(
                torch, native_draft, lambda: native_draft.run(*draft_args), native_draft.run,
                draft_args, lambda: native_draft.cache.copy_(draft_compact),
                lambda: dict(hidden=native_draft.hidden, base=native_draft.base),
                {"fp32": fp32_draft, "bf16": bf16_draft}, args.iterations,
                lambda: dict(context_unchanged=torch.equal(native_draft.cache, draft_compact)))
            report["draft"]["timing"]["eager_bf16"] = timed(torch, lambda: draft_reference(backbone, inputs), args.iterations)
            report["hardware"]["tensorrt"] = native_target.trt.__version__
            report["timing_scope"] = ("Diagnostic component calls; candidate timing includes cache reset and graph input copies; "
                                      "eager includes its native allocations/padding; excludes model load, reference prefill, engine creation, "
                                      "capture, report metrics and D2H snapshots. Not end-to-end service performance.")
            report["pass_gate"] = (report["target"]["pass_gate"] and report["draft"]["pass_gate"]
                                   and all(row["pass_gate"] for row in report["precision_comparisons"].values()))
            report["stage"] = "complete"
    finally:
        engine.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=6)
    parser.add_argument("--batch", type=int, choices=(1, 4, 8), required=True)
    parser.add_argument("--config", default=str(ROOT / "configs/runtime.yaml"))
    parser.add_argument("--reference", default="/workspace/index-tts/data/audio/old/mingxiang_gao.wav")
    parser.add_argument("--target-engine", type=Path)
    parser.add_argument("--draft-engine", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=113)
    parser.add_argument("--text", default="他正在整理文件。")
    args = parser.parse_args(argv)
    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    args.target_engine = args.target_engine or ROOT / f"artifacts/trt113_target_full/target_full_b{args.batch}.engine"
    args.draft_engine = args.draft_engine or ROOT / f"artifacts/trt113_draft_full/draft_full_b{args.batch}.engine"
    return args


def main(argv=None):
    args = parse_args(argv)
    report = dict(schema=1, batch=args.batch, physical_gpu=args.gpu, stage="preflight", pass_gate=False,
                  scope="SM89 same-input diagnostic component audit, not whole-trajectory equivalence or audio quality",
                  pass_gate_scope="Numerical and cache checks only; engine weight provenance reported separately",
                  tolerances=TOLERANCES, command_arguments={key: str(value) if isinstance(value, Path) else value
                                                           for key, value in vars(args).items()},
                  script_sha256=digest(__file__), metrics_sha256=digest(ROOT / "scripts/trt113_validation.py"))
    error = None
    try:
        report["source"] = source_identity(ROOT)
        report["source"]["snapshot"] = "local source files before AR audit, including current scripts and untracked source files"
        report["provenance_helper_sha256"] = digest(ROOT / "scripts/trt113_provenance.py")
        from acc_infer_clear.runtime.device import GPULease, select_gpu
        with GPULease(args.gpu):
            select_gpu(args.gpu)
            validate(args, report)
    except Exception as exc:
        report["pass_gate"] = False
        report["error"] = dict(type=type(exc).__name__, message=str(exc))
        error = exc
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        print(json.dumps(dict(output=str(args.output), stage=report["stage"], pass_gate=report["pass_gate"],
                              error=report.get("error")), ensure_ascii=False), flush=True)
    if error is not None:
        raise RuntimeError(f"AR audit failed during {report['stage']}; report retained at {args.output}") from error
    if not report["pass_gate"]:
        raise RuntimeError(f"AR audit numerical gate failed; report retained at {args.output}")


if __name__ == "__main__":
    main()
