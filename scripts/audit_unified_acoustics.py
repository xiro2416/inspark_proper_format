#!/usr/bin/env python3
"""Capture real deployed heads, then report frozen-input acoustic error.

Capture and audit are separate serial GPU jobs. The report compares the actual
TensorRT graph output with the exact static quantization recipe and with the
original FP32 model. It never applies numerical acceptance thresholds.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from inspark_infer.runtime.bundle_paths import read_json
import sys
from types import MethodType

import torch


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def metrics(reference, actual):
    if reference.shape != actual.shape:
        raise ValueError(f"Audit shape mismatch: {reference.shape} vs {actual.shape}")
    left, right = reference.detach().cpu().double().flatten(), actual.detach().cpu().double().flatten()
    finite = bool(torch.isfinite(left).all() and torch.isfinite(right).all())
    result = {"shape": list(reference.shape), "finite": finite,
              "reference_dtype": str(reference.dtype), "actual_dtype": str(actual.dtype)}
    if not finite:
        return dict(result, relative_l2=None, relative_l2_percent=None, cosine=None, max_abs=None, mean_abs=None,
                    reference_l2=None, reference_rms=None, actual_rms=None, error_rms=None)
    error = left - right
    ref_norm, actual_norm = float(torch.linalg.vector_norm(left)), float(torch.linalg.vector_norm(right))
    error_norm = float(torch.linalg.vector_norm(error))
    rel = error_norm / ref_norm if ref_norm else (0.0 if actual_norm == 0 else None)
    sqrt_elements = left.numel() ** .5
    cosine = float(torch.dot(left, right)) / (ref_norm * actual_norm) if ref_norm and actual_norm else None
    return dict(result, relative_l2=rel, relative_l2_percent=rel * 100 if rel is not None else None,
                cosine=cosine, max_abs=float(error.abs().max()), mean_abs=float(error.abs().mean()),
                reference_l2=ref_norm, reference_rms=ref_norm/sqrt_elements,
                actual_rms=actual_norm/sqrt_elements, error_rms=error_norm/sqrt_elements)


def cases_for_capture(manifest, batch, waves, start=0, split="evaluation"):
    cases = manifest.get("splits", {}).get(split, manifest.get(split))
    if cases is None:
        cases = [case for case in manifest.get("cases", []) if case.get("split") == split]
    selected = cases[start:start + batch * waves]
    if len(selected) != batch * waves:
        raise ValueError("Manifest has insufficient cases for complete fixed-batch waves")
    references = {ref["voice_id"]: ref["path"] for ref in manifest.get("references", [])}
    result = []
    for i, case in enumerate(selected):
        case = dict(case)
        case["id"] = case.get("id", case.get("case_id", f"audit-{start+i}"))
        source = case.get("reference_audio") or case.get("voice_path") or case.get("audio")
        if source is None:
            voice = case.get("voice_id", case.get("voice"))
            source = references.get(voice) or manifest.get("voices", {}).get(voice)
            if isinstance(source, dict):
                source = source.get("path") or source.get("reference_audio")
        if not isinstance(source, str) or not source:
            raise ValueError(f"Case has no reference audio: {case['id']}")
        case["reference_audio"] = source
        result.append(case)
    return result


def freeze_head_graphs(bank, batch, before):
    """Require a real single replay of each exact head graph, never warmup data."""
    if any(bank.hits[name] - before[name] != 1 for name in ("cfm", "vocoder")):
        raise ValueError("Expected exactly one CFM and Vocoder head replay in this wave")
    cfm, vocoder = bank.cfm[(batch, 258)], bank.vocoder[batch]
    routes = {"cfm": deepcopy(bank.routes["cfm"][(batch, 258)]),
              "vocoder": deepcopy(bank.routes["vocoder"][batch])}
    if routes["cfm"].get("kind") != "tensorrt" or routes["vocoder"].get("kind") not in ("tensorrt","hybrid") or any(route.get("fallback",False) for route in routes.values()):
        raise ValueError("Captured head must execute the selected TensorRT/hybrid acoustic route without fallback")
    copy_cpu = lambda values: tuple(value.detach().cpu().clone() for value in values)
    values = {"cfm_inputs": copy_cpu(cfm.inputs), "cfm_output": cfm.outputs.detach().cpu().clone(),
              "vocoder_inputs": copy_cpu(vocoder.inputs), "vocoder_output": vocoder.outputs.detach().cpu().clone(),
              "routes": routes}
    if not torch.equal(values["cfm_output"][:, :, 258:], values["vocoder_inputs"][0]):
        raise ValueError("Frozen CFM output and Vocoder input do not describe the same acoustic wave")
    return values


def capture(args):
    from inspark_infer.runtime.config import load
    from inspark_infer.runtime.device import GPULease, select_gpu
    from inspark_infer.runtime.deployment import load as load_deployment
    from inspark_infer.runtime.engine import Engine

    deployment = load_deployment(args.deployment)
    if deployment.get("schema") != 9 or not deployment.get("graphs"):
        raise ValueError("Capture requires a schema-9 deployment with graphs enabled")
    batch = deployment["batch"]
    manifest = json.loads(args.manifest.read_text())
    cases = cases_for_capture(manifest, batch, args.waves, args.start, args.split)
    config = load(args.config)
    config["max_batch"] = batch
    plans = {name: read_json(Path(deployment.get("cfm_microbatch_plan",deployment["cfm_plan"]) if name=="cfm" else deployment["vocoder_plan"])) for name in ("cfm", "vocoder")}
    payload = {"schema": 1, "kind": "unified_real_head_acoustic_capture", "batch": batch,
               "scheme": deployment["precision"], "cfm_intervals": [[0, .25], [.25, .5], [.5, .75], [.75, 1]],
               "calibration": {"path": deployment["calibration"], "sha256": digest(deployment["calibration"])},
               "deployment": {"path": str(args.deployment.resolve()), "sha256": digest(args.deployment), "resolved": deployment},
               "manifest": {"path": str(args.manifest.resolve()), "sha256": digest(args.manifest), "split": args.split},
               "plans": plans, "waves": [], "scope": "real first-chunk GPU acoustic outputs before PCM16 conversion"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with GPULease(args.gpu):
        select_gpu(args.gpu)
        engine = Engine(config)
        try:
            voices = {p: f"audit-voice-{i}" for i, p in enumerate(dict.fromkeys(c["reference_audio"] for c in cases))}
            for source, voice in voices.items():
                engine.prepare_reference(voice, source)
            engine.prepare_deployment(deployment)
            original_acoustic = engine.acoustic_rows
            dispatched_rows = []
            def record_rows(_engine, rows, owner, index, *call_args, **call_kwargs):
                if index == 0:
                    dispatched_rows.append([owner[id(row)]["case"]["id"] for row in rows])
                return original_acoustic(rows, owner, index, *call_args, **call_kwargs)
            engine.acoustic_rows = MethodType(record_rows, engine)
            for start in range(0, len(cases), batch):
                group = cases[start:start+batch]
                identifiers = [f"audit-wave-{start // batch}-row-{i}" for i in range(batch)]
                before = dict(engine.head_graphs.hits)
                dispatched_rows.clear()
                for ident, case in zip(identifiers, group):
                    engine.create_session(ident, voices[case["reference_audio"]], case["seed"], case.get("emotion"))
                    engine.push_text(ident, case["text"])
                    engine.finish_input(ident)
                pending, calls = set(identifiers), 0
                while pending:
                    events = engine.run_ready()
                    calls += 1
                    pending.difference_update(event["request_id"] for event in events)
                    if not events or calls > 256:
                        raise RuntimeError(f"Head capture stalled: {sorted(pending)}")
                engine.model.stream.synchronize()
                frozen = freeze_head_graphs(engine.head_graphs, batch, before)
                if len(dispatched_rows) != 1 or set(dispatched_rows[0]) != set(identifiers):
                    raise ValueError("Cannot prove acoustic graph row-to-request mapping")
                frozen["cases"] = group
                by_id = dict(zip(identifiers, group))
                frozen["request_ids"] = list(dispatched_rows[0])
                frozen["cases_by_graph_row"] = [by_id[ident] for ident in dispatched_rows[0]]
                frozen["pcm16_lengths"] = [len(engine.sessions[ident]["chunks"][0]["pcm"]) for ident in identifiers]
                payload["waves"].append(frozen)
                for ident in identifiers:
                    engine.cancel(ident)
                torch.save(payload, args.output)
                print(json.dumps({"captured_waves": len(payload["waves"]), "batch": batch,
                                  "output": str(args.output)}), flush=True)
        finally:
            engine.close()


def _reference_specs(engine, artifact, component, plan):
    from inspark_infer.quantization.unified import iter_roles
    from inspark_infer.build.unified_acoustic_export import acoustic_specs_from_artifact
    model = engine.student.model if component == "cfm" else engine.tts.bigvgan
    bindings = list(iter_roles(engine, artifact["scheme"], (component,)))
    specs, mapping = acoustic_specs_from_artifact(model, bindings, artifact, component)
    specs = deepcopy(specs)
    declared = plan["quantization_recipe"]["role_manifest"]["roles"]
    for role in declared:
        if role.get("conv_transpose_rewrite"):
            if role["conv_transpose_rewrite"] != "zero_insert_conv" or role["path"] not in specs:
                raise ValueError("Unknown deployed ConvTranspose graph rewrite")
            specs[role["path"]]["conv_transpose_rewrite"] = role["conv_transpose_rewrite"]
    for path, spec in specs.items():
        matched = [role for role in declared if role["path"] == path]
        if len(matched) != 1 or matched[0]["precision"] != spec["precision"]:
            raise ValueError(f"Reference precision does not match deployed engine at {mapping[path]}")
    return model, specs


def reference_rewrite_options(plan):
    manifest = plan["quantization_recipe"]["role_manifest"]
    paths = manifest.get("fir_polyphase_graphs", [])
    if not isinstance(paths, list) or any(not isinstance(path, str) for path in paths):
        raise ValueError("Malformed FIR graph rewrite declaration")
    if paths and plan.get("component") != "vocoder":
        raise ValueError("FIR polyphase rewrite belongs to the Vocoder")
    options = {"fir_polyphase": bool(paths)}
    overrides = manifest.get("arithmetic_overrides", {})
    if overrides:
        expected = {"cfm_sdpa_qkv_dtype": "bfloat16", "cfm_sdpa_output_dtype": "float32"}
        if plan.get("component") != "cfm" or overrides != expected:
            raise ValueError("Unknown deployed arithmetic override; cannot reconstruct the same recipe")
        options["cfm_sdpa_bf16"] = True
    return options


def replay_wave_record(original, cfm_output, vocoder_on_source_mel, pcm_output, routes):
    """Keep both same-input Vocoder comparison and the newly selected chain."""
    wave = deepcopy(original)
    wave.update(cfm_output=cfm_output, vocoder_inputs=(cfm_output[:, :, 258:].contiguous(),),
                vocoder_output=pcm_output, routes=deepcopy(routes), source_routes=deepcopy(original["routes"]),
                selected_vocoder_on_source_mel=vocoder_on_source_mel,
                replay_vs_capture={
                    "cfm": metrics(original["cfm_output"], cfm_output),
                    "vocoder_same_captured_mel": metrics(original["vocoder_output"], vocoder_on_source_mel),
                    "selected_acoustic_chain": metrics(original["vocoder_output"], pcm_output)})
    return wave


def replay(args):
    """Run selected engine files; a prior capture's outputs are never relabelled."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
    from compare_unified_acoustic_candidates import load_capture, validate_plan
    from inspark_infer.runtime.device import GPULease, select_gpu
    from inspark_infer.ops.tensorrt.native113 import NativeCFMSolver113, NativeVocoder113

    frozen, calibration = load_capture(args.capture, args.calibration)
    paths = {"cfm": args.cfm_plan.resolve(), "vocoder": args.vocoder_plan.resolve()}
    plans = {component: validate_plan(frozen, path) for component, path in paths.items()}
    if any(plans[component]["component"] != component for component in plans):
        raise ValueError("Selected CFM/Vocoder plan arguments are swapped")
    output = {key: deepcopy(value) for key, value in frozen.items() if key not in ("waves", "plans", "deployment")}
    output.update(kind="unified_frozen_acoustic_replay", replay_complete=False, waves=[], plans=plans,
                  source_capture={"path": str(args.capture.resolve()), "sha256": digest(args.capture)},
                  captured_deployment=deepcopy(frozen["deployment"]),
                  selected_plans={component: {"path": str(path), "plan_sha256": digest(path),
                                              "engine_sha256": plans[component]["sha256"]}
                                  for component, path in paths.items()},
                  scope="selected TensorRT engines replayed on real frozen AR conditions; not a new live AR/E2E capture",
                  replay_execution={"warmups": 1, "captures": 1, "graph_replays": 1,
                                    "numerical_thresholds": None})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with GPULease(args.gpu), torch.inference_mode():
        select_gpu(args.gpu)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            class NoFallback(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.marker = torch.nn.Parameter(torch.zeros(1, device="cuda"), requires_grad=False)
                def forward(self, *inputs):
                    raise RuntimeError("Selected-plan replay must not execute eager fallback")
            marker = NoFallback()
            class CFMReference:
                model = marker
                identity = {"scope": "device_metadata_only"}
                times = tuple(torch.tensor([pair], device="cuda", dtype=torch.float32) for pair in frozen["cfm_intervals"])
                def __call__(self, *inputs):
                    return self.model(*inputs)
            cfm = NativeCFMSolver113(paths["cfm"], CFMReference())
            vocoder = NativeVocoder113(paths["vocoder"], marker)

            def graph_output(wrapper, cpu_inputs):
                inputs = tuple(value.to("cuda").contiguous() for value in cpu_inputs)
                route = wrapper.route_for_signature(*inputs)
                if route["kind"] != "tensorrt":
                    raise ValueError(f"Selected engine cannot execute the frozen signature: {route}")
                wrapper(*inputs)
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    actual = wrapper(*inputs)
                graph.replay()
                stream.synchronize()
                copied = actual.detach().cpu().clone()
                if not bool(torch.isfinite(copied).all()):
                    raise RuntimeError("Selected engine produced a non-finite acoustic output")
                del graph, actual, inputs
                return copied, route

            for index, wave in enumerate(frozen["waves"]):
                mel, cfm_route = graph_output(cfm, wave["cfm_inputs"])
                same_input_pcm, _ = graph_output(vocoder, wave["vocoder_inputs"])
                chain_pcm, vocoder_route = graph_output(vocoder, (mel[:, :, 258:].contiguous(),))
                output["waves"].append(replay_wave_record(wave, mel, same_input_pcm, chain_pcm,
                                                          {"cfm": cfm_route, "vocoder": vocoder_route}))
                torch.save(output, args.output)
                print(json.dumps({"replayed_wave": index, "selected_plans": output["selected_plans"],
                                  "difference_vs_capture": output["waves"][-1]["replay_vs_capture"]}), flush=True)
            if cfm.fallbacks or vocoder.fallbacks:
                raise RuntimeError("Selected-plan replay entered a fallback route")
            output["replay_complete"] = True
            output["replay_execution"]["fallbacks"] = 0
            torch.save(output, args.output)


def audit(args):
    from inspark_infer.runtime.config import load
    from inspark_infer.runtime.device import GPULease, select_gpu
    from inspark_infer.runtime.engine import Engine
    from inspark_infer.quantization.unified import load_artifact
    from inspark_infer.build.unified_acoustic_export import ExportWeightOp, prepare_acoustic_model
    from trt113_provenance import capture_provenance

    frozen = torch.load(args.capture, map_location="cpu", weights_only=True)
    if frozen.get("schema") != 1 or frozen.get("kind") not in ("unified_real_head_acoustic_capture", "unified_frozen_acoustic_replay") or not frozen.get("waves"):
        raise ValueError("Expected an actual deployed acoustic head capture")
    if frozen["kind"] == "unified_frozen_acoustic_replay" and not frozen.get("replay_complete"):
        raise ValueError("Selected-plan replay has not completed all captured waves")
    calibration = args.calibration or Path(frozen["calibration"]["path"])
    if digest(calibration) != frozen["calibration"]["sha256"]:
        raise ValueError("Calibration artifact changed since capture")
    artifact = load_artifact(calibration, frozen["scheme"])
    config = load(args.config)
    config.update(max_batch=frozen["batch"], target_tf32=False)
    report = {"schema": 1, "audit_kind": "reporting_only_no_numerical_gate", "numerical_thresholds": None,
              "capture": {"path": str(args.capture.resolve()), "sha256": digest(args.capture)},
              "scheme": frozen["scheme"], "batch": frozen["batch"], "cfm_intervals": frozen["cfm_intervals"],
              "same_recipe": {"calibration_sha256": digest(calibration), "activation_scaling": "offline_static_per_tensor",
                              "weight_qdq": "materialized_once_in_Torch_reference", "deconv_rewrite": "matches_deployed_plan"},
              "high_precision": {"weights": "same_latest_checkpoint_unquantized_FP32", "tf32": False},
              "scope": "four-step final mel and raw PCM; fixed AR-generated acoustic conditions; no ASR/MOS claim",
              "actual_execution_routes": frozen["waves"][0].get("routes"),
              "input_evidence_kind": frozen["kind"],
              "audited_engines": {component: {"recipe_origin_engine_sha256": plan["sha256"],
                                              "kind": plan["kind"],
                                              "optimization_level": plan.get("optimization_level"),
                                              "tiling": plan.get("tiling_optimization_level"),
                                              "max_aux_streams": plan.get("max_aux_streams"),
                                              "fir_polyphase_graphs": plan["quantization_recipe"]["role_manifest"].get("fir_polyphase_graphs", []),
                                              "arithmetic_overrides": plan["quantization_recipe"]["role_manifest"].get("arithmetic_overrides", {})}
                                  for component, plan in frozen["plans"].items()},
              "selected_plans": frozen.get("selected_plans"),
              "waves": []}
    with GPULease(args.gpu):
        select_gpu(args.gpu)
        engine = Engine(config)
        try:
            report["weight_identity"] = {}
            for component in ("cfm", "vocoder"):
                current = capture_provenance(component, config, args.config, model=engine)
                actual = {row["role"]: row["sha256"] for row in current["model_sources"]}
                expected = {row["role"]: row["sha256"] for row in frozen["plans"][component]["provenance"]["model_sources"]}
                if actual != expected:
                    raise ValueError(f"Reference checkpoint files differ from deployed {component} engine provenance")
                report["weight_identity"][component] = {"verified": True, "sha256_by_role": actual}
            high_precision = []
            with torch.cuda.stream(engine.model.stream), torch.inference_mode():
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
                engine.student.model.float()
                engine.tts.bigvgan.float()
                for wave in frozen["waves"]:
                    inputs = tuple(value.cuda() for value in wave["cfm_inputs"])
                    mel_input = wave["vocoder_inputs"][0].cuda()
                    mel = engine.student(*inputs)
                    pcm = engine.tts.bigvgan(mel_input)
                    chain = engine.tts.bigvgan(mel[:, :, 258:].contiguous())
                    high_precision.append({"mel": mel.cpu(), "pcm": pcm.cpu(), "chain": chain.cpu()})
                role_manifests = {}
                for component in ("cfm", "vocoder"):
                    model, specs = _reference_specs(engine, artifact, component, frozen["plans"][component])
                    options = reference_rewrite_options(frozen["plans"][component])
                    role_manifests[component] = prepare_acoustic_model(model, specs, **options)
                    expected_fir = frozen["plans"][component]["quantization_recipe"]["role_manifest"].get("fir_polyphase_graphs", [])
                    if sorted(role_manifests[component].get("fir_polyphase_graphs", [])) != sorted(expected_fir):
                        raise ValueError("Reference FIR rewrite coverage differs from the selected engine")
                    expected_arithmetic = frozen["plans"][component]["quantization_recipe"]["role_manifest"].get("arithmetic_overrides", {})
                    if role_manifests[component].get("arithmetic_overrides", {}) != expected_arithmetic:
                        raise ValueError("Reference arithmetic override differs from the selected engine")
                    for module in model.modules():
                        if isinstance(module, ExportWeightOp):
                            module.prepare_reference_weights()
                report["same_recipe"]["role_manifests"] = role_manifests
                for index, (wave, fp32) in enumerate(zip(frozen["waves"], high_precision)):
                    inputs = tuple(value.cuda() for value in wave["cfm_inputs"])
                    mel_input = wave["vocoder_inputs"][0].cuda()
                    recipe_mel = engine.student(*inputs).cpu()
                    recipe_pcm = engine.tts.bigvgan(mel_input).cpu()
                    recipe_chain = engine.tts.bigvgan(recipe_mel[:, :, 258:].cuda().contiguous()).cpu()
                    trt_mel, trt_pcm = wave["cfm_output"], wave["vocoder_output"]
                    comparisons = {
                        "cfm_trt_vs_same_recipe": (recipe_mel, trt_mel),
                        "cfm_trt_vs_original_fp32": (fp32["mel"], trt_mel),
                        "cfm_same_recipe_vs_original_fp32": (fp32["mel"], recipe_mel),
                        "vocoder_trt_vs_same_recipe_frozen_mel": (recipe_pcm, trt_pcm),
                        "vocoder_trt_vs_original_fp32_frozen_mel": (fp32["pcm"], trt_pcm),
                        "acoustic_chain_trt_vs_same_recipe": (recipe_chain, trt_pcm),
                        "acoustic_chain_trt_vs_original_fp32": (fp32["chain"], trt_pcm),
                    }
                    row = {"wave": index, "cases": wave["cases"], "routes": wave["routes"],
                           "comparisons": {name: metrics(ref, actual) for name, (ref, actual) in comparisons.items()},
                           "per_row": [{"row": i, "case": wave["cases_by_graph_row"][i],
                                        "cfm": metrics(recipe_mel[i], trt_mel[i]),
                                        "vocoder": metrics(recipe_pcm[i], trt_pcm[i])} for i in range(frozen["batch"])]}
                    if any(not value["finite"] for value in row["comparisons"].values()):
                        raise RuntimeError("Acoustic audit contains a non-finite selected output or reference")
                    report["waves"].append(row)
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
                    print(json.dumps({"audited_wave": index, "comparisons": row["comparisons"]}), flush=True)
        finally:
            engine.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("capture", "replay", "audit"):
        command = commands.add_parser(name)
        command.add_argument("--gpu", type=int, default=7)
        command.add_argument("--config", default="artifacts/current_release/runtime.yaml")
        command.add_argument("--output", type=Path, required=True)
        if name == "capture":
            command.add_argument("--manifest", type=Path, required=True)
            command.add_argument("--deployment", type=Path, required=True)
            command.add_argument("--waves", type=int, default=1)
            command.add_argument("--start", type=int, default=0)
            command.add_argument("--split", default="evaluation")
        else:
            command.add_argument("--capture", type=Path, required=True)
            command.add_argument("--calibration", type=Path)
            if name == "replay":
                command.add_argument("--cfm-plan", type=Path, required=True)
                command.add_argument("--vocoder-plan", type=Path, required=True)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if args.command == "capture":
        if args.waves <= 0 or args.start < 0:
            parser.error("waves must be positive and start nonnegative")
        capture(args)
    elif args.command == "replay":
        replay(args)
    else:
        audit(args)


if __name__ == "__main__":
    main()
