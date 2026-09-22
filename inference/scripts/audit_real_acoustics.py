#!/usr/bin/env python3
"""Freeze real deployed CFM/Vocoder calls, then replay a single eager model.

Run capture, reference --precision fp32 and reference --precision bf16 as three
sequential processes. Capture includes full-EOS heads/tails, not synthetic mel.
It instruments inference and therefore makes NO latency/throughput claim.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import traceback
from types import MethodType

from acc_infer_clear.guardrails.numerics import TOLERANCES, compare
from acc_infer_clear.guardrails.snapshots import (
    compare_regions, cpu_copy, file_sha256, load_bundle, save_bundle, write_json,
)
from trt113_provenance import capture_provenance, file_record, source_identity


def _plain_route(component, fn, inputs):
    from acc_infer_clear.runtime.graphs import HeadGraphs
    return HeadGraphs._describe_route(component, fn, inputs)


class AcousticRecorder:
    """Script-level proxies only: never changes model math or runtime source."""
    def __init__(self, engine, directory, manifest):
        self.engine, self.directory, self.manifest = engine, Path(directory), manifest
        self.sessions, self.index, self.last_members = [], None, []
        self.original_student, self.original_vocoder = engine.student, engine.vocoder

    def members(self, component, inputs):
        if component == "cfm":
            frames = inputs[0].shape[-1]
            # This is the grouping key used by StreamingCore.acoustic_rows.
            plen = int(inputs[5][0, 0].sum())
            self.last_members = [s for s in self.sessions if "acoustic" in s
                                 and s["acoustic"][0].shape[-1] == frames
                                 and s["plen"] == plen]
        members = self.last_members
        if len(members) != inputs[0].shape[0]:
            raise ValueError("Unable to prove actual acoustic request-to-row mapping")
        return [{"id": s["case"]["id"], "codes": list(s["new_codes"]),
                 "eos": bool(s["eos"]), "core": s["core"], "horizon": s["horizon"],
                 "prompt_frames": s["plen"], "emitted_before": s["emitted"],
                 "accepted": list(s["accepted"]), "kv_head_length": s["kv_head_lengths"]}
                for s in members]

    def call(self, component, fn, inputs, route, direct=None):
        import torch
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Audit observers must not execute inside graph capture")
        members = self.members(component, inputs)
        frozen_inputs = cpu_copy(inputs)  # before even a stateful callable can mutate inputs
        output = fn(*inputs)
        frozen_output = cpu_copy(output)  # before any subsequent invocation overwrites it
        bundle = {"inputs": frozen_inputs, "output": frozen_output}
        graph_check = None
        if direct is not None:
            # Acoustic functions have no request-owned mutable cache. Frozen
            # copies still avoid aliasing graph static input/output storage.
            replay_args = tuple(value.to(inputs[0].device).clone() for value in frozen_inputs)
            direct_output = cpu_copy(direct(*replay_args))
            bundle["direct_output"] = direct_output
            graph_check = compare(direct_output, frozen_output, "fp32")
            graph_check["bitwise_exact"] = bool(torch.equal(direct_output, frozen_output))
            graph_check["exact_gate"] = graph_check["bitwise_exact"]
        ordinal = len(self.manifest["calls"])
        evidence = save_bundle(self.directory / "bundles", ordinal, bundle)
        row = {"index": ordinal, "component": component, "chunk_index": self.index,
               "head": self.index == 0, "route": deepcopy(route), "members": members,
               "evidence": evidence, "graph_vs_direct": graph_check,
               "direct_graph_scope": "same callable and frozen inputs; no timing claim"}
        self.manifest["calls"].append(row)
        write_json(self.directory / "capture.json", self.manifest)
        # A second direct invocation may reuse output; return the frozen value
        # on the original device, preserving values/dtype/shape for the pipeline.
        return frozen_output.to(output.device) if direct is not None else output

    def install(self):
        engine = self.engine
        original_acoustic = engine.acoustic_rows
        recorder = self
        def acoustic_rows(_engine, rows, owner, index, *args, **kwargs):
            recorder.sessions = [owner[id(row)] for row in rows]
            recorder.index = index
            try:
                return original_acoustic(rows, owner, index, *args, **kwargs)
            finally:
                recorder.sessions, recorder.last_members = [], []
        engine.acoustic_rows = MethodType(acoustic_rows, engine)
        for component, attribute, original in (
                ("cfm", "student", self.original_student),
                ("vocoder", "vocoder", self.original_vocoder)):
            def tail(*inputs, component=component, original=original):
                route = _plain_route(component, original, inputs)
                route.update(execution="direct", graph_fallback="tail_not_captured")
                return recorder.call(component, original, inputs, route)
            setattr(engine, attribute, tail)
        bank = engine.head_graphs
        if bank is not None:
            original_run = bank._run
            def run(_bank, component, key, inputs):
                graph = (bank.cfm if component == "cfm" else bank.vocoder).get(key)
                compatible = graph is not None and len(inputs) == len(graph.inputs) and all(
                    x.shape == y.shape and x.dtype == y.dtype and x.device == y.device
                    for x, y in zip(inputs, graph.inputs))
                if compatible:
                    route = deepcopy(bank.routes[component][key])
                    route.update(execution="graph", graph_key=list(key) if isinstance(key, tuple) else key)
                else:
                    route = _plain_route(component, bank._direct[component], inputs)
                    route.update(execution="direct", fallback=True,
                                 graph_fallback="missing_graph" if graph is None else "input_signature")
                return recorder.call(component, lambda *args: original_run(component, key, args),
                                     inputs, route, bank._direct[component] if compatible else None)
            bank._run = MethodType(run, bank)


def selected_cases(args):
    corpus = json.loads(args.corpus.read_text())
    cases = corpus["cases"][args.start:args.start + args.cases]
    if len(cases) != args.cases or len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Requested corpus range is missing or has duplicate IDs")
    return cases


def reference_provenance(engine, args):
    return {component: capture_provenance(component, engine.config, args.config, model=engine)
            for component in ("cfm", "vocoder")}


def _valid_sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def acoustic_engine_evidence(plan_path, component, current_paths, *, expected_engine_sha256=None,
                             expected_plan_sha256=None):
    """Match an ONNX-built engine's attested sources to actual loader files.

    Like the AR audit, declared paths never choose the reference checkpoint.
    ONNX/source hashes are build attestations, not an extraction of engine
    constants. Missing legacy provenance is explicitly unverified, never filled
    in using today's checkpoints or a newly added build metadata sidecar.
    """
    roles = {"cfm": {"s2mel_checkpoint", "student_checkpoint", "s2mel_config"},
             "vocoder": {"vocoder_checkpoint", "vocoder_config"}}
    if component not in roles or set(current_paths) != roles[component]:
        raise ValueError(f"Reference model source roles do not match {component}")
    plan_path = Path(plan_path).resolve()
    plan_file = file_record(plan_path, "acoustic_engine_plan")
    if expected_plan_sha256 is not None and plan_file["sha256"] != expected_plan_sha256:
        raise ValueError("Acoustic plan SHA256 changed after capture/preparation")
    plan = json.loads(plan_path.read_text())
    path = Path(plan["engine"])
    path = path if path.is_absolute() else plan_path.parent / path
    engine_file = file_record(path, "acoustic_engine")
    if plan.get("sha256") != engine_file["sha256"]:
        raise ValueError("Acoustic engine SHA256 differs from its build plan")
    if expected_engine_sha256 is not None and engine_file["sha256"] != expected_engine_sha256:
        raise ValueError("Acoustic engine SHA256 differs from the captured/loaded engine")
    actual = {role: file_record(path, role) for role, path in current_paths.items()}
    provenance = plan.get("provenance")
    result = {"component": component, "has_native_engine": True, "plan": plan_file,
              "engine": engine_file, "build_metadata": plan, "actual_model_sources": list(actual.values()),
              "model_source_checks": [], "weight_identity_verified": False,
              "provenance_status": "legacy_unverified", "engine_hash_verified": True,
              "engine_constants_independently_verified": False, "onnx_payloads_rehashed_by_audit": False,
              "weight_identity_note": "Legacy plan lacks complete hash-bound model-source provenance; numerical results do not attest engine weights"}
    if not isinstance(provenance, dict) or provenance.get("status") != "recorded_not_audited":
        return result
    if provenance.get("schema") != 1 or provenance.get("component") != component:
        raise ValueError("Recorded acoustic provenance schema/component mismatch")
    declared = provenance.get("model_sources")
    if not isinstance(declared, list) or any(not isinstance(row, dict) for row in declared):
        raise ValueError("Recorded acoustic provenance has no valid model_sources list")
    by_role = {row.get("role"): row for row in declared}
    if len(by_role) != len(declared) or set(by_role) != roles[component]:
        raise ValueError("Recorded acoustic model source roles are missing, duplicated or unexpected")
    for role, current in actual.items():
        build = by_role[role]
        if build.get("sha256") != current["sha256"]:
            raise ValueError(f"Acoustic build model source SHA256 mismatch for {role}")
        result["model_source_checks"].append({"role": role, "current": current, "build": build,
            "sha256_matches": True, "path_matches": build.get("path") == current["path"]})
    source = provenance.get("source", {})
    binding = provenance.get("onnx_binding", {})
    if not isinstance(source, dict) or not _valid_sha256(source.get("source_sha256")):
        raise ValueError("Recorded acoustic provenance lacks export source fingerprint")
    if (not isinstance(binding, dict) or not isinstance(binding.get("onnx"), dict)
            or not _valid_sha256(binding["onnx"].get("sha256"))
            or binding["onnx"]["sha256"] != plan.get("onnx_sha256")
            or not isinstance(binding.get("export_metadata"), dict)
            or not _valid_sha256(binding["export_metadata"].get("sha256"))
            or not isinstance(binding.get("external_data"), list)):
        raise ValueError("Recorded acoustic provenance lacks a hash-bound ONNX export chain")
    external = binding["external_data"]
    if any(not isinstance(row, dict) or not isinstance(row.get("path"), str) or not row["path"]
           or not _valid_sha256(row.get("sha256")) for row in external):
        raise ValueError("Recorded acoustic ONNX external-data inventory is invalid")
    if len({row["path"] for row in external}) != len(external):
        raise ValueError("Recorded acoustic ONNX external-data paths are duplicated")
    result.update(weight_identity_verified=True, provenance_status="recorded_not_audited",
                  onnx_build_binding=deepcopy(binding),
                  weight_identity_note="Every actual loader checkpoint/config hash matches its build role and engine bytes match the hash-bound plan. ONNX/source hashes are builder attestations; engine constants were not independently extracted.")
    return result


def capture_acoustic_engine_evidence(engine, model_provenance):
    evidence = {}
    for component, wrapper in (("cfm", engine.student), ("vocoder", engine.vocoder)):
        if not hasattr(wrapper, "engine_sha256") or not hasattr(wrapper, "plan"):
            evidence[component] = {"has_native_engine": False, "weight_identity_verified": False,
                                   "provenance_status": "not_applicable", "reason": "no native acoustic wrapper"}
            continue
        current_paths = {row["role"]: row["path"] for row in model_provenance[component]["model_sources"]}
        evidence[component] = acoustic_engine_evidence(wrapper.plan, component, current_paths,
            expected_engine_sha256=wrapper.engine_sha256,
            expected_plan_sha256=wrapper.provenance["plan_sha256"])
    return evidence


def replay_acoustic_engine_evidence(manifest, model_provenance):
    evidence = {}
    captured = manifest.get("acoustic_engines", {})
    for component in ("cfm", "vocoder"):
        row = captured.get(component)
        if row is None:
            # Do not infer historical proof by reading today's plan sidecar.
            evidence[component] = {"has_native_engine": None, "weight_identity_verified": False,
                "provenance_status": "legacy_unverified", "reason": "capture has no frozen acoustic engine provenance"}
        elif not row["has_native_engine"]:
            evidence[component] = deepcopy(row)
        else:
            current_paths = {item["role"]: item["path"] for item in model_provenance[component]["model_sources"]}
            current = acoustic_engine_evidence(row["plan"]["path"], component, current_paths,
                expected_plan_sha256=row["plan"]["sha256"], expected_engine_sha256=row["engine"]["sha256"])
            if current["weight_identity_verified"] and not row.get("weight_identity_verified", False):
                raise ValueError("Reference replay cannot retroactively upgrade unverified capture provenance")
            evidence[component] = current
    return evidence


def all_acoustic_weights_verified(evidence):
    native = [row for row in evidence.values() if row.get("has_native_engine") is not False]
    return bool(native) and all(row["weight_identity_verified"] for row in native)


def direct_graph_gate(row):
    if row["route"].get("execution") != "graph":
        return {"required": False, "pass_gate": True, "reason": "actual direct route; no graph invoked"}
    check = row.get("graph_vs_direct")
    return {"required": True, "pass_gate": isinstance(check, dict)
            and check.get("exact_gate") is True and check.get("pass_gate") is True,
            "reason": "requires recorded finite bitwise-exact direct/graph evidence"}


def acoustic_coverage(manifest):
    result = {}
    for component in ("cfm", "vocoder"):
        rows = [row for row in manifest["calls"] if row["component"] == component]
        native = [row for row in rows if row["route"].get("backend") == "tensorrt113"]
        declared = manifest.get("acoustic_engines", {}).get(component)
        native_required = (declared.get("has_native_engine", False) if declared is not None else
                           any(row["route"].get("candidate_backend") == "tensorrt113" for row in rows))
        result[component] = {"observed_calls": len(rows), "native_calls": len(native),
                             "other_calls": len(rows) - len(native), "native_required": native_required,
                             "actual_native_batches": sorted({row["route"]["batch"] for row in native}),
                             "pass_gate": bool(rows) and (not native_required or bool(native))}
    return {"components": result, "pass_gate": all(row["pass_gate"] for row in result.values()),
            "scope": "installed native components require at least one observed native execution; fallbacks remain separately audited"}


def capture_run(args):
    import numpy as np
    import soundfile as sf
    import torch
    from acc_infer_clear.runtime.config import load
    from acc_infer_clear.runtime.deployment import load as deployment_load
    from acc_infer_clear.runtime.engine import Engine

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Use a fresh output directory; captures are immutable")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = selected_cases(args)
    config = load(args.config); config["max_batch"] = args.batch
    manifest = {"schema": 1, "scope": "real_trajectory_acoustic_same_input_audit",
                "status": "running", "calls": [], "requests": [], "cases": cases,
                "corpus": file_record(args.corpus, "corpus"), "tolerances": TOLERANCES,
                "config": config, "source": source_identity(), "performance_claim": False,
                "coverage": ["CFM full output/masked/generated", "Vocoder pre-clamp waveform",
                             "existing head graph versus direct", "full-EOS codes and chunks"],
                "not_covered": ["AR internal tensors", "latent and condition internal tensors",
                                "CFM estimator intermediate steps", "crossfade arithmetic replay"]}
    engine = None
    try:
        engine = Engine(config)
        voices = {path: f"voice-{index}" for index, path in enumerate(
            dict.fromkeys(case["reference_audio"] for case in cases))}
        manifest["references"] = [file_record(path, "reference_audio") for path in voices]
        for path, name in voices.items():
            engine.prepare_reference(name, path)
        manifest["model_provenance"] = reference_provenance(engine, args)
        manifest["deployment"] = engine.prepare_deployment(deployment_load(args.deployment))
        manifest["deployment_file"] = file_record(args.deployment, "deployment")
        manifest["acoustic_engines"] = capture_acoustic_engine_evidence(engine, manifest["model_provenance"])
        manifest["acoustic_weight_identity_verified"] = all_acoustic_weights_verified(manifest["acoustic_engines"])
        manifest["hardware"] = {"name": torch.cuda.get_device_name(),
                                "sm": list(torch.cuda.get_device_capability())}
        AcousticRecorder(engine, args.output_dir, manifest).install()
        for start in range(0, len(cases), args.batch):
            group = cases[start:start + args.batch]
            for case in group:
                engine.create_session(case["id"], voices[case["reference_audio"]],
                                      case["seed"], case["emotion"])
                engine.push_text(case["id"], case["text"])
                engine.finish_input(case["id"])
            while engine.ready():
                engine.run_ready()
            for case in group:
                session = engine.sessions[case["id"]]
                if not session["complete"] or session["error"] or not session.get("eos"):
                    raise RuntimeError(f"Request did not complete through EOS: {case['id']}")
                pcm = np.concatenate([chunk["pcm"] for chunk in session["chunks"]])
                if not len(pcm):
                    raise RuntimeError("Empty completed waveform")
                path = args.output_dir / f"{case['id']}.wav"
                sf.write(path, pcm, 22050, subtype="PCM_16")
                manifest["requests"].append({"id": case["id"], "complete": True, "eos": True,
                    "codes": session["codes"], "accepted": session["accepted"],
                    "rounds": session["rounds"], "wave": file_record(path, "generated_audio"),
                    "samples": len(pcm), "chunks": [{k: v for k, v in chunk.items() if k != "pcm"}
                                                     for chunk in session["chunks"]]})
                engine.release(case["id"])
        if not manifest["calls"]:
            raise RuntimeError("No actual acoustic invocations were observed")
        manifest["head_routes"] = engine.head_graphs.stats() if engine.head_graphs else None
        manifest["acoustic_capture_coverage"] = acoustic_coverage(manifest)
        manifest["status"] = "captured"
        manifest["direct_graph_exact"] = all(row["graph_vs_direct"]["exact_gate"]
            for row in manifest["calls"] if row["graph_vs_direct"] is not None)
    except Exception as error:
        manifest["status"] = "error"
        manifest["error"] = {"type": type(error).__name__, "message": str(error),
                             "traceback": traceback.format_exc()}
        raise
    finally:
        if engine is not None:
            engine.close()
        write_json(args.output_dir / "capture.json", manifest)


def replay_reference(args):
    import torch
    from acc_infer_clear.runtime.config import load
    from acc_infer_clear.runtime.engine import Engine

    manifest_path = args.capture_dir / "capture.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "captured" or not manifest.get("calls"):
        raise ValueError("A complete nonempty trajectory capture is required")
    output_dir = args.capture_dir / f"reference_{args.precision}"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Reference replay already exists; preserve previous evidence")
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load(args.config); config["max_batch"] = manifest["config"]["max_batch"]
    config["target_tf32"] = False
    report = {"schema": 1, "scope": "same_frozen_real_acoustic_inputs", "status": "running",
              "capture": file_record(manifest_path, "capture_manifest"), "calls": [],
              "precision": args.precision, "tolerances": TOLERANCES, "source": source_identity(),
              "reference": "raw Engine; no deployment, no graphs, no custom acoustic kernels",
              "arithmetic": {"tf32": False, "policy": args.precision,
                 "bf16": "Linear/Conv wrapper arithmetic; FP32 interfaces; not identical to TensorRT arithmetic"},
              "performance_claim": False, "pass_gate": False}
    engine = None
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        engine = Engine(config)
        report["model_provenance"] = reference_provenance(engine, args)
        for component, provenance in report["model_provenance"].items():
            expected = {r["role"]: r["sha256"] for r in manifest["model_provenance"][component]["model_sources"]}
            actual = {r["role"]: r["sha256"] for r in provenance["model_sources"]}
            if expected != actual:
                raise ValueError(f"{component} reference checkpoint/config hashes changed after capture")
        report["same_loader_checkpoint_identity"] = True
        report["acoustic_engines"] = replay_acoustic_engine_evidence(manifest, report["model_provenance"])
        report["acoustic_weight_identity_verified"] = all_acoustic_weights_verified(report["acoustic_engines"])
        if args.precision == "bf16":
            engine.prepare_precision("bf16", ["cfm", "vocoder"], True)
        with torch.cuda.stream(engine.model.stream), torch.inference_mode():
            for row in manifest["calls"]:
                bundle = load_bundle(args.capture_dir / "bundles", row["evidence"])
                inputs = tuple(value.to("cuda:0") for value in bundle["inputs"])
                fn = engine.student if row["component"] == "cfm" else engine.vocoder
                expected = cpu_copy(fn(*inputs))
                output = save_bundle(output_dir / "bundles", row["index"], {"output": expected})
                mask = bundle["inputs"][5] if row["component"] == "cfm" else None
                comparison = compare_regions(expected, bundle["output"], "bf16", mask)
                checks = {"deployed_vs_reference": comparison}
                if "direct_output" in bundle:
                    checks["direct_vs_reference"] = compare_regions(expected, bundle["direct_output"], "bf16", mask)
                if args.precision == "bf16":
                    fp_report_path = args.capture_dir / "reference_fp32" / "report.json"
                    if fp_report_path.exists():
                        fp_report = json.loads(fp_report_path.read_text())
                        fp_row = next(item for item in fp_report["calls"] if item["index"] == row["index"])
                        if fp_report["capture"]["sha256"] != report["capture"]["sha256"]:
                            raise ValueError("FP32 reference was not generated from the same capture")
                        fp = load_bundle(args.capture_dir / "reference_fp32" / "bundles", fp_row["evidence"])["output"]
                        checks["bf16_reference_vs_fp32_reference"] = compare_regions(fp, expected, "bf16", mask)
                exact_graph = direct_graph_gate(row)
                report["calls"].append({"index": row["index"], "component": row["component"],
                                        "route": row["route"], "evidence": output, "checks": checks,
                                        "direct_graph_gate": exact_graph,
                                        "pass_gate": exact_graph["pass_gate"] and all(v["pass_gate"] for v in checks.values())})
                write_json(output_dir / "report.json", report)
        report["status"] = "completed"
        report["coverage"] = acoustic_coverage(manifest)
        report["numerical_pass_gate"] = all(row["pass_gate"] for row in report["calls"])
        report["pass_gate"] = report["coverage"]["pass_gate"] and report["numerical_pass_gate"]
        report["conclusion_scope"] = "Numerical evidence only; legacy engine source identity and perceptual quality are separate gates"
    except Exception as error:
        report["status"] = "error"
        report["error"] = {"type": type(error).__name__, "message": str(error),
                           "traceback": traceback.format_exc()}
        raise
    finally:
        if engine is not None:
            engine.close()
        write_json(output_dir / "report.json", report)
    if not report["pass_gate"]:
        raise SystemExit(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=6)
    parser.add_argument("--config", default="configs/runtime.yaml")
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("capture")
    capture.add_argument("--deployment", required=True)
    capture.add_argument("--corpus", type=Path, default=Path("configs/sm89_quality_256.json"))
    capture.add_argument("--start", type=int, default=0)
    capture.add_argument("--cases", type=int, default=1)
    capture.add_argument("--batch", type=int, choices=(1, 4, 8, 16), default=1)
    capture.add_argument("--output-dir", type=Path, required=True)
    reference = commands.add_parser("reference")
    reference.add_argument("--capture-dir", type=Path, required=True)
    reference.add_argument("--precision", choices=("fp32", "bf16"), required=True)
    args = parser.parse_args()
    if args.command == "capture" and (args.cases <= 0 or args.start < 0):
        parser.error("--cases must be positive and --start nonnegative")
    from acc_infer_clear.runtime.device import GPULease, select_gpu
    with GPULease(args.gpu):
        select_gpu(args.gpu)
        {"capture": capture_run, "reference": replay_reference}[args.command](args)


if __name__ == "__main__":
    main()
