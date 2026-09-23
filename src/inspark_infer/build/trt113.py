"""Build one complete static TensorRT 11.3 first-chunk bundle per exact batch.

The child exporters/builders own their GPU leases.  This orchestrator never
imports CUDA or TensorRT, and it never builds in a request-serving process.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid


PROFILE = "first_chunk_p258_f52_k128"
BATCHES = (1, 4, 8)
COMPONENTS = ("target", "draft", "cfm", "vocoder")
BUILDER_DEFAULT = "configs/common/trt113_builder.json"
QUANTIZATION_DEFAULT = "configs/common/trt113_quantization.json"


def _repository_root() -> Path:
    """The wheel supplies the CLI, while checkout scripts/configs remain required."""
    candidates = [os.getenv("INSPARK_REPO_ROOT"), Path.cwd(), Path(__file__).resolve().parents[3]]
    for candidate in candidates:
        if not candidate:
            continue
        root = Path(candidate).resolve()
        if (root.is_relative_to(Path("/workspace")) and (root / "scripts/run.sh").is_file()
                and (root / "configs/common/model_sources.json").is_file()):
            return root
    raise RuntimeError("Run in an inspark-infer checkout under /workspace or set INSPARK_REPO_ROOT")


ROOT = _repository_root()


def ensure_trt113_site() -> Path:
    site = Path(os.getenv("ACC_TRT113_SITE", str(ROOT / ".venv-trt113/lib/python3.11/site-packages")))
    if not site.is_dir():
        raise ValueError("TensorRT 11.3 environment missing; run scripts/bootstrap_trt113.sh")
    site = site.resolve()
    os.environ["ACC_TRT113_SITE"] = str(site)
    return site


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


def _digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def parse_batches(raw: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in raw.split(","))
    except ValueError as exc:
        raise ValueError("--batches must be comma-separated integers") from exc
    if not values or len(set(values)) != len(values) or any(value < 1 for value in values):
        raise ValueError("--batches requires unique positive integers")
    return values


def builder_policy(path: Path) -> dict:
    data = json.loads(path.read_text())
    if set(data) != {"schema", "optimization_level", "tiling_optimization_level",
                     "workspace_bytes", "strongly_typed", "tf32"} or data["schema"] != 1:
        raise ValueError("Unsupported TensorRT builder policy schema")
    levels = data["optimization_level"]
    if (not isinstance(levels, dict) or set(levels) != set(COMPONENTS)
            or any(type(level) is not int or level not in range(6) for level in levels.values())
            or data["tiling_optimization_level"] not in ("none", "fast", "moderate", "full")
            or type(data["workspace_bytes"]) is not int or data["workspace_bytes"] < 1
            or data["strongly_typed"] is not True or data["tf32"] is not False):
        raise ValueError("Invalid TensorRT 11.3 builder settings or precision contract")
    return data


def gpu_info(gpu: int) -> dict:
    if gpu < 0:
        raise ValueError("--gpu must be a nonnegative physical GPU index")
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,compute_cap,memory.total,memory.used",
         "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True,
    )
    for row in csv.reader(result.stdout.splitlines()):
        if len(row) < 5 or int(row[0].strip()) != gpu:
            continue
        major, minor = (int(part) for part in row[2].strip().split("."))
        return {"physical_gpu": gpu, "name": row[1].strip(), "sm": major * 10 + minor,
                "memory_total_mib": int(row[3].strip()), "memory_used_mib": int(row[4].strip())}
    raise ValueError(f"Physical GPU {gpu} was not reported by nvidia-smi")


def unsupported(gpu: dict, profile: str, batches: tuple[int, ...]) -> dict:
    reasons = []
    if gpu["sm"] < 80:
        reasons.append(f"SM{gpu['sm']} is below the supported SM80+ runtime capability")
    if profile != PROFILE:
        reasons.append(f"Shape profile {profile!r} is not implemented; supported: {PROFILE}")
    if any(batch not in BATCHES for batch in batches):
        reasons.append(f"Exact batches must be in {BATCHES}")
    return {"status": "unsupported", "reasons": reasons, "gpu": gpu, "profile": profile,
            "batches": list(batches), "codex_task": {
                "source_entries": ["scripts/build_trt113_target_full.py", "scripts/build_trt113_draft_full.py",
                                   "scripts/export_trt113_cfm_onnx.py", "scripts/export_trt113_vocoder_onnx.py",
                                   "src/inspark_infer/ops/tensorrt/native113.py"],
                "required_work": "Implement each requested profile/batch on the target SM without changing model semantics; update IO checks, export, plugins and runtime dispatch only as required.",
                "acceptance": "Same-input eager boundary audit; real four-component route; full EOS quality; first PCM, power, 32/64 concurrency, long-sequence and soak evidence.",
            }}


def _run(step: str, args: list[str], *, stage: Path, gpu: int) -> None:
    log = stage / "logs" / f"{step}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.pop("HF_TOKEN", None)  # builders use already-downloaded, pinned local weights
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    with log.open("w") as output:
        result = subprocess.run(["bash", "scripts/run.sh", *args], cwd=ROOT, env=env,
                                stdout=output, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"{step} failed with exit {result.returncode}; see {log}")


def _steps(batch: int, stage: Path, gpu: int, builder: dict | None = None) -> list[tuple[str, list[str]]]:
    def path(component: str, name: str) -> str:
        return str(stage / component / name)

    steps = [
        ("target", ["scripts/build_trt113_target_full.py", "--gpu", str(gpu), "--batch", str(batch),
                    "--out-dir", str(stage / "target"), "--plan", path("target", "plan.json")]),
        ("draft", ["scripts/build_trt113_draft_full.py", "--gpu", str(gpu), "--batch", str(batch),
                   "--out-dir", str(stage / "draft"), "--plan", path("draft", "plan.json")]),
        ("cfm_export", ["scripts/export_trt113_cfm_onnx.py", "--gpu", str(gpu),
                        "--batch", str(batch), "--frames", "310", "--prompt-frames", "258",
                        "--config", "configs/common/runtime_reference.yaml",
                        "--output", path("cfm", "solver.onnx")]),
        ("cfm", ["scripts/build_trt113_cfm_onnx.py", "--gpu", str(gpu),
                   "--batch", str(batch), "--frames", "310", "--onnx", path("cfm", "solver.onnx"),
                   "--engine", path("cfm", "solver.engine"), "--plan", path("cfm", "plan.json")]),
        ("vocoder_export", ["scripts/export_trt113_vocoder_onnx.py", "--gpu", str(gpu),
                            "--batch", str(batch), "--frames", "52", "--regular-conv", "native",
                            "--config", "configs/common/runtime_reference.yaml",
                            "--output", path("vocoder", "vocoder.onnx")]),
        ("vocoder", ["scripts/build_trt113_vocoder_onnx.py", "--gpu", str(gpu),
                       "--batch", str(batch), "--frames", "52", "--strongly-typed",
                       "--onnx", path("vocoder", "vocoder.onnx"),
                       "--engine", path("vocoder", "vocoder.engine"),
                       "--plan", path("vocoder", "plan.json")]),
    ]
    if builder is not None:
        for name, command in steps:
            if name in COMPONENTS:
                command.extend(("--optimization-level", str(builder["optimization_level"][name]),
                                "--workspace-bytes", str(builder["workspace_bytes"]),
                                "--tiling-optimization-level", builder["tiling_optimization_level"]))
    return steps


def _engine_record(stage: Path, component: str, batch: int, gpu: dict) -> dict:
    plan_path = stage / component / "plan.json"
    plan = json.loads(plan_path.read_text())
    if component in ("target", "draft"):
        raw_path = plan["engines"][str(batch)]
        expected = plan["engine_sha256"][str(batch)]
        provenance = plan["provenance"][str(batch)]
        tensors = plan["tensors"][str(batch)]
        if plan.get("kv_limit") != 128:
            raise ValueError(f"{component} KV limit is not 128")
    else:
        raw_path = plan["engine"]
        expected = plan["sha256"]
        provenance = plan["provenance"]
        if plan.get("batch") != batch or plan.get("frames") != (310 if component == "cfm" else 52):
            raise ValueError(f"{component} static shape is not the requested B{batch} profile")
        tensors = plan["tensors"]
    if (not isinstance(tensors, list) or not tensors
            or any(not isinstance(row.get("shape"), list) or row["shape"][0] != batch
                   for row in tensors)):
        raise ValueError(f"{component} engine IO is not static B{batch}")
    engine = Path(raw_path)
    engine = (engine if engine.is_absolute() else plan_path.parent / engine).resolve()
    if not engine.is_relative_to(stage.resolve()) or not engine.is_file():
        raise ValueError(f"{component} engine is absent or escapes its bundle")
    digest = _digest(engine)
    if (digest != expected or plan.get("sm") != gpu["sm"]
            or plan.get("gpu_name") != gpu["name"]
            or not str(plan.get("trt", "")).startswith("11.3.")):
        raise ValueError(f"{component} engine/hash/hardware/SDK mismatch")
    if not isinstance(provenance, dict) or provenance.get("status") != "recorded_not_audited":
        raise ValueError(f"{component} has no source-attested build provenance")
    source = provenance.get("source", {}).get("source_sha256")
    if not source:
        raise ValueError(f"{component} has no source fingerprint")
    model_sources = provenance.get("model_sources")
    if not isinstance(model_sources, list) or not model_sources:
        raise ValueError(f"{component} has no pinned model-source record")
    return {"engine": str(engine.relative_to(stage)), "engine_sha256": digest,
            "plan": str(plan_path.relative_to(stage)), "plan_sha256": _digest(plan_path),
            "trt": plan["trt"], "sm": plan["sm"], "gpu_name": plan.get("gpu_name"),
            "source_sha256": source, "provenance_status": provenance["status"],
            "model_sources": [{"role": item["role"], "sha256": item["sha256"]}
                              for item in model_sources],
            "io": tensors, "precision": plan.get("precision"),
            "tf32": plan.get("tf32"), "strongly_typed": plan.get("strongly_typed"),
            "optimization_level": plan.get("optimization_level"),
            "tiling_optimization_level": plan.get("tiling_optimization_level", "none"),
            "workspace_bytes": plan.get("workspace_bytes"),
            "plugins": plan.get("plugins", [])}


def _deployment(stage: Path, batch: int, sm: int = 89) -> Path:
    template = ROOT / "configs/common/trt113_runtime_template.json"
    data = json.loads(template.read_text())
    data["status"] = f"sm{sm}_trt113_b{batch}_request_isolated_numerically_experimental"
    for component, key in (("target", "tensorrt113_target_full_plan"),
                           ("draft", "tensorrt113_draft_full_plan"),
                           ("cfm", "tensorrt113_cfm_plan"),
                           ("vocoder", "tensorrt113_vocoder_plan")):
        data[key] = f"{component}/plan.json"
    output = stage / "deployment.json"
    _write_json(output, data)
    return output


def build_one(batch: int, gpu: dict, ref_audio: Path, output_root: Path,
              builder: dict | None = None, quantization: dict | None = None) -> Path:
    stage = output_root / ".staging" / uuid.uuid4().hex
    stage.mkdir(parents=True, exist_ok=False)
    try:
        for name, command in _steps(batch, stage, gpu["physical_gpu"], builder):
            _run(name, command, stage=stage, gpu=gpu["physical_gpu"])
        records = {component: _engine_record(stage, component, batch, gpu) for component in COMPONENTS}
        selected_builder = builder or builder_policy(ROOT / BUILDER_DEFAULT)
        for component, record in records.items():
            if (record["optimization_level"] != selected_builder["optimization_level"][component]
                    or record["workspace_bytes"] != selected_builder["workspace_bytes"]
                    or record["tiling_optimization_level"] != selected_builder["tiling_optimization_level"]
                    or record["tf32"] is not False or record["strongly_typed"] is not True):
                raise ValueError(f"{component} engine builder settings differ from requested policy")
        if len({value["source_sha256"] for value in records.values()}) != 1:
            raise ValueError("Component source fingerprints differ; build source changed mid-bundle")
        if len({value["trt"] for value in records.values()}) != 1:
            raise ValueError("Component TensorRT versions differ")
        deployment = _deployment(stage, batch, gpu["sm"])
        _run("route", ["scripts/validate_trt113_first_chunks.py", "--gpu", str(gpu["physical_gpu"]),
                       "--batch", str(batch), "--deployment", str(deployment),
                       "--reference", str(ref_audio), "--output", str(stage / "route_report.json")],
             stage=stage, gpu=gpu["physical_gpu"])
        route = json.loads((stage / "route_report.json").read_text())
        if route.get("status") != "passed":
            raise ValueError(f"B{batch} route validation did not pass")
        compatible_hardware = {key: gpu[key] for key in ("name", "sm", "memory_total_mib")}
        identity = {"schema": 2, "model": "indextts2", "profile": PROFILE, "batch": batch,
                    "precision_policy": "mixed_bf16_fp32", "quantization": "none",
                    "hardware": compatible_hardware, "trt": records["target"]["trt"],
                    "source_sha256": records["target"]["source_sha256"],
                    "engines": {name: record["engine_sha256"] for name, record in records.items()}}
        identity["builder"] = selected_builder
        identity["quantization_policy"] = quantization or {"schema": 1, "scheme": "none",
            "calibration": None, "scale_format": None, "qdq_graph_sha256": None}
        bundle_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
        manifest = {**identity, "bundle_id": bundle_id, "build_gpu": gpu,
                    "components": records,
                    "deployment": "deployment.json", "deployment_sha256": _digest(deployment),
                    "route_report": "route_report.json", "route_pass": True,
                    "numerical_pass": False, "numerical_status": "experimental_existing_gates_failed",
                    "quality_pass": None, "certified_for_production": False}
        manifest["files"] = {
            str(path.relative_to(stage)): _digest(path)
            for path in sorted(stage.rglob("*"))
            if path.is_file() and not path.is_relative_to(stage / "logs")
        }
        _write_json(stage / "manifest.json", manifest)
        final = output_root / f"sm{gpu['sm']}" / PROFILE / f"b{batch}" / bundle_id
        final.parent.mkdir(parents=True, exist_ok=True)
        if final.exists():
            if validate_bundle(final)["bundle_id"] != bundle_id:
                raise ValueError("Existing bundle directory has a conflicting identity")
            shutil.rmtree(stage)
            return final
        stage.rename(final)
        return final
    except Exception as exc:
        _write_json(stage / "build_failure.json", {"status": "failed", "batch": batch,
                                                    "error": f"{type(exc).__name__}: {exc}"})
        raise


def validate_bundle(root: Path) -> dict:
    root = root.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema") not in (1, 2) or manifest.get("profile") != PROFILE or manifest.get("batch") not in BATCHES:
        raise ValueError("Unsupported bundle manifest")
    identity_keys = ("schema", "model", "profile", "batch", "precision_policy",
                     "quantization", "hardware", "trt",
                     "source_sha256", "engines")
    if manifest["schema"] == 2:
        identity_keys += ("builder", "quantization_policy")
    if any(key not in manifest for key in identity_keys):
        raise ValueError("Bundle identity is incomplete")
    identity = {key: manifest[key] for key in identity_keys}
    if manifest["schema"] == 2:
        from inspark_infer.quantization.trt113 import validate_policy
        validate_policy(manifest["quantization_policy"])
        if manifest["quantization_policy"]["scheme"] != "none" or manifest["quantization"] != "none":
            raise ValueError("Quantized TensorRT bundle requires independent implementation and audit")
        builder = manifest["builder"]
        if (not isinstance(builder, dict) or set(builder) != {"schema", "optimization_level",
                "tiling_optimization_level", "workspace_bytes", "strongly_typed", "tf32"}
                or builder["schema"] != 1 or set(builder["optimization_level"]) != set(COMPONENTS)
                or any(type(level) is not int or level not in range(6)
                       for level in builder["optimization_level"].values())
                or builder["tiling_optimization_level"] not in ("none", "fast", "moderate", "full")
                or type(builder["workspace_bytes"]) is not int or builder["workspace_bytes"] < 1
                or builder["tf32"] is not False or builder["strongly_typed"] is not True):
            raise ValueError("Invalid TensorRT bundle builder identity")
    expected_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    if manifest.get("bundle_id") != expected_id:
        raise ValueError("Bundle ID does not match its content identity")
    if (manifest.get("certified_for_production") is True
            and not all(manifest.get(key) is True for key in
                        ("route_pass", "numerical_pass", "quality_pass"))):
        raise ValueError("Production certification requires route, numerical and quality passes")
    if set(manifest.get("components", {})) != set(COMPONENTS):
        raise ValueError("Bundle must contain all four components")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Bundle file inventory is missing")
    required = {manifest.get("deployment"), manifest.get("route_report")}
    for record in manifest["components"].values():
        required.update((record.get("engine"), record.get("plan")))
    if None in required or not required <= set(files):
        raise ValueError("Bundle inventory omits an engine, plan, deployment or route report")
    for name, digest in files.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or _digest(path) != digest:
            raise ValueError(f"Bundle file hash/path mismatch: {name}")
    for component, record in manifest["components"].items():
        if manifest["schema"] == 2 and (record.get("optimization_level") != builder["optimization_level"][component]
                or record.get("workspace_bytes") != builder["workspace_bytes"]
                or record.get("tiling_optimization_level") != builder["tiling_optimization_level"]
                or record.get("tf32") is not False or record.get("strongly_typed") is not True
                or record.get("trt") != manifest["trt"] or record.get("sm") != manifest["hardware"]["sm"]
                or record.get("gpu_name") != manifest["hardware"]["name"]
                or record.get("source_sha256") != manifest["source_sha256"]):
            raise ValueError(f"{component} settings or provenance differ from bundle identity")
        for key, digest_key in (("engine", "engine_sha256"), ("plan", "plan_sha256")):
            path = (root / record[key]).resolve()
            if not path.is_relative_to(root) or _digest(path) != record[digest_key]:
                raise ValueError(f"{component} {key} hash/path mismatch")
        if manifest["engines"].get(component) != record["engine_sha256"]:
            raise ValueError(f"{component} engine disagrees with bundle identity")
    deployment = (root / manifest["deployment"]).resolve()
    if not deployment.is_relative_to(root) or _digest(deployment) != manifest["deployment_sha256"]:
        raise ValueError("Bundle deployment hash/path mismatch")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True, help="Physical GPU index")
    parser.add_argument("--model", default="indextts2", choices=("indextts2",))
    parser.add_argument("--profile", default=PROFILE)
    parser.add_argument("--batches", default="1,4,8")
    parser.add_argument("--ref-audio", type=Path, help="Required for supported builds; local WAV used by route gate")
    parser.add_argument("--preflight-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output-root", type=Path, default=ROOT / "artifacts/trt113_bundles")
    parser.add_argument("--builder-config", type=Path, default=ROOT / BUILDER_DEFAULT)
    parser.add_argument("--quantization-config", type=Path, default=ROOT / QUANTIZATION_DEFAULT)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)
    try:
        batches = parse_batches(args.batches)
        from inspark_infer.quantization.trt113 import load_policy
        quantization = load_policy(args.quantization_config)
        builder = builder_policy(args.builder_config)
        gpu = gpu_info(args.gpu)
        decision = unsupported(gpu, args.profile, batches)
        if decision["reasons"]:
            print(json.dumps(decision, ensure_ascii=False, indent=2))
            return 2
        if args.preflight_only:
            print(json.dumps({"status": "supported_build_candidate", "gpu": gpu,
                              "profile": args.profile, "batches": batches}, ensure_ascii=False))
            return 0
        if args.ref_audio is None:
            raise ValueError("--ref-audio is required to validate a supported bundle")
        ref = args.ref_audio.resolve()
        output_root = args.output_root.resolve()
        if not ref.is_file() or not ref.is_relative_to(Path("/workspace")):
            raise ValueError("--ref-audio must be an existing WAV under /workspace")
        if not output_root.is_relative_to(Path("/workspace")):
            raise ValueError("Build artifacts must stay under /workspace")
        ensure_trt113_site()
        results = []
        for batch in batches:
            results.append({"batch": batch, "bundle": str(build_one(batch, gpu, ref, output_root,
                                                                       builder, quantization))})
        report = {"status": "built_route_passed_numerically_experimental", "profile": PROFILE,
                  "gpu": gpu, "bundles": results}
        if args.json_out:
            _write_json(args.json_out.resolve(), report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": f"{type(exc).__name__}: {exc}"},
                         ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
