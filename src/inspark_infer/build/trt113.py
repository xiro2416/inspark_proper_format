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
ROOT = Path(__file__).resolve().parents[3]


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
    if gpu["sm"] != 89:
        reasons.append(f"SM{gpu['sm']} has no certified TensorRT 11.3 builder in this release")
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


def _steps(batch: int, stage: Path, gpu: int) -> list[tuple[str, list[str]]]:
    def path(component: str, name: str) -> str:
        return str(stage / component / name)

    return [
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
    if digest != expected or plan.get("sm") != gpu["sm"] or not str(plan.get("trt", "")).startswith("11.3."):
        raise ValueError(f"{component} engine/hash/hardware/SDK mismatch")
    if not isinstance(provenance, dict) or provenance.get("status") != "recorded_not_audited":
        raise ValueError(f"{component} has no source-attested build provenance")
    source = provenance.get("source", {}).get("source_sha256")
    if not source:
        raise ValueError(f"{component} has no source fingerprint")
    return {"engine": str(engine.relative_to(stage)), "engine_sha256": digest,
            "plan": str(plan_path.relative_to(stage)), "plan_sha256": _digest(plan_path),
            "trt": plan["trt"], "sm": plan["sm"], "gpu_name": plan.get("gpu_name"),
            "source_sha256": source, "provenance_status": provenance["status"],
            "io": tensors, "precision": plan.get("precision")}


def _deployment(stage: Path, batch: int) -> Path:
    template = ROOT / "configs/hardware/sm89" / f"sm89_trt113_safe_b{batch}.json"
    data = json.loads(template.read_text())
    for component, key in (("target", "tensorrt113_target_full_plan"),
                           ("draft", "tensorrt113_draft_full_plan"),
                           ("cfm", "tensorrt113_cfm_plan"),
                           ("vocoder", "tensorrt113_vocoder_plan")):
        data[key] = f"{component}/plan.json"
    output = stage / "deployment.json"
    _write_json(output, data)
    return output


def build_one(batch: int, gpu: dict, ref_audio: Path, output_root: Path) -> Path:
    stage = output_root / ".staging" / uuid.uuid4().hex
    stage.mkdir(parents=True, exist_ok=False)
    try:
        for name, command in _steps(batch, stage, gpu["physical_gpu"]):
            _run(name, command, stage=stage, gpu=gpu["physical_gpu"])
        records = {component: _engine_record(stage, component, batch, gpu) for component in COMPONENTS}
        if len({value["source_sha256"] for value in records.values()}) != 1:
            raise ValueError("Component source fingerprints differ; build source changed mid-bundle")
        if len({value["trt"] for value in records.values()}) != 1:
            raise ValueError("Component TensorRT versions differ")
        deployment = _deployment(stage, batch)
        _run("route", ["scripts/validate_trt113_first_chunks.py", "--gpu", str(gpu["physical_gpu"]),
                       "--batch", str(batch), "--deployment", str(deployment),
                       "--reference", str(ref_audio), "--output", str(stage / "route_report.json")],
             stage=stage, gpu=gpu["physical_gpu"])
        route = json.loads((stage / "route_report.json").read_text())
        if route.get("status") != "passed":
            raise ValueError(f"B{batch} route validation did not pass")
        compatible_hardware = {key: gpu[key] for key in ("name", "sm", "memory_total_mib")}
        identity = {"schema": 1, "model": "indextts2", "profile": PROFILE, "batch": batch,
                    "hardware": compatible_hardware, "trt": records["target"]["trt"],
                    "source_sha256": records["target"]["source_sha256"],
                    "engines": {name: record["engine_sha256"] for name, record in records.items()}}
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
        final = output_root / "sm89" / PROFILE / f"b{batch}" / bundle_id
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
    if manifest.get("schema") != 1 or manifest.get("profile") != PROFILE or manifest.get("batch") not in BATCHES:
        raise ValueError("Unsupported bundle manifest")
    if set(manifest.get("components", {})) != set(COMPONENTS):
        raise ValueError("Bundle must contain all four components")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Bundle file inventory is missing")
    for name, digest in files.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or _digest(path) != digest:
            raise ValueError(f"Bundle file hash/path mismatch: {name}")
    for component, record in manifest["components"].items():
        for key, digest_key in (("engine", "engine_sha256"), ("plan", "plan_sha256")):
            path = (root / record[key]).resolve()
            if not path.is_relative_to(root) or _digest(path) != record[digest_key]:
                raise ValueError(f"{component} {key} hash/path mismatch")
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
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)
    try:
        batches = parse_batches(args.batches)
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
        site = Path(os.getenv("ACC_TRT113_SITE", str(ROOT / ".venv-trt113/lib/python3.11/site-packages")))
        if not site.is_dir():
            raise ValueError("TensorRT 11.3 environment missing; run scripts/bootstrap_trt113.sh")
        os.environ["ACC_TRT113_SITE"] = str(site.resolve())
        results = []
        for batch in batches:
            results.append({"batch": batch, "bundle": str(build_one(batch, gpu, ref, output_root))})
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
