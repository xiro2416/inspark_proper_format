"""Resolve a complete fixed-shape TRT bundle locally, from a pinned cache, or build it."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from inspark_infer.build import hf_cache, trt113
from inspark_infer.quantization.trt113 import load_policy


REGISTRY = trt113.ROOT / "configs/hardware/sm89/trt113_hf_registry.json"


def _compatible(manifest: dict, gpu: dict, batch: int, builder: dict, source: str | None,
                trt_version: str | None = None) -> bool:
    hardware = manifest.get("hardware", {})
    if (manifest.get("profile") != trt113.PROFILE or manifest.get("batch") != batch
            or manifest.get("quantization") != "none" or not str(manifest.get("trt", "")).startswith("11.3.")
            or (trt_version is not None and manifest.get("trt") != trt_version)
            or any(hardware.get(key) != gpu[key] for key in ("name", "sm", "memory_total_mib"))):
        return False
    if manifest["schema"] == 2:
        return manifest.get("builder") == builder and manifest.get("source_sha256") == source
    # Legacy SM89 bundles are reused only through the immutable, audited registry.
    entry = _registry_entry(gpu, batch, builder)
    return bool(entry and manifest.get("bundle_id") == entry["bundle_id"])


def _registry_entry(gpu: dict, batch: int, builder: dict) -> dict | None:
    registry = json.loads(REGISTRY.read_text())
    baseline = trt113.builder_policy(trt113.ROOT / trt113.BUILDER_DEFAULT)
    if (builder != baseline or registry.get("schema") != 1 or registry.get("profile") != trt113.PROFILE
            or registry.get("quantization") != "none"
            or any(registry["hardware"].get(key) != gpu[key] for key in ("name", "sm", "memory_total_mib"))):
        return None
    entry = registry["bundles"].get(str(batch))
    return {**entry, "repo_id": registry["repo_id"]} if entry else None


def _local(output_root: Path, gpu: dict, batch: int, builder: dict, source: str | None,
           trt_version: str | None = None) -> Path | None:
    parent = output_root / f"sm{gpu['sm']}" / trt113.PROFILE / f"b{batch}"
    if not parent.is_dir():
        return None
    for candidate in sorted(parent.iterdir()):
        if not candidate.is_dir():
            continue
        try:
            manifest = trt113.validate_bundle(candidate)
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue
        if _compatible(manifest, gpu, batch, builder, source, trt_version):
            return candidate
    return None


def ensure(gpu: dict, batches: tuple[int, ...], ref_audio: Path, output_root: Path,
           mode: str, builder: dict, quantization: dict, *, allow_experimental: bool,
           endpoint: str) -> dict:
    if not allow_experimental:
        raise ValueError("Current TRT bundles fail strict floating-point parity; pass --allow-experimental explicitly")
    if mode not in ("auto", "reuse-only", "build-only"):
        raise ValueError("Invalid ensure mode")
    if quantization["scheme"] != "none":
        raise ValueError("Quantized TensorRT ensure is not implemented")
    decision = trt113.unsupported(gpu, trt113.PROFILE, batches)
    if decision["reasons"]:
        raise ValueError("; ".join(decision["reasons"]))
    ref_audio = ref_audio.resolve()
    output_root = output_root.resolve()
    if not ref_audio.is_file() or not ref_audio.is_relative_to(Path("/workspace")):
        raise ValueError("--ref-audio must be an existing file under /workspace")
    if not output_root.is_relative_to(Path("/workspace")):
        raise ValueError("--output-root must be under /workspace")
    trt113.ensure_trt113_site()
    from inspark_infer.ops.tensorrt.native113 import _import_trt113
    trt_version = _import_trt113().__version__
    if not trt_version.startswith("11.3."):
        raise ValueError(f"Expected TensorRT 11.3, got {trt_version}")
    from scripts.trt113_provenance import source_identity
    source = source_identity(trt113.ROOT)["source_sha256"]
    results = []
    for batch in batches:
        if mode != "build-only":
            local = _local(output_root, gpu, batch, builder, source, trt_version)
            if local:
                results.append({"batch": batch, "origin": "local", "bundle": str(local)})
                continue
            entry = _registry_entry(gpu, batch, builder)
            if entry and json.loads(REGISTRY.read_text())["trt"] != trt_version:
                entry = None
            if entry:
                # A pinned private cache entry is authoritative. Authentication or
                # integrity failures are fatal, not permission to surprise-rebuild.
                path = f"bundles/sm{gpu['sm']}/{trt113.PROFILE}/b{batch}/{entry['bundle_id']}"
                fetched = hf_cache.fetch(entry["repo_id"], entry["revision"], path,
                                         gpu["physical_gpu"], ref_audio, output_root, endpoint)
                if fetched["bundle_id"] != entry["bundle_id"]:
                    raise ValueError("Pinned Hugging Face bundle ID mismatch")
                if not _compatible(trt113.validate_bundle(Path(fetched["bundle"])), gpu, batch,
                                   builder, source, trt_version):
                    raise ValueError("Pinned Hugging Face bundle is incompatible with this host")
                results.append({"batch": batch, "origin": "pinned_private_hf", "bundle": fetched["bundle"]})
                continue
            if mode == "reuse-only":
                raise FileNotFoundError(f"No matching local or pinned private bundle for SM{gpu['sm']} B{batch}")
        bundle = trt113.build_one(batch, gpu, ref_audio, output_root, builder, quantization)
        results.append({"batch": batch, "origin": "built", "bundle": str(bundle)})
    return {"status": "experimental_not_numerically_certified", "profile": trt113.PROFILE,
            "gpu": gpu, "bundles": results}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--model", choices=("indextts2",), default="indextts2")
    parser.add_argument("--profile", default=trt113.PROFILE)
    parser.add_argument("--batches", default="1,4,8")
    parser.add_argument("--ref-audio", type=Path)
    parser.add_argument("--output-root", type=Path, default=trt113.ROOT / "artifacts/trt113_bundles")
    parser.add_argument("--builder-config", type=Path, default=trt113.ROOT / trt113.BUILDER_DEFAULT)
    parser.add_argument("--quantization-config", type=Path, default=trt113.ROOT / trt113.QUANTIZATION_DEFAULT)
    parser.add_argument("--mode", choices=("auto", "reuse-only", "build-only"), default="auto")
    parser.add_argument("--allow-experimental", action="store_true")
    parser.add_argument("--endpoint", default="https://hf-mirror.com")
    parser.add_argument("--preflight-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        batches = trt113.parse_batches(args.batches)
        quantization = load_policy(args.quantization_config)
        builder = trt113.builder_policy(args.builder_config)
        gpu = trt113.gpu_info(args.gpu)
        decision = trt113.unsupported(gpu, args.profile, batches)
        if decision["reasons"]:
            print(json.dumps(decision, ensure_ascii=False, indent=2))
            return 2
        if args.preflight_only:
            print(json.dumps({"status": "supported_build_candidate_uncertified", "gpu": gpu,
                              "profile": args.profile, "batches": batches}, ensure_ascii=False))
            return 0
        if args.ref_audio is None:
            raise ValueError("--ref-audio is required")
        report = ensure(gpu, batches, args.ref_audio, args.output_root, args.mode,
                        builder, quantization, allow_experimental=args.allow_experimental,
                        endpoint=args.endpoint)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": f"{type(exc).__name__}: {exc}"},
                         ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
