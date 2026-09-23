"""Explicit private Hugging Face cache for complete, hash-bound TRT bundles."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import uuid

from inspark_infer.build.trt113 import (
    ROOT, PROFILE, _run, ensure_trt113_site, gpu_info, validate_bundle,
)


def _repo_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError("--repo-id must be a Hugging Face namespace/model-repository name")
    return value


def _remote_path(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not str(path).startswith("bundles/"):
        raise ValueError("--bundle-path must stay under bundles/ without traversal")
    return str(path)


def _token() -> str:
    if os.getenv("HF_HUB_OFFLINE") == "1":
        raise ValueError("Set HF_HUB_OFFLINE=0 for private Hugging Face publish/fetch")
    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("Set HF_TOKEN in the environment; never put it in command arguments")
    return token


def _attestation(path: Path, manifest: dict) -> dict:
    data = json.loads(path.read_text())
    if (data.get("schema") != 1 or data.get("reviewed") is not True
            or data.get("redistribution_permitted") is not True):
        raise ValueError("Distribution attestation must explicitly approve redistribution")
    source_manifest = json.loads((ROOT / "configs/common/model_sources.json").read_text())
    by_hash = {item["sha256"]: item["repo"] for item in source_manifest["files"]}
    embedded = {item["sha256"] for component in manifest["components"].values()
                for item in component["model_sources"]}
    unknown = embedded - by_hash.keys()
    if unknown:
        raise ValueError(f"Embedded weight sources are not pinned: {sorted(unknown)}")
    required = {by_hash[digest] for digest in embedded}
    approvals = data.get("sources")
    if not isinstance(approvals, dict) or not required <= set(approvals):
        raise ValueError("Distribution attestation must cover every source embedded in this bundle")
    for repo in required:
        value = approvals[repo]
        if not isinstance(value, dict) or value.get("permitted") is not True or not value.get("evidence"):
            raise ValueError(f"Distribution attestation is incomplete for {repo}")
    return data


def publish(bundle: Path, repo_id: str, attestation: Path) -> dict:
    from huggingface_hub import CommitOperationAdd, HfApi
    from huggingface_hub.utils import RepositoryNotFoundError

    token = _token()
    repo_id = _repo_id(repo_id)
    root = bundle.resolve()
    if not root.is_relative_to(Path("/workspace")):
        raise ValueError("Bundle must be under /workspace")
    manifest = validate_bundle(root)
    attestation = attestation.resolve()
    if not attestation.is_relative_to(Path("/workspace")):
        raise ValueError("Distribution attestation must be under /workspace")
    _attestation(attestation, manifest)
    api = HfApi(endpoint="https://huggingface.co", token=token)
    try:
        info = api.model_info(repo_id, token=token)
    except RepositoryNotFoundError:
        api.create_repo(repo_id, private=True, repo_type="model", token=token)
        info = api.model_info(repo_id, token=token)
    if info.private is not True:
        raise ValueError("Refusing to publish TensorRT engines to a public repository")
    prefix = f"bundles/sm{manifest['hardware']['sm']}/{PROFILE}/b{manifest['batch']}/{manifest['bundle_id']}"
    operations = [CommitOperationAdd(path_in_repo=f"{prefix}/{name}", path_or_fileobj=root / name)
                  for name in sorted(manifest["files"])]
    operations.extend((
        CommitOperationAdd(path_in_repo=f"{prefix}/manifest.json", path_or_fileobj=root / "manifest.json"),
        CommitOperationAdd(path_in_repo=f"{prefix}/distribution_attestation.json",
                           path_or_fileobj=attestation),
        CommitOperationAdd(path_in_repo=f"{prefix}/LICENSE", path_or_fileobj=ROOT / "LICENSE"),
        CommitOperationAdd(path_in_repo=f"{prefix}/THIRD_PARTY_NOTICES.md",
                           path_or_fileobj=ROOT / "THIRD_PARTY_NOTICES.md"),
        CommitOperationAdd(path_in_repo=f"{prefix}/licenses/BigVGAN.txt",
                           path_or_fileobj=ROOT / "licenses/BigVGAN.txt"),
    ))
    commit = api.create_commit(repo_id=repo_id, repo_type="model", token=token,
                               operations=operations,
                               commit_message=f"TRT11.3 SM{manifest['hardware']['sm']} B{manifest['batch']} {manifest['bundle_id']}")
    return {"status": "published_private", "repo_id": repo_id, "revision": commit.oid,
            "bundle_path": prefix, "bundle_id": manifest["bundle_id"]}


def fetch(repo_id: str, revision: str, bundle_path: str, gpu: int,
          ref_audio: Path, output_root: Path, endpoint: str) -> dict:
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    token = _token()
    repo_id = _repo_id(repo_id)
    bundle_path = _remote_path(bundle_path)
    if not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise ValueError("--revision must be an immutable 40-character Hub commit SHA")
    hardware = gpu_info(gpu)
    ref_audio = ref_audio.resolve()
    output_root = output_root.resolve()
    if (not ref_audio.is_file() or not ref_audio.is_relative_to(Path("/workspace"))
            or not output_root.is_relative_to(Path("/workspace"))):
        raise ValueError("Reference audio and output root must be under /workspace")
    ensure_trt113_site()
    api = HfApi(endpoint="https://huggingface.co", token=token)
    info = api.model_info(repo_id, revision=revision, token=token)
    if info.private is not True:
        raise ValueError("Refusing to fetch engines from a public repository through private-cache flow")
    cache_dir = ROOT / ".cache/huggingface"
    selected_endpoint = endpoint
    def download(name: str) -> Path:
        nonlocal selected_endpoint
        try:
            return Path(hf_hub_download(repo_id=repo_id, repo_type="model", revision=revision,
                                        filename=f"{bundle_path}/{name}", token=token,
                                        cache_dir=cache_dir, endpoint=selected_endpoint))
        except LocalEntryNotFoundError:
            if selected_endpoint.rstrip("/") != "https://hf-mirror.com":
                raise
            # The mirror may not expose authenticated private-repository
            # metadata. Retry only that failure against the canonical Hub;
            # keep the immutable revision and digest gate unchanged.
            selected_endpoint = "https://huggingface.co"
            return Path(hf_hub_download(repo_id=repo_id, repo_type="model", revision=revision,
                                        filename=f"{bundle_path}/{name}", token=token,
                                        cache_dir=cache_dir, endpoint=selected_endpoint))

    manifest = json.loads(download("manifest.json").read_text())
    if manifest.get("schema") not in (1, 2) or manifest.get("profile") != PROFILE:
        raise ValueError("Remote TensorRT bundle schema/profile unsupported")
    expected_remote = f"bundles/sm{manifest.get('hardware', {}).get('sm')}/{PROFILE}/b{manifest.get('batch')}/{manifest.get('bundle_id')}"
    if bundle_path != expected_remote:
        raise ValueError("Remote bundle path differs from manifest identity")
    if manifest.get("hardware", {}).get("sm") != hardware["sm"]:
        raise ValueError("Remote bundle SM differs from the target GPU")
    if manifest.get("hardware", {}).get("name") != hardware["name"]:
        raise ValueError("Remote bundle GPU model differs; rebuild locally for safe tactics")
    if manifest.get("hardware", {}).get("memory_total_mib") != hardware["memory_total_mib"]:
        raise ValueError("Remote bundle GPU memory class differs; rebuild locally")
    _attestation(download("distribution_attestation.json"), manifest)
    stage = output_root / ".staging" / uuid.uuid4().hex
    stage.mkdir(parents=True, exist_ok=False)
    shutil.copy2(download("manifest.json"), stage / "manifest.json")
    for notice in ("LICENSE", "THIRD_PARTY_NOTICES.md", "distribution_attestation.json",
                   "licenses/BigVGAN.txt"):
        destination = stage / notice
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(download(notice), destination)
    for name in manifest.get("files", {}):
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Remote bundle inventory contains an unsafe path")
        target = stage.joinpath(*relative.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(download(name), target)
    validated = validate_bundle(stage)
    if validated["bundle_id"] != manifest["bundle_id"]:
        raise ValueError("Remote bundle identity mismatch")
    _run("route_local", ["scripts/validate_trt113_first_chunks.py", "--gpu", str(gpu),
                         "--batch", str(manifest["batch"]),
                         "--deployment", str(stage / manifest["deployment"]),
                         "--reference", str(ref_audio),
                         "--output", str(stage / "route_report_local.json")], stage=stage, gpu=gpu)
    if json.loads((stage / "route_report_local.json").read_text()).get("status") != "passed":
        raise ValueError("Downloaded bundle failed the target GPU route gate")
    final = output_root / f"sm{hardware['sm']}" / PROFILE / f"b{manifest['batch']}" / manifest["bundle_id"]
    final.parent.mkdir(parents=True, exist_ok=True)
    if final.exists():
        validate_bundle(final)
        shutil.rmtree(stage)
    else:
        stage.rename(final)
    return {"status": "fetched_route_passed_numerically_experimental", "bundle": str(final),
            "repo_id": repo_id, "revision": revision, "bundle_id": manifest["bundle_id"],
            "download_endpoint": selected_endpoint}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    p_publish = sub.add_parser("publish")
    p_publish.add_argument("--bundle", type=Path, required=True)
    p_publish.add_argument("--repo-id", required=True)
    p_publish.add_argument("--attestation", type=Path, required=True)
    p_fetch = sub.add_parser("fetch")
    p_fetch.add_argument("--repo-id", required=True)
    p_fetch.add_argument("--revision", required=True)
    p_fetch.add_argument("--bundle-path", required=True)
    p_fetch.add_argument("--gpu", type=int, required=True)
    p_fetch.add_argument("--ref-audio", type=Path, required=True)
    p_fetch.add_argument("--output-root", type=Path, default=ROOT / "artifacts/trt113_bundles")
    p_fetch.add_argument("--endpoint", default=os.getenv("HF_ENDPOINT", "https://hf-mirror.com"))
    args = parser.parse_args(argv)
    try:
        result = (publish(args.bundle, args.repo_id, args.attestation)
                  if args.operation == "publish" else
                  fetch(args.repo_id, args.revision, args.bundle_path, args.gpu,
                        args.ref_audio, args.output_root, args.endpoint))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": f"{type(exc).__name__}: {exc}"},
                         ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
