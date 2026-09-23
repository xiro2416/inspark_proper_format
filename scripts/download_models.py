#!/usr/bin/env python3
"""Assemble the runtime model tree from immutable upstream revisions."""
import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def install(entry: dict, root: Path) -> None:
    target = root / entry["local"]
    if target.is_file() and sha256(target) == entry["sha256"]:
        print(f"ok       {entry['local']}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    source = Path(hf_hub_download(
        repo_id=entry["repo"], filename=entry["remote"],
        revision=entry["revision"], token=os.getenv("HF_TOKEN"),
    ))
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as temporary:
        with source.open("rb") as stream:
            shutil.copyfileobj(stream, temporary, length=16 << 20)
        temporary_path = Path(temporary.name)
    actual = sha256(temporary_path)
    if actual != entry["sha256"]:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(f"Hash mismatch for {entry['repo']}/{entry['remote']}: {actual}")
    temporary_path.replace(target)
    print(f"installed {entry['local']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, default=ROOT.parent / "models")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "configs/common/model_sources.json").read_text())
    if manifest["schema"] != 1:
        raise RuntimeError("Unsupported model-source manifest")
    if any(item["revision"] == "__CUSTOM_REVISION__" for item in manifest["files"]):
        raise RuntimeError("Release manifest has not been pinned to the custom-weight revision")
    for entry in manifest["files"]:
        install(entry, args.model_root.resolve())
    print(f"All {len(manifest['files'])} model assets verified in {args.model_root.resolve()}")


if __name__ == "__main__":
    main()

