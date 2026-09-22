from copy import deepcopy
import hashlib
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from evaluate_quality_corpus import validate_paired_generation_identity, corpus_source_declaration


def pair():
    paths = [f"/workspace/offline-references/voice-{index}.wav" for index in range(9)]
    cases = [{"reference_audio": path} for path in paths]
    references = [{"path": path, "sha256": hashlib.sha256(path.encode()).hexdigest(), "bytes": 100 + index}
                  for index, path in enumerate(paths)]
    roles = {"target": ("target_checkpoint", "target_config"),
             "draft": ("draft_checkpoint", "draft_config"),
             "cfm": ("s2mel_checkpoint", "student_checkpoint", "s2mel_config"),
             "vocoder": ("vocoder_checkpoint", "vocoder_config")}
    model_provenance = {component: {"model_sources": [
        {"role": role, "sha256": hashlib.sha256(role.encode()).hexdigest()} for role in names]}
        for component, names in roles.items()}
    baseline = {"references": references, "model_provenance": model_provenance}
    return baseline, deepcopy(baseline), cases


def test_nine_reference_pair_matches_without_reopening_historical_files(monkeypatch):
    baseline, candidate, cases = pair()
    def no_filesystem(*args, **kwargs):
        raise AssertionError("historical assets must not be accessed")
    for method in ("open", "exists", "resolve", "stat"):
        monkeypatch.setattr(Path, method, no_filesystem)
    result = validate_paired_generation_identity(baseline, candidate, cases)
    assert result["same_reference_audio_identity"] and result["same_loader_checkpoint_identity"]
    assert result["reference_audio_count"] == 9
    assert not result["reference_audio_files_rehashed_by_evaluator"]


@pytest.mark.parametrize("mutation", ["hash", "bytes", "missing", "duplicate", "extra", "invalid_hash", "invalid_bytes"])
def test_reference_mismatch_or_bad_inventory_fails(mutation):
    baseline, candidate, cases = pair()
    if mutation == "hash": candidate["references"][0]["sha256"] = "0" * 64
    if mutation == "bytes": candidate["references"][0]["bytes"] += 1
    if mutation == "missing": candidate["references"].pop()
    if mutation == "duplicate": candidate["references"].append(deepcopy(candidate["references"][0]))
    if mutation == "extra": candidate["references"].append({**candidate["references"][0], "path": "/workspace/extra.wav"})
    if mutation == "invalid_hash": candidate["references"][0]["sha256"] = "not-sha256"
    if mutation == "invalid_bytes": candidate["references"][0]["bytes"] = True
    with pytest.raises(ValueError):
        validate_paired_generation_identity(baseline, candidate, cases)


def test_lexical_duplicate_reference_is_not_silently_deduplicated():
    baseline, candidate, cases = pair()
    row = deepcopy(candidate["references"][0]); row["path"] = row["path"].replace("voice-0", "./voice-0")
    candidate["references"].append(row)
    with pytest.raises(ValueError, match="Duplicate"):
        validate_paired_generation_identity(baseline, candidate, cases)


@pytest.mark.parametrize("component", ["target", "draft", "cfm", "vocoder"])
def test_original_four_component_checkpoint_comparison_is_preserved(component):
    baseline, candidate, cases = pair()
    candidate["model_provenance"][component]["model_sources"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match=component):
        validate_paired_generation_identity(baseline, candidate, cases)


def test_matching_but_incomplete_model_roles_still_fail():
    baseline, candidate, cases = pair()
    baseline["model_provenance"]["cfm"]["model_sources"].pop()
    candidate["model_provenance"]["cfm"]["model_sources"].pop()
    with pytest.raises(ValueError, match="roles"):
        validate_paired_generation_identity(baseline, candidate, cases)


def test_original_corpus_text_source_hash_remains_a_declaration():
    result = corpus_source_declaration({"source": "/workspace/no-longer-present.txt", "source_sha256": "f" * 64})
    assert result["sha256"] == "f" * 64
    assert not result["independently_verified"]
    assert "not read or rehashed" in result["scope"]
