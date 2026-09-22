import json
from pathlib import Path
import sys

import numpy as np
import pytest
import soundfile as sf

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from evaluate_quality_corpus import load_generation
from acc_infer_clear.guardrails.snapshots import file_sha256


def evidence(tmp_path):
    case = {"id": "sm89-q000", "seed": 113, "text": "测试。", "reference_audio": "/workspace/reference.wav",
            "emotion": [0.] * 8}
    audio = tmp_path / "sm89-q000.wav"
    sf.write(audio, np.zeros(32, dtype=np.int16), 22050, subtype="PCM_16")
    summary = {"status": "completed", "corpus": {"sha256": "corpus-hash"}}
    row = {key: case[key] for key in ("id", "seed", "text", "reference_audio")}
    row.update(complete=True, eos=True, samples=32, sample_rate=22050, sha256=file_sha256(audio))
    (tmp_path / "generation_summary.json").write_text(json.dumps(summary))
    (tmp_path / "generation.jsonl").write_text(json.dumps(row) + "\n")
    return case, row


def test_load_generation_verifies_full_eos_audio_and_corpus_fields(tmp_path):
    case, row = evidence(tmp_path)
    result = load_generation(tmp_path, [case], "corpus-hash")
    assert result["validation"]["audio_hashes_verified"]
    assert "not independently observed" in result["validation"]["emotion_evidence"]


@pytest.mark.parametrize("field,value", [("seed", 0), ("seed", True), ("text", "改文"),
    ("reference_audio", "/workspace/another.wav"), ("samples", 31), ("sample_rate", 16000),
    ("emotion", [1.] * 8), ("complete", False), ("eos", False), ("sha256", "wrong")])
def test_generation_evidence_rejects_mismatched_rows(tmp_path, field, value):
    case, row = evidence(tmp_path)
    row[field] = value
    (tmp_path / "generation.jsonl").write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError):
        load_generation(tmp_path, [case], "corpus-hash")


def test_generation_evidence_rejects_changed_corpus_missing_and_duplicate(tmp_path):
    case, row = evidence(tmp_path)
    with pytest.raises(ValueError, match="corpus"):
        load_generation(tmp_path, [case], "different-corpus")
    for lines in ("", (json.dumps(row) + "\n") * 2):
        (tmp_path / "generation.jsonl").write_text(lines)
        with pytest.raises(ValueError, match="exactly once"):
            load_generation(tmp_path, [case], "corpus-hash")
