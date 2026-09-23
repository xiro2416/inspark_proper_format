from copy import deepcopy
import pytest
from inspark_infer.guardrails.quality import aggregate, paired_quality


def row(name="a", characters=2, errors=1):
    return {"id": name, "utmos": 3., "cer": errors / characters,
            "reference_characters": characters, "reference_normalized": "x" * characters,
            "substitutions": errors, "deletions": 0, "insertions": 0}


def test_quality_uses_corpus_character_denominator():
    rows = [row(), row("b", 8, 0)]
    result = aggregate(rows, ["a", "b"])
    assert result["cer"] == .1
    assert result["mean_utterance_cer"] == .25
    assert paired_quality(rows, deepcopy(rows), ["a", "b"])["pass_gate"]


def test_missing_duplicate_nonfinite_and_empty_quality_fail():
    for rows in ([], [row(), row()], [dict(row(), utmos=float("nan"))],
                 [dict(row(), reference_characters=0)]):
        with pytest.raises(ValueError):
            aggregate(rows, ["a"])


def test_quality_regression_gate_not_relaxed():
    baseline = [row(errors=0)]
    assert not paired_quality(baseline, [dict(row(errors=0), utmos=2.8)], ["a"])["pass_gate"]
    assert not paired_quality(baseline, [row(errors=1)], ["a"])["pass_gate"]


def test_cer_can_exceed_one_but_cannot_disagree_with_counts():
    assert aggregate([row(errors=3)], ["a"])["cer"] == 1.5
    with pytest.raises(ValueError, match="counts"):
        aggregate([dict(row(), cer=0)], ["a"])
