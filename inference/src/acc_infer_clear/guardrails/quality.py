"""Coverage/finite checks and paired corpus quality policy, separate from numerics."""
from __future__ import annotations

import math


POLICY = {"utmos_relative_drop": .03, "cer_absolute_increase": .02}


def aggregate(rows, expected_ids):
    if not expected_ids or len(set(expected_ids)) != len(expected_ids):
        raise ValueError("Expected case IDs must be nonempty and unique")
    by_id = {row["id"]: row for row in rows}
    if len(by_id) != len(rows) or set(by_id) != set(expected_ids):
        raise ValueError("Quality rows must cover every expected case exactly once")
    errors = characters = 0
    for row in rows:
        if not math.isfinite(row["utmos"]) or not math.isfinite(row["cer"]):
            raise ValueError("Non-finite quality metric")
        if row["reference_characters"] <= 0:
            raise ValueError("Empty normalized reference")
        edits = sum(row[key] for key in ("substitutions", "deletions", "insertions"))
        if any(type(row[key]) is not int or row[key] < 0 for key in (
                "substitutions", "deletions", "insertions", "reference_characters")):
            raise ValueError("Invalid character edit counts")
        if not math.isclose(row["cer"], edits / row["reference_characters"], abs_tol=1e-12):
            raise ValueError("CER does not match character counts")
        errors += edits; characters += row["reference_characters"]
    return {"cases": len(rows), "utmos": sum(row["utmos"] for row in rows) / len(rows),
            "cer": errors / characters, "mean_utterance_cer": sum(row["cer"] for row in rows) / len(rows),
            "total_edit_operations": errors, "total_reference_characters": characters,
            "cer_aggregation": "sum(S+D+I) / sum(reference characters), not mean utterance CER"}


def paired_quality(baseline_rows, candidate_rows, expected_ids):
    baseline, candidate = aggregate(baseline_rows, expected_ids), aggregate(candidate_rows, expected_ids)
    if baseline["utmos"] <= 0:
        raise ValueError("Positive baseline UTMOS is required for the relative-drop gate")
    if baseline["total_reference_characters"] != candidate["total_reference_characters"]:
        raise ValueError("Baseline/candidate normalized reference coverage differs")
    baseline_by_id = {row["id"]: row for row in baseline_rows}
    candidate_by_id = {row["id"]: row for row in candidate_rows}
    paired = []
    for case_id in expected_ids:
        a, b = baseline_by_id[case_id], candidate_by_id[case_id]
        if a["reference_normalized"] != b["reference_normalized"]:
            raise ValueError("Paired normalized references differ")
        paired.append({"id": case_id, "utmos_delta": b["utmos"] - a["utmos"],
                       "cer_delta": b["cer"] - a["cer"]})
    drop = 1 - candidate["utmos"] / baseline["utmos"]
    increase = candidate["cer"] - baseline["cer"]
    checks = {"utmos": {"relative_drop": drop, "limit": POLICY["utmos_relative_drop"],
                         "pass_gate": drop <= POLICY["utmos_relative_drop"]},
              "cer": {"absolute_increase": increase, "limit": POLICY["cer_absolute_increase"],
                       "pass_gate": increase <= POLICY["cer_absolute_increase"]}}
    return {"baseline": baseline, "candidate": candidate, "paired": paired,
            "policy": POLICY, "checks": checks,
            "pass_gate": all(check["pass_gate"] for check in checks.values())}
