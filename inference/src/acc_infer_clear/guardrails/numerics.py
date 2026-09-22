"""Same-input metrics for native TRT component validation commands."""
from __future__ import annotations

import math

TOLERANCES = {
    "fp32": {"atol": 1e-5, "rtol": 1e-4},
    "bf16": {"atol": 1e-2, "rtol": 1e-2},
}


def compare(expected, actual, precision):
    """Compare outputs under a declared arithmetic policy, not a storage dtype.

    BF16 matrix/convolution arithmetic may return FP32 tensors. These declared
    tolerances are not fitted to observed candidate errors or quality scores.
    """
    import torch

    policy = TOLERANCES[precision]
    result = dict(candidate_precision=precision, **policy,
                  expected_shape=list(expected.shape), actual_shape=list(actual.shape),
                  reference_dtype=str(expected.dtype), candidate_dtype=str(actual.dtype),
                  storage_dtypes_match=expected.dtype == actual.dtype,
                  total_elements=expected.numel(), candidate_elements=actual.numel(),
                  tolerance_provenance={
                      "source": 'src/acc_infer_clear/guardrails/numerics.py:TOLERANCES',
                      "selection": "declared candidate arithmetic precision",
                      "empirically_calibrated": False,
                      "storage_dtype_equality_required": False,
                      "comparison_dtype": "torch.float64",
                      "rule": "abs(actual-reference) <= atol + rtol * abs(reference)",
                  })
    result["finite_reference"] = bool(torch.isfinite(expected).all())
    result["finite_actual"] = bool(torch.isfinite(actual).all())
    result["pass_gate"] = False
    if expected.shape != actual.shape:
        result["reason"] = "shape mismatch"
        return result
    if expected.numel() == 0:
        result["reason"] = "empty tensors provide no numerical evidence"
        return result
    if not result["finite_reference"] or not result["finite_actual"]:
        result["reason"] = "non-finite input values"
        return result
    x, y = expected.double().flatten(), actual.double().flatten()
    delta = (x - y).abs()
    allowed = policy["atol"] + policy["rtol"] * x.abs()
    mismatches = int((delta > allowed).sum())
    metrics = dict(
        max_abs=float(delta.max()), mean_abs=float(delta.mean()),
        relative_l2=float(torch.linalg.vector_norm(x - y) / torch.linalg.vector_norm(x).clamp_min(1e-30)),
        cosine=float(torch.nn.functional.cosine_similarity(x, y, dim=0)),
        snr_db=float(10 * torch.log10(x.square().mean().clamp_min(1e-30)
                                    / (x - y).square().mean().clamp_min(1e-30))),
    )
    invalid_metrics = [name for name, value in metrics.items() if not math.isfinite(value)]
    # Finite inputs can still overflow reductions. Keep the failure report
    # serializable with allow_nan=False instead of losing the evidence.
    result.update({name: None if name in invalid_metrics else value
                   for name, value in metrics.items()})
    result["mismatched_elements"] = mismatches
    result["pass_gate"] = mismatches == 0 and not invalid_metrics
    if invalid_metrics:
        result["reason"] = "non-finite comparison metrics"
        result["nonfinite_metrics"] = invalid_metrics
    elif mismatches:
        result["reason"] = "tolerance exceeded"
    return result
