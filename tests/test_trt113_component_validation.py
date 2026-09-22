"""CPU-only tests for numerical gates; native engine execution is GPU validation."""
import importlib.util
import json
from pathlib import Path
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("trt113_validation", ROOT / "scripts/trt113_validation.py")
METRICS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(METRICS)


class ComponentValidationTest(unittest.TestCase):
    def test_explicit_precision_policy(self):
        self.assertEqual(METRICS.TOLERANCES["fp32"], {"atol": 1e-5, "rtol": 1e-4})
        self.assertEqual(METRICS.TOLERANCES["bf16"], {"atol": 1e-2, "rtol": 1e-2})
        reference = torch.tensor([1.0])
        candidate = torch.tensor([1.005])
        self.assertTrue(METRICS.compare(reference, candidate, "bf16")["pass_gate"])
        self.assertFalse(METRICS.compare(reference, candidate, "fp32")["pass_gate"])

    def test_near_zero_uses_absolute_tolerance(self):
        result = METRICS.compare(torch.zeros(2), torch.tensor([0.009, 0.011]), "bf16")
        self.assertFalse(result["pass_gate"])
        self.assertEqual(result["mismatched_elements"], 1)

    def test_nonfinite_and_shape_mismatch_fail_closed(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            for reference, candidate in (
                (torch.ones(1), torch.tensor([value])),
                (torch.tensor([value]), torch.ones(1)),
            ):
                with self.subTest(reference=reference, candidate=candidate):
                    result = METRICS.compare(reference, candidate, "bf16")
                    self.assertFalse(result["pass_gate"])
                    self.assertEqual(result["reason"], "non-finite input values")
                    self.assertFalse(result["finite_reference"] and result["finite_actual"])
                    json.dumps(result, allow_nan=False)
        for candidate in (torch.ones(2), torch.ones(1, 1)):
            result = METRICS.compare(torch.ones(1), candidate, "bf16")
            self.assertFalse(result["pass_gate"])
            self.assertEqual(result["reason"], "shape mismatch")
            self.assertEqual(result["actual_shape"], list(candidate.shape))

    def test_empty_tensors_fail_without_reduction_errors(self):
        for shape in ((0,), (2, 0, 3)):
            with self.subTest(shape=shape):
                result = METRICS.compare(torch.empty(shape), torch.empty(shape), "bf16")
                self.assertFalse(result["pass_gate"])
                self.assertEqual(result["reason"], "empty tensors provide no numerical evidence")
                self.assertEqual(result["total_elements"], 0)
                self.assertEqual(result["candidate_elements"], 0)
                json.dumps(result, allow_nan=False)

    def test_storage_dtypes_are_reported_without_requiring_equality(self):
        reference = torch.tensor([1.0, -2.0], dtype=torch.float32)
        for dtype in (torch.bfloat16, torch.float32):
            with self.subTest(candidate_dtype=dtype):
                result = METRICS.compare(reference, reference.to(dtype), "bf16")
                self.assertTrue(result["pass_gate"])
                self.assertEqual(result["reference_dtype"], "torch.float32")
                self.assertEqual(result["candidate_dtype"], str(dtype))
                self.assertEqual(result["storage_dtypes_match"], dtype == torch.float32)
                self.assertEqual(result["candidate_precision"], "bf16")
                policy = result["tolerance_provenance"]
                self.assertFalse(policy["storage_dtype_equality_required"])
                self.assertFalse(policy["empirically_calibrated"])
                self.assertEqual(policy["selection"], "declared candidate arithmetic precision")
                self.assertIn("TOLERANCES", policy["source"])

    def test_relative_tolerance_remains_bound_to_reference(self):
        result = METRICS.compare(
            torch.tensor([100.0, 100.0], dtype=torch.float64),
            torch.tensor([101.005, 101.015], dtype=torch.float64),
            "bf16",
        )
        self.assertFalse(result["pass_gate"])
        self.assertEqual(result["mismatched_elements"], 1)
        self.assertEqual((result["atol"], result["rtol"]), (1e-2, 1e-2))

    def test_reduction_overflow_fails_with_serializable_evidence(self):
        reference = torch.tensor([1e308], dtype=torch.float64)
        result = METRICS.compare(reference, reference.clone(), "fp32")
        self.assertFalse(result["pass_gate"])
        self.assertEqual(result["reason"], "non-finite comparison metrics")
        self.assertTrue(result["nonfinite_metrics"])
        json.dumps(result, allow_nan=False)

    def test_identity_passes_and_records_complete_metrics(self):
        result = METRICS.compare(torch.tensor([0.0, 1.0, -2.0]), torch.tensor([0.0, 1.0, -2.0]), "fp32")
        self.assertTrue(result["pass_gate"])
        self.assertEqual(result["max_abs"], 0)
        self.assertEqual(result["relative_l2"], 0)
        self.assertEqual(result["mismatched_elements"], 0)


if __name__ == "__main__":
    unittest.main()
