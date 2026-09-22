from pathlib import Path
import json
import unittest
from unittest.mock import patch

from acc_infer_clear.ops.backend import validate_reference_plan
from acc_infer_clear.ops.compile import CompiledOp, CompileBank, signature
from acc_infer_clear.runtime.deployment import validate


class ReferenceBackendTests(unittest.TestCase):
    def plan(self, name="sm89_eager_fp32"):
        return json.loads((Path(__file__).resolve().parents[1]/"configs"/(name+".json")).read_text())

    def test_reference_configs_validate(self):
        for name in ("sm89_eager_fp32", "sm89_eager_bf16", "sm89_compile_bf16"):
            validate(self.plan(name))

    def test_no_custom_kernels_in_reference(self):
        plan = self.plan()
        plan["target_graphs"] = True
        with self.assertRaisesRegex(ValueError, "forbids"):
            validate(plan)

    def test_shared_rng_rejected_early(self):
        plan = self.plan()
        plan["device_round_b8"] = True
        with self.assertRaisesRegex(ValueError, "isolation"):
            validate(plan)

    def test_compile_requires_explicit_valid_components(self):
        plan = self.plan("sm89_compile_bf16")
        plan["compile_components"] = ["cfm", "cfm"]
        with self.assertRaises(ValueError):
            validate_reference_plan(plan)

    def test_no_implicit_compile_or_unknown_shape_compile(self):
        import torch
        calls = []
        def compiler(fn, **kwargs):
            calls.append(kwargs)
            return fn
        op = CompiledOp(lambda x: x+1, "test", compiler=compiler)
        bank = CompileBank({"test": op})
        x = torch.zeros(2)
        op(x)
        self.assertEqual(calls, [])
        with bank.offline_warmup():
            op(x)
        self.assertFalse(op.warming)
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0]["options"]["emulate_precision_casts"])
        self.assertFalse(calls[0]["options"]["pattern_matcher"])
        op(x)
        op(torch.zeros(3))
        self.assertEqual(op.stats()["compiled_calls"], 2)
        self.assertEqual(op.stats()["eager_calls"], 2)
        self.assertFalse(op.stats()["online_compile"])

    def test_compile_error_explicit(self):
        import torch
        def fail(*args, **kwargs):
            raise RuntimeError("compile failed")
        op = CompiledOp(lambda x: x, "test", compiler=fail)
        with self.assertRaisesRegex(RuntimeError, "compile failed"):
            with CompileBank({"test": op}).offline_warmup():
                op(torch.zeros(1))
        self.assertFalse(op.warming)
        self.assertEqual(op.stats()["failed_signatures"], 1)
        soft = CompiledOp(lambda x: x, "test", compiler=fail, error_fallback=True)
        with CompileBank({"test": soft}).offline_warmup():
            soft(torch.zeros(1))
        self.assertEqual(soft.stats()["compiled_calls"], 0)
        # Failed specialization attempts also consume the finite offline budget.
        with CompileBank({"test": soft}).offline_warmup():
            soft(torch.zeros(2))
        self.assertEqual(soft.stats()["failed_signatures"], 1)

    def test_signature_tracks_layout_dtype_and_scalars(self):
        import torch
        x = torch.zeros(2, 3)
        self.assertNotEqual(signature(x), signature(x.t()))
        self.assertNotEqual(signature(x), signature(x.bfloat16()))
        self.assertNotEqual(signature((x, None)), signature((x, 0)))


if __name__ == "__main__":
    unittest.main()
