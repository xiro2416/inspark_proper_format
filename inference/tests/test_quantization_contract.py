"""CPU contracts for offline conversion and unchanged BF16 rounding boundaries."""
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest

import torch
from transformers.pytorch_utils import Conv1D

from acc_infer_clear.ops.eager.matrix import MatrixConv, MatrixLinear
from acc_infer_clear.quantization import precision
from acc_infer_clear.quantization.weights import pack_weight


ROOT = Path(__file__).resolve().parents[1]


class QuantizationContractTest(unittest.TestCase):
    def test_compatibility_exports_preserve_class_identity(self):
        self.assertIs(precision.MatrixLinear, MatrixLinear)
        self.assertIs(precision.MatrixConv, MatrixConv)

    def test_bf16_linear_preserves_input_weight_output_rounding(self):
        for source in (torch.nn.Linear(5, 3), Conv1D(3, 5)):
            prepared = MatrixLinear(source, "bf16", None, {})
            x = torch.linspace(-2, 3, 20).reshape(2, 2, 5)
            weight = source.weight.t() if isinstance(source, Conv1D) else source.weight
            expected = torch.nn.functional.linear(
                x.bfloat16(), weight.bfloat16(), source.bias.bfloat16()).to(x.dtype)
            torch.testing.assert_close(prepared(x), expected, rtol=0, atol=0)
            self.assertEqual(prepared.weight.dtype, torch.bfloat16)
            self.assertEqual(list(prepared.parameters()), [])

    def test_bf16_convolutions_preserve_rounding_and_layout(self):
        for transpose in (False, True):
            source = (torch.nn.ConvTranspose1d(2, 3, 3, stride=2, padding=1, output_padding=1)
                      if transpose else torch.nn.Conv1d(2, 3, 3, stride=2, padding=1))
            prepared = MatrixConv(source, "bf16", None, {})
            x = torch.linspace(-2, 3, 28).reshape(2, 2, 7)
            args = (x.bfloat16(), source.weight.bfloat16(), source.bias.bfloat16(),
                    source.stride, source.padding)
            expected = (torch.nn.functional.conv_transpose1d(
                *args, source.output_padding, source.groups, source.dilation)
                if transpose else torch.nn.functional.conv1d(
                    *args, source.dilation, source.groups)).to(x.dtype)
            torch.testing.assert_close(prepared(x), expected, rtol=0, atol=0)

    def test_fp8_pack_layout_and_row_scales(self):
        weight = torch.arange(35 * 7, dtype=torch.float32).reshape(35, 7) / 100
        weight[0].zero_()
        packed, scales = pack_weight(weight)
        expected_scales = (weight.abs().amax(1) / 448).clamp_min(1e-12)
        expected = (weight / expected_scales[:, None]).to(torch.float8_e4m3fn).t()
        self.assertEqual(packed.shape, (7, 64))
        self.assertEqual(packed.dtype, torch.float8_e4m3fn)
        torch.testing.assert_close(scales, expected_scales, rtol=0, atol=0)
        torch.testing.assert_close(packed[:, :35].float(), expected.float(), rtol=0, atol=0)
        self.assertEqual(torch.count_nonzero(packed[:, 35:].float()).item(), 0)

    def test_bf16_bias_unfusion_is_a_documented_rounding_pitfall(self):
        # This is a counterexample to an algebraic rewrite, NOT a compiled
        # backend pass/fail audit. 16.125**2 = 260.015625 exactly: fused addmm
        # subtracts 260 before BF16 output rounding, whereas BF16 mm stores 260.
        x = torch.tensor([[16.125]], dtype=torch.bfloat16)
        weight = x.clone()
        bias = torch.tensor([-260.0], dtype=torch.bfloat16)
        eager = torch.nn.functional.linear(x, weight, bias).float()
        unfused = (x @ weight.t()).float() + bias.float()
        torch.testing.assert_close(eager, torch.tensor([[0.015625]]), rtol=0, atol=0)
        torch.testing.assert_close(unfused, torch.zeros_like(unfused), rtol=0, atol=0)
        self.assertFalse(torch.allclose(eager, unfused, atol=0.01, rtol=0.01))

    def test_unsupported_fp8_hardware_fails_before_kernel_import(self):
        caps = SimpleNamespace(native_fp8=False)
        with self.assertRaisesRegex(ValueError, "Native FP8 unavailable"):
            MatrixLinear(torch.nn.Linear(5, 3), "fp8", caps, {})
        with self.assertRaisesRegex(ValueError, "Native FP8 unavailable"):
            MatrixConv(torch.nn.Conv1d(2, 3, 3), "fp8", caps, {})

    def test_bf16_import_and_forward_do_not_load_triton_kernels(self):
        code = """
import sys
import torch
from acc_infer_clear.quantization.precision import MatrixLinear, MatrixConv
MatrixLinear(torch.nn.Linear(5, 3), 'bf16', None, {})(torch.ones(2, 5))
MatrixConv(torch.nn.Conv1d(2, 3, 3), 'bf16', None, {})(torch.ones(2, 2, 7))
assert not any(name.startswith('acc_infer_clear.ops.triton.') for name in sys.modules)
"""
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=environment,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
