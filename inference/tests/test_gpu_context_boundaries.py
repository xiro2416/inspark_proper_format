"""Opt-in, single-GPU regressions for compact context writes and head limits.

Run explicitly with ACC_RUN_GPU_TESTS=1 and one CUDA_VISIBLE_DEVICES value.
Collection and the default skipped run do not import torch or initialize CUDA.
"""
import os
import unittest


_VISIBLE = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
_GPU_OPT_IN = (
    os.environ.get("ACC_RUN_GPU_TESTS") == "1"
    and bool(_VISIBLE)
    and "," not in _VISIBLE
    and _VISIBLE not in ("-1", "all")
)


@unittest.skipUnless(
    _GPU_OPT_IN,
    "Requires ACC_RUN_GPU_TESTS=1 and exactly one CUDA_VISIBLE_DEVICES value",
)
class ContextBoundaryGpuTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        if torch.cuda.device_count() != 1:
            raise RuntimeError("GPU boundary tests require exactly one visible CUDA device")
        cls.torch = torch

    def _check_scatter(self, capacity):
        from acc_infer_clear.ops.triton.context_scatter import scatter

        torch = self.torch
        layers, heads, total, dim, slots = 2, 3, 40, 4, 4
        keys = torch.arange(
            layers * heads * total * dim, dtype=torch.float32, device="cuda:0"
        ).reshape(layers, 1, heads, total, dim)
        values = keys + 10000
        pool = torch.full(
            (layers, 2, slots, heads, capacity, dim),
            -999999.0,
            device="cuda:0",
        )
        expected = pool.clone()
        source_offsets = [5, 19, 29]
        lengths = [8, 0, 3]
        slot_ids = [1, 2, 0]
        destinations = [126, 17, 4]
        for source, length, slot, destination in zip(
            source_offsets, lengths, slot_ids, destinations
        ):
            count = min(length, capacity - destination)
            if count:
                expected[:, 0, slot, :, destination:destination + count] = (
                    keys[:, 0, :, source:source + count]
                )
                expected[:, 1, slot, :, destination:destination + count] = (
                    values[:, 0, :, source:source + count]
                )
        metadata = [
            torch.tensor(data, device="cuda:0", dtype=torch.int32)
            for data in (source_offsets, lengths, slot_ids, destinations)
        ]
        scatter(keys, values, pool, *metadata, max_commit=8)
        torch.cuda.synchronize()
        # Check the entire arena, including untouched heads, the zero-length
        # row's slot, and the slot following the one whose write crosses K128.
        torch.testing.assert_close(pool, expected, atol=0, rtol=0)

    def test_canonical_k2048_writes_all_eight_tokens_at_126(self):
        self._check_scatter(2048)

    def test_compact_k128_clips_to_two_tokens_without_neighbour_corruption(self):
        self._check_scatter(128)

    def _status(self, ready_values, failures=7, initial=7, past=None, draft=None):
        from acc_infer_clear.ops.triton.device_commit import status

        torch = self.torch
        ready = torch.tensor(ready_values, dtype=torch.bool, device="cuda:0")
        failed = torch.tensor(failures, dtype=torch.int32, device="cuda:0")
        output = torch.full((), -1, dtype=torch.int32, device="cuda:0")
        if past is None and draft is None:
            status(ready, failed, output, initial)
        else:
            past_tensor = torch.tensor(past, dtype=torch.int32, device="cuda:0")
            draft_tensor = torch.tensor(draft, dtype=torch.int32, device="cuda:0")
            status(ready, failed, output, initial, past_tensor, draft_tensor)
        return int(output.item())

    def test_status_optional_lengths_preserve_ready_and_failure_bits(self):
        # Three rows exercise masked padding in the power-of-two reduction.
        cases = (
            ([False, True, True], 7, 0),
            ([True, True, True], 7, 1),
            ([False, True, True], 8, 2),
            ([True, True, True], 8, 3),
        )
        for ready, failures, expected in cases:
            with self.subTest(ready=ready, failures=failures):
                self.assertEqual(self._status(ready, failures=failures), expected)

    def test_status_capacity_checks_all_eight_positions_in_both_caches(self):
        cases = (
            ([False, False, False], 7, [120, 120, 120], [120, 120, 120], 0),
            ([False, False, False], 7, [120, 121, 120], [120, 120, 120], 4),
            ([False, False, False], 7, [120, 120, 120], [120, 121, 120], 4),
            ([True, True, True], 7, [120, 120, 120], [120, 120, 120], 1),
            ([True, True, True], 7, [121, 120, 120], [120, 120, 120], 5),
            ([False, False, False], 8, [121, 120, 120], [120, 120, 120], 6),
            ([True, True, True], 8, [120, 120, 120], [120, 121, 120], 7),
            # Even an already-ready row is evaluated by a fixed-batch graph.
            ([True, False, False], 7, [121, 120, 120], [120, 120, 120], 4),
        )
        for ready, failures, past, draft, expected in cases:
            with self.subTest(ready=ready, failures=failures, past=past, draft=draft):
                self.assertEqual(
                    self._status(ready, failures=failures, past=past, draft=draft),
                    expected,
                )


if __name__ == "__main__":
    unittest.main()
