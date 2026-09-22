"""The same controlled-input RNG audit can be run separately on one GPU."""
import importlib.util
from pathlib import Path
import unittest

import torch

path=Path(__file__).resolve().parents[1]/'scripts/validate_request_rng_isolation.py'
spec=importlib.util.spec_from_file_location('request_rng_isolation',path)
rng=importlib.util.module_from_spec(spec);spec.loader.exec_module(rng)


class RequestRNG(unittest.TestCase):
    def test_real_sampling_adapters_preserve_request_stream_under_batch_and_churn(self):
        report=rng.audit('cpu',rounds=2,seed=2026)
        self.assertTrue(report['passed'])
        self.assertTrue(report['shared_rng_negative_control_detected'])
        self.assertTrue(report['cancel_recreate_same_seed_exact'])
        self.assertGreater(report['baseline']['residual_fallbacks'],0)
        self.assertTrue(all(row['random_stream_exact'] for row in report['scenarios']))
        self.assertFalse(torch.cuda.is_initialized())


if __name__=='__main__':unittest.main()
