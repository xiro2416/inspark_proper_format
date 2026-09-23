"""CPU toy-GPT2 parity for the production Target body's cache-format adapter."""
from types import SimpleNamespace
import unittest

import torch
from transformers import GPT2Config, GPT2LMHeadModel
from transformers.cache_utils import DynamicCache

from inspark_infer.models.indextts2.dspark.target import IndexTTS2TargetEngine
from inspark_infer.ops.eager.target import TargetCacheAdapter


class TargetCompileAdapterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(7)
            config = GPT2Config(n_layer=22, n_head=2, n_embd=8, n_positions=32,
                                vocab_size=13, resid_pdrop=0, embd_pdrop=0, attn_pdrop=0)
            config._attn_implementation = "sdpa"
            model = GPT2LMHeadModel(config).eval()
        cls.target = IndexTTS2TargetEngine(SimpleNamespace(inference_model=model),
                                         [1, 6, 11, 16, 21])
        cls.body = cls.target._block_forward_with_hidden_states
        cls.adapter = TargetCacheAdapter(cls.body)
        cls.prefix = torch.linspace(-1, 1, 48).reshape(2, 3, 8)
        cls.continuation = torch.linspace(-0.5, 0.5, 32).reshape(2, 2, 8)
        cls.prefill_mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.long)
        cls.decode_mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 0, 1, 1]], dtype=torch.long)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def assert_same(self, actual, expected):
        for index in (0, 2, 3):
            torch.testing.assert_close(actual[index], expected[index], rtol=0, atol=0)
        self.assertEqual(len(actual[1]), len(expected[1]))
        for got, ref in zip(actual[1], expected[1]):
            for actual_tensor, expected_tensor in zip(got, ref):
                torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)

    @torch.inference_mode()
    def test_none_prefill_retains_legacy_tuple_and_all_hidden_outputs(self):
        expected = self.body(self.prefix, None, self.prefill_mask, None)
        actual = self.adapter(self.prefix, None, self.prefill_mask, None)
        self.assertIsInstance(actual[1], tuple)
        self.assert_same(actual, expected)

    @torch.inference_mode()
    def test_tuple_decode_has_no_input_mutation_or_cross_call_cache(self):
        past = self.body(self.prefix, None, self.prefill_mask, None)[1]
        snapshot = tuple(tuple(t.clone() for t in layer) for layer in past)
        position = torch.arange(3, 5)
        expected = self.body(self.continuation, past, self.decode_mask, position)
        first = self.adapter(self.continuation, past, self.decode_mask, position)
        second = self.adapter(self.continuation, past, self.decode_mask, position)
        self.assertIsInstance(first[1], tuple)
        self.assert_same(first, expected)
        self.assert_same(second, expected)
        for layer, original in zip(past, snapshot):
            for tensor, saved in zip(layer, original):
                torch.testing.assert_close(tensor, saved, rtol=0, atol=0)
        self.assertEqual(first[1][0][0].shape[-2], 5)

    @torch.inference_mode()
    def test_existing_cache_keeps_original_in_place_api_semantics(self):
        past = self.body(self.prefix, None, self.prefill_mask, None)[1]
        reference_cache = DynamicCache.from_legacy_cache(past)
        candidate_cache = DynamicCache.from_legacy_cache(past)
        expected = self.body(self.continuation, reference_cache, self.decode_mask, None)
        actual = self.adapter(self.continuation, candidate_cache, self.decode_mask, None)
        self.assertIs(actual[1], candidate_cache)
        self.assertIs(expected[1], reference_cache)
        self.assertEqual(candidate_cache.get_seq_length(), 5)
        self.assert_same(actual, expected)

    @torch.inference_mode()
    def test_fullgraph_cpu_capture_preserves_tuple_cache_and_outputs(self):
        # backend=eager verifies Dynamo capture, not Inductor numerical fidelity
        # or GPU speed. Actual-model GPU audit is a separate required result.
        compiled = torch.compile(self.adapter, backend="eager", fullgraph=True, dynamic=False)
        expected = self.body(self.prefix, None, self.prefill_mask, None)
        actual = compiled(self.prefix, None, self.prefill_mask, None)
        self.assert_same(actual, expected)
        past = expected[1]
        expected_decode = self.body(self.continuation, past, self.decode_mask, None)
        actual_decode = compiled(self.continuation, past, self.decode_mask, None)
        self.assertIsInstance(actual_decode[1], tuple)
        self.assert_same(actual_decode, expected_decode)


if __name__ == "__main__":
    unittest.main()
