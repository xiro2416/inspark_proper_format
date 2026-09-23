"""CPU checks for replay fixtures and fail-closed AR audit evidence."""
from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("validate_trt113_ar", ROOT / "scripts/validate_trt113_ar.py")
AUDIT = importlib.util.module_from_spec(SPEC)
with patch.object(sys, "path", [str(ROOT / "scripts"), *sys.path]):
    SPEC.loader.exec_module(AUDIT)


def fixture():
    target_model = SimpleNamespace(embeddings=torch.nn.Embedding(32, 2),
                                   text_pos_embedding=SimpleNamespace(emb=torch.nn.Embedding(32, 2)))
    engine = SimpleNamespace(rt=SimpleNamespace(engine=SimpleNamespace(target=SimpleNamespace(model=target_model))))
    rows = []
    for index in range(2):
        key = torch.arange(6, dtype=torch.float32).reshape(1, 1, 3, 2) + index * 10
        value = key + 100
        draft_key = torch.arange(8, dtype=torch.float32).reshape(1, 1, 4, 2) + index * 20
        cache = SimpleNamespace(length=4, keys=[draft_key], values=[draft_key + 200])
        rows.append(SimpleNamespace(past_length=3, mel_length=2, kv=((key, value),), cache=cache,
                                    codes=[torch.tensor([9 + index])], mask=torch.tensor([[1, 0, 1]])))
    return engine, rows


class ArValidationTest(unittest.TestCase):
    def test_capacity_rejects_bad_or_nonuniform_prefixes(self):
        self.assertEqual(AUDIT.uniform_length([120, 120], "Target"), 120)
        for values in ([], [0], [121], [3, 4]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                AUDIT.uniform_length(values, "Target")

    def test_replay_uses_real_prefix_and_detaches_request_storage(self):
        engine, rows = fixture()
        result = AUDIT.replay_inputs(engine, rows)
        self.assertEqual(tuple(result["target_cache"].shape), (1, 2, 2, 1, 3, 2))
        self.assertEqual(tuple(result["draft_cache"].shape), (1, 2, 2, 1, 4, 2))
        self.assertEqual(result["tokens"].tolist(), [[9, 1, 2, 3, 4, 5, 6, 7], [10, 1, 2, 3, 4, 5, 6, 7]])
        self.assertEqual(result["draft_positions"].tolist(), [list(range(2, 9))] * 2)
        self.assertEqual(result["keep"][0, :11].tolist(), [1, 0, 1] + [1] * 8)
        self.assertEqual(int(result["keep"][:, 11:].count_nonzero()), 0)
        before = result["target_cache"].clone()
        rows[0].kv[0][0].fill_(-1)
        rows[0].cache.keys[0].fill_(-2)
        torch.testing.assert_close(result["target_cache"], before, atol=0, rtol=0)
        self.assertEqual(float(result["draft_cache"][0, 0, 0, 0, 0, 0]), 0.0)

    def test_replay_rejects_cache_extent_or_mask_mismatch(self):
        engine, rows = fixture()
        rows[0].kv = ((torch.zeros(1, 1, 2, 2), torch.zeros(1, 1, 2, 2)),)
        rows[1].kv = rows[0].kv
        with self.assertRaisesRegex(ValueError, "cache extents"):
            AUDIT.replay_inputs(engine, rows)
        engine, rows = fixture()
        for row in rows:
            row.mask = torch.ones(1, 2)
        with self.assertRaisesRegex(ValueError, "mask shape"):
            AUDIT.replay_inputs(engine, rows)

    def test_draft_reference_masks_padded_cache_and_preserves_real_values(self):
        engine, rows = fixture()
        inputs = AUDIT.replay_inputs(engine, rows)
        captured = {}

        def forward(anchors, positions, keys, values, keep, context_positions):
            captured.update(keys=keys, values=values, keep=keep, context_positions=context_positions)
            return torch.zeros(2, 7, 2), torch.zeros(2, 7, 4)

        output = AUDIT.draft_reference(SimpleNamespace(forward=forward, context_uses_positions=False), inputs)
        self.assertEqual(set(output), {"hidden", "base"})
        self.assertEqual(tuple(captured["keys"][0].shape), (2, 1, 128, 2))
        self.assertTrue(bool(captured["keep"][:, :4].all()))
        self.assertFalse(bool(captured["keep"][:, 4:].any()))
        self.assertIsNone(captured["context_positions"])
        torch.testing.assert_close(captured["keys"][0][:, :, :4], inputs["draft_cache"][0, 0])
        self.assertEqual(int(captured["keys"][0][:, :, 4:].count_nonzero()), 0)

    def test_target_reference_compares_only_eight_new_kv_positions(self):
        engine, rows = fixture()
        inputs = AUDIT.replay_inputs(engine, rows)
        captured = {}

        def forward(embeddings, past, mask, position):
            captured.update(past=past, mask=mask, position=position)
            new_key = torch.full((2, 1, 8, 2), 31.0)
            new_value = torch.full_like(new_key, 37.0)
            present = ((torch.cat((past[0][0], new_key), dim=2),
                        torch.cat((past[0][1], new_value), dim=2)),)
            return torch.zeros(2, 8, 4), present, torch.ones(2, 8, 2), torch.ones(2, 8, 2)

        target = SimpleNamespace(_block_forward_with_hidden_states=forward)
        output = AUDIT.target_reference(target, inputs, inputs["target_cache"])
        self.assertEqual(set(output), {"logits", "selected", "final", "new_kv"})
        self.assertEqual(tuple(output["new_kv"].shape), (1, 2, 2, 1, 8, 2))
        self.assertTrue(bool((output["new_kv"][0, 0] == 31).all()))
        self.assertTrue(bool((output["new_kv"][0, 1] == 37).all()))
        torch.testing.assert_close(captured["past"][0][0], inputs["target_cache"][0, 0])
        self.assertIsNone(captured["position"])

    def test_output_comparison_cannot_pass_missing_or_nonfinite_evidence(self):
        with self.assertRaises(ValueError):
            AUDIT.compare_outputs({}, {}, "bf16")
        with self.assertRaises(ValueError):
            AUDIT.compare_outputs({"hidden": torch.ones(1)}, {"base": torch.ones(1)}, "bf16")
        for candidate in (torch.tensor([float("nan")]), torch.ones(2), torch.empty(0)):
            report = AUDIT.compare_outputs({"hidden": torch.ones(1)}, {"hidden": candidate}, "bf16")
            self.assertFalse(report["pass_gate"])
            json.dumps(report, allow_nan=False)

    def test_engine_hash_and_checkpoint_attestation_are_distinct(self):
        with tempfile.TemporaryDirectory(dir=ROOT, prefix="ar-audit-test-") as directory:
            path = Path(directory) / "target_full_b1.engine"
            path.write_bytes(b"test engine bytes")
            checkpoint = Path(directory) / "weights.bin"
            checkpoint.write_bytes(b"actual loaded checkpoint")
            config = Path(directory) / "config.json"
            config.write_text('{}')
            sources = {"target_checkpoint": checkpoint, "target_config": config}
            legacy = AUDIT.engine_evidence(path, 1, "target", sources)
            self.assertFalse(legacy["weight_identity_verified"])
            metadata = dict(batch=1, sha256=AUDIT.digest(path), model_sha256=AUDIT.digest(checkpoint))
            path.with_suffix(".json").write_text(json.dumps(metadata))
            self.assertFalse(AUDIT.engine_evidence(path, 1, "target", sources)["weight_identity_verified"])
            metadata.pop("model_sha256")
            metadata["provenance"] = dict(
                schema=1, status="recorded_not_audited", component="target",
                model_sources=[AUDIT.file_record(source, role) for role, source in sources.items()],
                constant_data_sha256="a" * 64,
            )
            path.with_suffix(".json").write_text(json.dumps(metadata))
            recorded = AUDIT.engine_evidence(path, 1, "target", sources)
            self.assertTrue(recorded["weight_identity_verified"])
            self.assertEqual(len(recorded["model_source_checks"]), 2)
            self.assertFalse(recorded["constant_data_independently_verified"])
            checkpoint.write_bytes(b"changed loaded checkpoint")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch for target_checkpoint"):
                AUDIT.engine_evidence(path, 1, "target", sources)
            checkpoint.write_bytes(b"actual loaded checkpoint")
            config.write_text('{"changed": true}')
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch for target_config"):
                AUDIT.engine_evidence(path, 1, "target", sources)
            config.write_text('{}')
            with self.assertRaisesRegex(ValueError, "batch mismatch"):
                AUDIT.engine_evidence(path, 4, "target", sources)
            path.write_bytes(b"changed engine bytes")
            with self.assertRaisesRegex(ValueError, "Engine SHA256"):
                AUDIT.engine_evidence(path, 1, "target", sources)

    def test_recorded_provenance_requires_all_unique_roles_and_engine_binding(self):
        with tempfile.TemporaryDirectory(dir=ROOT, prefix="ar-audit-test-") as directory:
            path = Path(directory) / "draft_full_b1.engine"
            path.write_bytes(b"test engine bytes")
            checkpoint = Path(directory) / "weights.bin"
            checkpoint.write_bytes(b"actual loaded checkpoint")
            config = Path(directory) / "config.json"
            config.write_text('{}')
            sources = {"draft_checkpoint": checkpoint, "draft_config": config}
            records = [AUDIT.file_record(source, role) for role, source in sources.items()]
            for declared in ([], records[:1], [records[0], records[0]], records + [dict(role="extra")]):
                metadata = dict(batch=1, sha256=AUDIT.digest(path), provenance=dict(
                    schema=1, status="recorded_not_audited", component="draft",
                    model_sources=declared, constant_data_sha256="a" * 64))
                path.with_suffix(".json").write_text(json.dumps(metadata))
                with self.subTest(declared=declared), self.assertRaisesRegex(ValueError, "source roles"):
                    AUDIT.engine_evidence(path, 1, "draft", sources)
            metadata["provenance"]["model_sources"] = records
            metadata.pop("sha256")
            path.with_suffix(".json").write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "engine SHA256 binding"):
                AUDIT.engine_evidence(path, 1, "draft", sources)
            metadata["sha256"] = AUDIT.digest(path)
            metadata["provenance"]["constant_data_sha256"] = "invalid"
            path.with_suffix(".json").write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "constant_data_sha256"):
                AUDIT.engine_evidence(path, 1, "draft", sources)

    def test_build_metadata_does_not_choose_which_checkpoint_to_hash(self):
        with tempfile.TemporaryDirectory(dir=ROOT, prefix="ar-audit-test-") as directory:
            path = Path(directory) / "draft_full_b1.engine"
            path.write_bytes(b"engine")
            checkpoint = Path(directory) / "weights.bin"
            checkpoint.write_bytes(b"checkpoint")
            config = Path(directory) / "config.json"
            config.write_text('{}')
            sources = {"draft_checkpoint": checkpoint, "draft_config": config}
            records = [dict(AUDIT.file_record(source, role), path="/unreadable/build-machine/path")
                       for role, source in sources.items()]
            metadata = dict(batch=1, sha256=AUDIT.digest(path), provenance=dict(
                schema=1, status="recorded_not_audited", component="draft",
                model_sources=records, constant_data_sha256="b" * 64))
            path.with_suffix(".json").write_text(json.dumps(metadata))
            result = AUDIT.engine_evidence(path, 1, "draft", sources)
            self.assertTrue(result["weight_identity_verified"])
            self.assertTrue(all(not row["path_matches"] for row in result["model_source_checks"]))

    def test_failures_retain_partial_report_without_running_gpu(self):
        device = ModuleType("inspark_infer.runtime.device")
        device.GPULease = lambda gpu: nullcontext()
        device.select_gpu = lambda gpu: None

        def fail(args, report):
            report["stage"] = "injected_cpu_failure"
            report["target"] = {"pass_gate": False, "comparisons": {}}
            raise ValueError("intentional validation failure")

        with tempfile.TemporaryDirectory(dir=ROOT, prefix="ar-audit-test-") as directory:
            output = Path(directory) / "nested/report.json"
            with patch.dict(sys.modules, {"inspark_infer.runtime.device": device}), patch.object(AUDIT, "validate", fail):
                with self.assertRaisesRegex(RuntimeError, "report retained"):
                    AUDIT.main(["--batch", "1", "--output", str(output)])
            report = json.loads(output.read_text())
            self.assertFalse(report["pass_gate"])
            self.assertEqual(report["stage"], "injected_cpu_failure")
            self.assertEqual(report["error"]["type"], "ValueError")
            self.assertIn("target", report)

    def test_cli_defaults_match_batch_and_bound_iterations(self):
        args = AUDIT.parse_args(["--batch", "4", "--output", "audit.json"])
        self.assertEqual(args.iterations, 20)
        self.assertEqual(args.target_engine.name, "target_full_b4.engine")
        self.assertEqual(args.draft_engine.name, "draft_full_b4.engine")
        with patch("sys.stderr"), self.assertRaises(SystemExit):
            AUDIT.parse_args(["--batch", "4", "--output", "audit.json", "--iterations", "0"])


if __name__ == "__main__":
    unittest.main()
