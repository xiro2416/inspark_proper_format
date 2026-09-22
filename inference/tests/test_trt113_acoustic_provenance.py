"""CPU-only source/engine hash evidence; temporary files stay in /workspace."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import audit_real_acoustics as audit
from trt113_provenance import file_record


class AcousticProvenanceTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix=".acoustic-provenance-", dir=ROOT)
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def fixture(self, component="cfm", recorded=True):
        roles = ("s2mel_checkpoint", "student_checkpoint", "s2mel_config") if component == "cfm" else (
            "vocoder_checkpoint", "vocoder_config")
        paths = {}
        for role in roles:
            path = self.directory / role
            path.write_bytes(("actual loader " + role).encode())
            paths[role] = path
        engine = self.directory / f"{component}.engine"
        engine.write_bytes(b"immutable test engine")
        plan = {"format": 1, "batch": 1, "engine": engine.name,
                "sha256": file_record(engine, "engine")["sha256"], "onnx_sha256": "a" * 64}
        if recorded:
            sources = [file_record(path, role) for role, path in paths.items()]
            for row in sources:
                # Descriptive build paths must never select reference files.
                row["path"] = "/not-read/historical/" + row["role"]
            plan["provenance"] = {"schema": 1, "component": component, "status": "recorded_not_audited",
                "model_sources": sources, "source": {"source_sha256": "b" * 64},
                "onnx_binding": {"onnx": {"sha256": "a" * 64},
                                 "external_data": [{"path": "weights.data", "sha256": "c" * 64}],
                                 "export_metadata": {"sha256": "d" * 64}}}
        plan_file = self.directory / f"{component}.json"
        plan_file.write_text(json.dumps(plan))
        return plan_file, plan, paths

    def test_both_components_require_every_actual_loader_source_hash(self):
        for component in ("cfm", "vocoder"):
            with self.subTest(component=component):
                path, plan, current = self.fixture(component)
                evidence = audit.acoustic_engine_evidence(path, component, current)
                self.assertTrue(evidence["weight_identity_verified"])
                self.assertFalse(evidence["engine_constants_independently_verified"])
                self.assertFalse(evidence["onnx_payloads_rehashed_by_audit"])
                self.assertTrue(all(not row["path_matches"] for row in evidence["model_source_checks"]))
                next(iter(current.values())).write_bytes(b"changed actual checkpoint")
                with self.assertRaisesRegex(ValueError, "model source SHA256 mismatch"):
                    audit.acoustic_engine_evidence(path, component, current)

    def test_legacy_and_top_level_claim_do_not_fabricate_provenance(self):
        path, plan, current = self.fixture(recorded=False)
        plan["provenance_status"] = "recorded_not_audited"
        path.write_text(json.dumps(plan))
        evidence = audit.acoustic_engine_evidence(path, "cfm", current)
        self.assertFalse(evidence["weight_identity_verified"])
        self.assertEqual(evidence["provenance_status"], "legacy_unverified")

    def test_invalid_roles_schema_component_source_or_onnx_binding_fail_closed(self):
        path, original, current = self.fixture()
        for mutation in ("missing", "duplicate", "extra", "schema", "component", "source", "onnx", "external"):
            with self.subTest(mutation=mutation):
                plan = deepcopy(original); provenance = plan["provenance"]
                if mutation == "missing": provenance["model_sources"].pop()
                if mutation == "duplicate": provenance["model_sources"].append(provenance["model_sources"][0])
                if mutation == "extra": provenance["model_sources"].append({"role": "unexpected"})
                if mutation == "schema": provenance["schema"] = 2
                if mutation == "component": provenance["component"] = "vocoder"
                if mutation == "source": provenance["source"]["source_sha256"] = None
                if mutation == "onnx": plan["onnx_sha256"] = "e" * 64
                if mutation == "external": provenance["onnx_binding"]["external_data"].append(
                    provenance["onnx_binding"]["external_data"][0])
                path.write_text(json.dumps(plan))
                with self.assertRaises(ValueError):
                    audit.acoustic_engine_evidence(path, "cfm", current)

    def test_engine_and_plan_must_match_prepared_or_captured_hashes(self):
        path, plan, current = self.fixture()
        with self.assertRaisesRegex(ValueError, "captured/loaded"):
            audit.acoustic_engine_evidence(path, "cfm", current, expected_engine_sha256="e" * 64)
        with self.assertRaisesRegex(ValueError, "plan SHA256"):
            audit.acoustic_engine_evidence(path, "cfm", current, expected_plan_sha256="e" * 64)
        (self.directory / plan["engine"]).write_bytes(b"different engine bytes")
        with self.assertRaisesRegex(ValueError, "build plan"):
            audit.acoustic_engine_evidence(path, "cfm", current)

    def test_old_capture_remains_unverified_without_reading_new_metadata(self):
        with patch.object(audit, "acoustic_engine_evidence", side_effect=AssertionError("must not reread")):
            evidence = audit.replay_acoustic_engine_evidence({}, {})
        self.assertFalse(audit.all_acoustic_weights_verified(evidence))
        self.assertTrue(all(row["provenance_status"] == "legacy_unverified" for row in evidence.values()))

    def test_reference_rechecks_frozen_plan_engine_and_current_model_sources(self):
        path, plan, current = self.fixture()
        evidence = audit.acoustic_engine_evidence(path, "cfm", current)
        manifest = {"acoustic_engines": {"cfm": evidence,
            "vocoder": {"has_native_engine": False, "weight_identity_verified": False}}}
        models = {"cfm": {"model_sources": [file_record(source, role) for role, source in current.items()]}}
        repeated = audit.replay_acoustic_engine_evidence(manifest, models)
        self.assertTrue(repeated["cfm"]["weight_identity_verified"])
        plan["new_field"] = "changed after capture"
        path.write_text(json.dumps(plan))
        with self.assertRaisesRegex(ValueError, "plan SHA256"):
            audit.replay_acoustic_engine_evidence(manifest, models)


if __name__ == "__main__":
    unittest.main()
