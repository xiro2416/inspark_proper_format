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

    def test_direct_graph_gate_requires_finite_exact_evidence_for_graph_routes(self):
        row = {"route": {"execution": "graph"}}
        self.assertFalse(audit.direct_graph_gate(row)["pass_gate"])
        for value in ({"exact_gate": True, "pass_gate": False},
                      {"exact_gate": False, "pass_gate": True}):
            self.assertFalse(audit.direct_graph_gate({**row, "graph_vs_direct": value})["pass_gate"])
        self.assertTrue(audit.direct_graph_gate({**row, "graph_vs_direct": {
            "exact_gate": True, "pass_gate": True}})["pass_gate"])
        self.assertTrue(audit.direct_graph_gate({"route": {"execution": "direct"}})["pass_gate"])

    def test_installed_native_engines_are_not_actual_execution_coverage(self):
        manifest = {"calls": [{"component": component, "route": {
            "backend": "eager", "candidate_backend": "tensorrt113", "batch": 1}}
            for component in ("cfm", "vocoder")]}
        self.assertFalse(audit.acoustic_coverage(manifest)["pass_gate"])
        for row in manifest["calls"]:
            row["route"]["backend"] = "tensorrt113"
        coverage = audit.acoustic_coverage(manifest)
        self.assertTrue(coverage["pass_gate"])
        self.assertEqual(coverage["components"]["cfm"]["actual_native_batches"], [1])

    def test_same_weight_gate_separates_numerical_source_and_na_evidence(self):
        native = {"has_native_engine": True, "weight_identity_verified": True}
        absent = {"has_native_engine": False, "weight_identity_verified": False,
                  "provenance_status": "not_applicable"}
        for rows in ((native, native), (native, absent), (absent, absent)):
            with self.subTest(rows=rows):
                evidence = dict(zip(("cfm", "vocoder"), rows))
                self.assertTrue(audit.same_weight_numerical_gate(True, True, evidence))
                self.assertFalse(audit.same_weight_numerical_gate(False, True, evidence))
                self.assertFalse(audit.same_weight_numerical_gate(True, False, evidence))
        # Unknown/missing historical metadata must never become explicit N/A.
        for unverified in ({}, {"has_native_engine": None},
                           {"has_native_engine": True, "weight_identity_verified": False}):
            evidence = {"cfm": native, "vocoder": unverified}
            self.assertFalse(audit.same_weight_numerical_gate(True, True, evidence))
        self.assertFalse(audit.same_weight_numerical_gate(True, True, {"cfm": native}))


if __name__ == "__main__":
    unittest.main()
