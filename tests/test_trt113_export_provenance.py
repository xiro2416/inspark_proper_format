"""CPU-only checks for inheriting exact ONNX export provenance."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("trt113_provenance", ROOT / "scripts/trt113_provenance.py")
PROVENANCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROVENANCE)


class ExportProvenanceTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="onnx-provenance-", dir=ROOT)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.onnx = self.root / "candidate.onnx"
        self.onnx.write_bytes(b"test graph; no ONNX parser needed by the metadata loader")

    def metadata(self, external=False):
        artifact = PROVENANCE.file_record(self.onnx, "onnx")
        artifact["external_data"] = []
        if external:
            payload = self.root / "weights.data"
            payload.write_bytes(b"exported tensor data")
            entry = PROVENANCE.file_record(payload, "onnx_external_data")
            entry["path"] = payload.name
            artifact["external_data"].append(entry)
        document = {
            "onnx_sha256": artifact["sha256"], "onnx_artifact": artifact,
            "provenance": {"schema": 1, "status": "recorded_not_audited", "component": "cfm",
                           "model_sources": [{"path": "/workspace/historical-model.pt", "sha256": "a" * 64}],
                           "source": {"source_sha256": "b" * 64}},
        }
        self.onnx.with_suffix(".export.json").write_text(json.dumps(document))
        return document

    def test_old_export_is_explicitly_unverified(self):
        _, result = PROVENANCE.load_onnx_export(self.onnx)
        self.assertEqual(result["status"], "legacy_unverified")
        self.assertNotIn("model_sources", result)

    def test_matching_export_preserves_original_checkpoint_identity(self):
        original = self.metadata(external=True)
        _, result = PROVENANCE.load_onnx_export(self.onnx)
        self.assertEqual(result["status"], "recorded_not_audited")
        self.assertEqual(result["model_sources"], original["provenance"]["model_sources"])
        self.assertEqual(len(result["onnx_binding"]["external_data"]), 1)
        # The recorded historical checkpoint deliberately does not exist: the
        # builder must not reread today's checkpoint or substitute its identity.
        self.assertEqual(result["onnx_binding"]["onnx"]["sha256"], original["onnx_sha256"])

    def test_changed_graph_fails_before_native_build(self):
        self.metadata()
        self.onnx.write_bytes(b"changed graph")
        with self.assertRaisesRegex(ValueError, "ONNX hash"):
            PROVENANCE.load_onnx_export(self.onnx)

    def test_changed_external_weights_fail(self):
        self.metadata(external=True)
        (self.root / "weights.data").write_bytes(b"different weights")
        with self.assertRaisesRegex(ValueError, "external tensor hash mismatch"):
            PROVENANCE.load_onnx_export(self.onnx)

    def test_unbound_claimed_provenance_is_rejected(self):
        document = self.metadata()
        document.pop("onnx_sha256")
        self.onnx.with_suffix(".export.json").write_text(json.dumps(document))
        with self.assertRaisesRegex(ValueError, "requires graph"):
            PROVENANCE.load_onnx_export(self.onnx)

    @unittest.skipUnless(importlib.util.find_spec("onnx") is not None, "ONNX is an optional exporter dependency")
    def test_capture_binds_real_external_tensor_payload(self):
        import numpy as np
        import onnx

        weight = onnx.numpy_helper.from_array(np.ones(4, dtype=np.float32), name="weight")
        graph = onnx.helper.make_graph(
            [onnx.helper.make_node("Identity", ["weight"], ["output"])], "external-test", [],
            [onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [4])], [weight],
        )
        onnx.save_model(onnx.helper.make_model(graph), str(self.onnx), save_as_external_data=True,
                        all_tensors_to_one_file=True, location="weights.data", size_threshold=0)
        record = PROVENANCE.capture_onnx_artifact(self.onnx)
        self.assertEqual([item["path"] for item in record["external_data"]], ["weights.data"])
        self.assertEqual(record["external_data"][0]["sha256"],
                         PROVENANCE.file_record(self.root / "weights.data", "external")["sha256"])


if __name__ == "__main__":
    unittest.main()
