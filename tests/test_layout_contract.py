"""CPU-only checks for the public imports and relocated external resources."""
import importlib.resources
from pathlib import Path
import unittest

from inspark_infer import config
from inspark_infer.runtime import config as runtime_config
from inspark_infer.runtime.deployment import load as load_deployment


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = ROOT


class LayoutContractTest(unittest.TestCase):
    def test_public_config_is_a_thin_export(self):
        self.assertIs(config.load, runtime_config.load)
        self.assertIs(config.atomic_json, runtime_config.atomic_json)

    def test_runtime_assets_remain_outside_source_distribution(self):
        values = config.load(ROOT / "configs/common/runtime.yaml")
        self.assertEqual(Path(values["weights"]), REPOSITORY / "models")
        self.assertEqual(Path(values["student"]), REPOSITORY / "models/cfm_oracle500.pt")
        self.assertEqual(Path(values["cache"]), REPOSITORY / ".cache")

    def test_acoustic_plans_resolve_to_repository_artifacts(self):
        for batch in (1, 4):
            plan = load_deployment(ROOT / f"configs/hardware/sm89/sm89_bf16_trt113_full_b{batch}.json")
            for key in ("tensorrt113_target_full_plan", "tensorrt113_draft_full_plan",
                        "tensorrt113_cfm_plan", "tensorrt113_vocoder_plan"):
                self.assertTrue(Path(plan[key]).is_relative_to(REPOSITORY / "artifacts"))

    def test_cuda_extension_sources_and_notices_are_package_resources(self):
        cases = {
            "inspark_infer.ops.cuda.target_seven": ("exact_norm.cu", "LICENSE.pytorch"),
            "inspark_infer.ops.cuda.target_norm_quant": ("fused.cu", "NOTICE.md"),
        }
        for package, names in cases.items():
            resources = importlib.resources.files(package)
            for name in names:
                self.assertTrue(resources.joinpath(name).is_file(), f"{package}:{name}")

    def test_sm89_safe_trt_profiles_are_request_isolated_and_batch_specific(self):
        keys = ("tensorrt113_target_full_plan", "tensorrt113_draft_full_plan",
                "tensorrt113_cfm_plan", "tensorrt113_vocoder_plan")
        for batch in (1, 4, 8):
            with self.subTest(batch=batch):
                plan = load_deployment(ROOT / f"configs/hardware/sm89/sm89_trt113_safe_b{batch}.json")
                self.assertIs(plan["strict_request_isolation"], True)
                for option in ("device_round_b8", "device_accept_plan", "device_residual",
                               "batched_proposal_rng"):
                    self.assertIs(plan[option], False, option)
                self.assertEqual(plan["precision"], "bf16")
                self.assertEqual(set(plan["components"]), {"target", "draft", "cfm", "vocoder"})
                self.assertIn("numerically_experimental", plan["status"])
                self.assertNotIn("cfm_triton_fusions", plan)
                for option in ("target_graphs", "draft_graphs", "proposal_graphs",
                               "prefix_graphs", "head_graphs"):
                    self.assertIs(plan[option], True, option)
                for key in keys:
                    path = Path(plan[key])
                    self.assertTrue(path.is_relative_to(REPOSITORY / "artifacts"), key)
                    self.assertRegex(path.name, rf"^plan_b{batch}(?:_|\.)", key)


if __name__ == "__main__":
    unittest.main()
