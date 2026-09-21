import json
import tempfile
import unittest
from pathlib import Path

from acc_infer_clear.planner_v2.formulas import estimate, generate_candidates
from acc_infer_clear.planner_v2.inventory import canonical_inventory
from acc_infer_clear.planner_v2.manifest import DeploymentManifest, RolePolicy, load
from acc_infer_clear.planner_v2.model import HardwareProfile, OperatorSignature, ScheduleSpec
from acc_infer_clear.planner_v2.runtime import ScheduleRegistry
from acc_infer_clear.runtime.deployment import validate


class PlannerV2ContractTest(unittest.TestCase):
    def signature(self):
        return OperatorSignature('target','qkv','gemm',64,3840,1280,'fp8','fp8',batch=8)

    def schedule(self):
        return ScheduleSpec('explicit','full_m',64,32,128,8,2,warp_m=2,warp_n=4)

    def test_precision_fallback(self):
        self.assertEqual(HardwareProfile.synthetic(80).preferred_matrix_dtype,'bf16')
        self.assertEqual(HardwareProfile.synthetic(86).preferred_matrix_dtype,'bf16')
        self.assertEqual(HardwareProfile.synthetic(89).preferred_matrix_dtype,'fp8')

    def test_resource_and_candidates(self):
        profile=HardwareProfile.synthetic(120,sms=156)
        self.assertTrue(estimate(profile,self.signature(),self.schedule())['legal'])
        rows=generate_candidates(profile,self.signature(),limit=8)
        self.assertTrue(rows);self.assertLessEqual(len(rows),8)
        self.assertEqual({row[0].schedule for row in rows},{'full_m'})

    def test_inventory_covers_pipeline(self):
        rows=canonical_inventory(HardwareProfile.synthetic(89),batches=(1,8))
        kinds={row.kind for row in rows}
        self.assertTrue({'gemm','conv','attention','pointwise','layout','control'}<=kinds)

    def test_manifest_roundtrip_and_apply_guard(self):
        profile=HardwareProfile.synthetic(120);signature=self.signature();schedule=self.schedule()
        policy=RolePolicy('explicit','mk_nk',{signature.shape_key:schedule})
        manifest=DeploymentManifest(profile,'model','source',{}, {'target:qkv':policy},{'q':signature})
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'manifest.json';manifest.write(path);restored=load(path)
        self.assertEqual(restored.manifest_hash,manifest.manifest_hash)
        with self.assertRaises(ValueError):ScheduleRegistry(restored,apply=True)

    def test_deployment_requires_manifest_for_apply(self):
        root=Path(__file__).resolve().parents[1];plan=json.loads((root/'configs/sm120.json').read_text())
        plan['planner_v2_apply']=True
        with self.assertRaises(ValueError):validate(plan)


if __name__=='__main__':unittest.main()
