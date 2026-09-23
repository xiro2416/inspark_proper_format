import unittest
from pathlib import Path

from inspark_infer.models.indextts2.dspark.logic import accepted_prefix
from inspark_infer.runtime.deployment import load
from inspark_infer.runtime.graph_policy import BATCHES

ROOT=Path(__file__).resolve().parents[1]

class ReleaseContractTest(unittest.TestCase):
    def test_device_deployment_and_rollback(self):
        current=load(ROOT/'configs/hardware/sm120/sm120.json')
        rollback=load(ROOT/'configs/hardware/sm120/sm120_pre_device_commit.json')
        self.assertTrue(current['device_round_b8'])
        self.assertTrue(current['context_scatter'])
        self.assertFalse(rollback.get('device_round_b8',False))

    def test_graph_inventory(self):
        self.assertEqual(BATCHES,(1,2,3,4,5,6,7,8,16,32))

    def test_prefix_semantics(self):
        self.assertEqual(accepted_prefix([[1,3],[1,4],[0,5]],7,9),(2,False))
        self.assertEqual(accepted_prefix([[1,3],[1,9],[1,5]],7,9),(2,True))

if __name__=='__main__':unittest.main()
