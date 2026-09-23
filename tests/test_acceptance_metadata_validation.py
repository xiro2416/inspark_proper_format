"""CPU reference/negative-control checks; these do not execute fused CUDA code."""
import importlib.util
from pathlib import Path
import unittest

import torch

from inspark_infer.models.indextts2.pcg.asg import AcousticGroups
from inspark_infer.runtime.indextts2.batch_pcg import BatchedAcceptance

path=Path(__file__).resolve().parents[1]/'scripts/validate_acceptance_metadata.py'
spec=importlib.util.spec_from_file_location('acceptance_metadata_validation',path)
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


def groups():
    forward=[[0,1],[1,2],[2,3],[3],[4],[5],[6],[7]]
    reverse=[[index for index,row in enumerate(forward) if token in row] for token in range(8)]
    def csr(rows):
        sizes=torch.tensor([len(row) for row in rows]);offsets=torch.cat((torch.zeros(1,dtype=torch.long),sizes.cumsum(0)))
        return offsets,torch.tensor([value for row in rows for value in row],dtype=torch.long)
    go,gm=csr(forward);to,tg=csr(reverse)
    sparse=AcousticGroups(go,gm,to,tg,to[1:]-to[:-1],{})
    sparse.validate();return sparse.dense('cpu')


def candidate(group,temperature,*args):
    return BatchedAcceptance(group,temperature).tensor_body(*args)


class AcceptanceMetadataAudit(unittest.TestCase):
    def setUp(self):self.groups=groups()

    def fixture(self,scenario='all_accept',batch=4):
        return audit.make_fixture(self.groups,batch,scenario,123,eos=7,max_tokens=100)

    def test_all_declared_scenarios_realize_reference_flags_for_all_batches(self):
        for batch in (1,4,8):
            for scenario in audit.SCENARIOS:
                with self.subTest(batch=batch,scenario=scenario):
                    result=audit.audit_case(self.groups,self.fixture(scenario,batch),7,100)
                    self.assertTrue(result['reference_fixture_passed'])
                    self.assertFalse(result['fused_tested'])
                    self.assertNotIn('pass_gate',result)
        self.assertFalse(torch.cuda.is_initialized())

    def test_prefix_semantics_include_rejection_eos_and_budget(self):
        expected={'all_accept':(7,False,True,False),'reject_first':(0,False,True,True),
                  'reject_middle':(3,False,True,True),'reject_last':(6,False,True,True),
                  'eos_first':(1,True,False,False),'eos_middle':(4,True,False,False),
                  'eos_last':(7,True,False,False),'rejected_eos':(2,False,True,True),
                  'eos_after_rejection':(1,False,True,True),'remaining_zero':(0,False,False,False),
                  'remaining_one':(1,False,False,False),'remaining_three':(3,False,False,False),
                  'reject_at_cap':(2,False,True,True),'eos_outside_budget':(3,False,False,False),
                  'eos_at_budget_end':(3,True,False,False),'all_accept_exact_cap':(7,False,False,False)}
        for scenario,plan in expected.items():
            result=audit.audit_case(self.groups,self.fixture(scenario,1),7,100)
            self.assertEqual(tuple(result['reference_prefix'][name][0] for name in ('n','eos','correction','residual')),plan)

    def test_identity_controls_pass_all_float_and_exact_gates(self):
        result=audit.audit_case(self.groups,self.fixture('mixed_rows',8),7,100,candidate,audit.reference_prefix)
        self.assertTrue(result['pass_gate']);self.assertTrue(result['inputs_unchanged'])
        self.assertEqual(result['floating']['q']['atol'],1e-5)
        self.assertEqual(result['floating']['q']['rtol'],1e-4)

    def test_discrete_flag_mismatch_cannot_hide_in_float_tolerance(self):
        def bad(group,temp,*args):
            q,accept,packed=candidate(group,temp,*args);packed=packed.clone();packed[0,0,0]=0
            return q,accept,packed
        result=audit.audit_case(self.groups,self.fixture(),7,100,bad,audit.reference_prefix)
        self.assertFalse(result['pass_gate']);self.assertFalse(result['discrete']['flags']['pass_gate'])
        self.assertTrue(all(value['pass_gate'] for value in result['floating'].values()))
        self.assertTrue(result['prefix']['candidate_packed']['pass_gate'])
        self.assertFalse(result['prefix']['end_to_end']['pass_gate'])

    def test_prefix_kernel_failure_is_separate_from_acceptance(self):
        def bad(*args):
            n,eos,correction,residual=audit.reference_prefix(*args)
            return n+1,eos,correction,residual
        result=audit.audit_case(self.groups,self.fixture(),7,100,candidate,bad)
        self.assertFalse(result['pass_gate']);self.assertFalse(result['prefix']['reference_packed']['pass_gate'])
        self.assertTrue(all(value['pass_gate'] for value in result['discrete'].values()))

    def test_strict_float_gate_rejects_q_error(self):
        def bad(group,temp,*args):
            q,accept,packed=candidate(group,temp,*args)
            return q+1e-3,accept,packed
        result=audit.audit_case(self.groups,self.fixture(),7,100,bad,audit.reference_prefix)
        self.assertFalse(result['pass_gate']);self.assertFalse(result['floating']['q']['pass_gate'])
        self.assertTrue(result['prefix']['end_to_end']['pass_gate'])

    def test_input_mutation_fails_even_with_matching_decisions(self):
        def bad(group,temp,*args):
            result=candidate(group,temp,*args);args[1].mul_(2)
            return result
        result=audit.audit_case(self.groups,self.fixture(),7,100,bad,audit.reference_prefix)
        self.assertFalse(result['inputs_unchanged']);self.assertFalse(result['pass_gate'])


if __name__=='__main__':unittest.main()
