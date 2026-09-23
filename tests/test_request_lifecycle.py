"""CPU fault injection for request slot ownership and cleanup transactions."""
import copy
from types import SimpleNamespace as NS
import unittest

import torch

from inspark_infer.models.indextts2.streaming import StreamingCore
from inspark_infer.runtime.engine import Engine
from inspark_infer.runtime.indextts2.batch_context import BatchedContextAppend
from inspark_infer.runtime.indextts2.slot_draft import DraftPool
from inspark_infer.runtime.indextts2.slot_target import SlotKV, SlotTarget
from inspark_infer.runtime.pool import _cache_stats


def target_pool(slots=2):
    pool=SlotTarget.__new__(SlotTarget)
    pool.capacity=16;pool.max_slots=slots;pool.max_batch=slots
    pool.storage=torch.empty(1,2,slots,1,16,2)
    pool.keep=torch.zeros(slots,16,dtype=torch.int32)
    pool.free=list(reversed(range(slots)));pool.generations=[0]*slots;pool.leased=[False]*slots
    pool.native_attention=None;pool.native_full_bank=None;pool.pool_import=pool.import_cache
    def body(x,past,mask,positions):
        n,t,_=x.shape;kv=((torch.ones(n,1,t,2),torch.ones(n,1,t,2)),)
        return x,kv,x,x
    pool.target=NS(_block_forward_with_hidden_states=body)
    return pool


def draft_pool(slots=2):
    pool=DraftPool.__new__(DraftPool)
    pool.capacity=16;pool.max_slots=slots;pool.storage=torch.empty(1,2,slots,1,16,2)
    pool.free=list(reversed(range(slots)));pool.generations=[0]*slots;pool.leased=[False]*slots
    pool.native_bank=None
    return pool


def cache(length=2):
    return NS(length=length,keys=[torch.ones(1,1,length,2)],values=[torch.ones(1,1,length,2)])


class SlotOwnership(unittest.TestCase):
    def test_target_alias_release_is_pool_idempotent(self):
        pool=target_pool();kv=pool.import_cache(torch.zeros(1,2,1,1,2,2),0,2)
        alias=SlotKV(pool,kv.slot,kv.length,kv.generation)
        pool.release(kv);pool.release(alias);pool.release(kv)
        self.assertEqual(pool.free,[1,0]);self.assertEqual(pool.leased,[False,False])
        with self.assertRaisesRegex(RuntimeError,'Stale'):alias.check()

    def test_old_target_alias_cannot_release_reused_slot(self):
        pool=target_pool();packed=torch.zeros(1,2,1,1,2,2)
        old=pool.import_cache(packed,0,2);alias=SlotKV(pool,old.slot,old.length,old.generation)
        pool.release(old);new=pool.import_cache(packed,0,2)
        with self.assertRaisesRegex(RuntimeError,'Stale'):pool.release(alias)
        new.check();self.assertEqual(pool.free,[1])

    def test_foreign_target_release_rejected(self):
        pool=target_pool();kv=pool.import_cache(torch.zeros(1,2,1,1,2,2),0,2)
        with self.assertRaisesRegex(ValueError,'Foreign'):target_pool().release(kv)
        kv.check()

    def test_import_mirror_failure_returns_reservation(self):
        pool=target_pool()
        def fail(*args):raise RuntimeError('mirror fail')
        pool.native_full_bank=NS(import_slot=fail)
        with self.assertRaisesRegex(RuntimeError,'mirror fail'):
            pool.import_cache(torch.zeros(1,2,1,1,2,2),0,2)
        self.assertEqual(pool.free,[1,0]);self.assertFalse(any(pool.leased))

    def test_second_prefill_import_rolls_back_entire_batch(self):
        pool=target_pool()
        def mirror(packed,row,slot,length):
            if row==1:raise RuntimeError('second import')
        pool.native_full_bank=NS(import_slot=mirror)
        jobs=[(torch.zeros(1,2,2),torch.ones(1,2,dtype=torch.long))]*2
        with self.assertRaisesRegex(RuntimeError,'second import'):pool.prefill(jobs)
        self.assertEqual(pool.free,[1,0]);self.assertTrue(_cache_stats(pool)['valid'])

    def test_draft_attach_failure_restores_cache_and_pool(self):
        pool=draft_pool();value=cache();original_keys=value.keys
        def fail(*args):raise RuntimeError('draft mirror')
        pool.native_bank=NS(import_slot=fail)
        with self.assertRaisesRegex(RuntimeError,'draft mirror'):pool.attach(value)
        self.assertFalse(hasattr(value,'pool_slot'));self.assertFalse(hasattr(value,'storage_keys'))
        self.assertIs(value.keys,original_keys);self.assertEqual(pool.free,[1,0])
        pool.native_bank=None;pool.attach(value);pool.check(value)

    def test_draft_alias_and_foreign_pool(self):
        pool=draft_pool();value=cache();pool.attach(value);alias=copy.copy(value)
        with self.assertRaisesRegex(ValueError,'Foreign'):draft_pool().release(value)
        pool.release(value);pool.release(alias)
        self.assertEqual(pool.free,[1,0]);self.assertFalse(any(pool.leased))

    def test_context_capacity_checked_before_attach(self):
        pool=draft_pool();context=BatchedContextAppend.__new__(BatchedContextAppend)
        context.model=None;context.pool=pool
        with self.assertRaisesRegex(ValueError,'capacity'):
            context([((cache(15),torch.zeros(1,2,2),torch.zeros(1,2,2)),{})])
        self.assertEqual(pool.free,[1,0])


class EngineCleanup(unittest.TestCase):
    def engine(self,target_release,draft_release):
        engine=Engine.__new__(Engine);engine.closed=False
        engine.rt=NS(target=NS(release=target_release),context=NS(release=draft_release),proposal=NS(batched_rng=False))
        engine.sessions={'r':{'_row':NS(kv='target',cache='draft'),'error':None}}
        return engine

    def test_cancel_continues_cleanup_and_retains_failed_handle_for_retry(self):
        calls=[]
        def target(value):
            calls.append('target')
            if calls.count('target')==1:raise RuntimeError('target failed')
        engine=self.engine(target,lambda value:calls.append('draft'))
        with self.assertRaisesRegex(RuntimeError,'target failed'):engine.cancel('r')
        self.assertEqual(calls,['target','draft']);self.assertIn('r',engine.sessions)
        self.assertIsNone(engine.sessions['r']['_row'].cache)
        self.assertTrue(engine.sessions['r']['error'])
        engine.cancel('r');self.assertEqual(calls,['target','draft','target'])
        self.assertEqual(engine.sessions,{})

    def test_close_attempts_all_cleanup_and_is_idempotent_after_error(self):
        calls=[]
        engine=self.engine(lambda value:calls.append('target'),lambda value:calls.append('draft'))
        def runtime_close():calls.append('runtime');raise RuntimeError('runtime close')
        engine.rt.close=runtime_close
        engine.model=NS(stream=NS(synchronize=lambda:calls.append('stream')),close=lambda:calls.append('model'))
        engine.acoustic_stream=NS(synchronize=lambda:calls.append('acoustic'))
        with self.assertRaisesRegex(RuntimeError,'runtime close'):engine.close()
        self.assertEqual(calls,['stream','acoustic','target','draft','runtime','model'])
        self.assertTrue(engine.closed);self.assertIsNone(engine.rt);self.assertIsNone(engine.model)
        self.assertEqual(engine.sessions,{})
        engine.close();self.assertEqual(len(calls),6)

    def test_strict_legacy_isolation_does_not_remap_seed(self):
        engine=self.engine(lambda value:None,lambda value:None)
        result=engine.validate_request_isolation(strict=True)
        self.assertEqual(result['rng_policy'],'legacy_per_request')
        engine.rt.proposal.batched_rng=True
        with self.assertRaisesRegex(ValueError,'batched_proposal_rng'):engine.validate_request_isolation(strict=True)
        self.assertEqual(engine.validate_request_isolation(False)['rng_policy'],'legacy_shared')
        engine.rt.proposal.batched_rng=False;engine.device_round_b8=True
        with self.assertRaisesRegex(ValueError,'device_round_b8'):engine.validate_request_isolation(strict=True)
        engine.device_round_b8=False;engine.rt.residual=NS(device_normal=True)
        with self.assertRaisesRegex(ValueError,'device_residual'):engine.validate_request_isolation(strict=True)

    def test_rng_snapshot_does_not_advance_request_generator(self):
        engine=self.engine(lambda value:None,lambda value:None);engine.torch=torch
        gen=torch.Generator().manual_seed(123);before=gen.get_state()
        engine.sessions={'r':dict(gen=gen,case=dict(seed=123),codes=[],noise=torch.zeros(1,80,16))}
        first=engine.request_rng_snapshot('r');second=engine.request_rng_snapshot('r')
        self.assertEqual(first,second);self.assertTrue(torch.equal(before,gen.get_state()))


class PreparationTransaction(unittest.TestCase):
    def test_prefill_failure_restores_text_rng_and_both_pools(self):
        target=target_pool();draft=draft_pool()
        gen=torch.Generator().manual_seed(123);before=gen.get_state()
        session=dict(text='old',parts=[' new'],case=dict(id='r',voice_id='v',seed=123,emotion=[0.]*8),
                     arrival=0,codes=[],gen=gen)
        segment=NS(prefix=torch.zeros(1,2,2),mask=torch.ones(1,2,dtype=torch.long))
        def context(jobs):
            for args,kwargs in jobs:draft.attach(args[0])
            raise RuntimeError('context projection failed')
        context.release=draft.release
        def sample(row,logits):
            torch.rand(1,generator=row.generator)
            return torch.tensor([1])
        rt=NS(target=target,context=context,sample=sample,
              frontend=NS(prepare=lambda requests:{'r':NS(prepared_segments=[segment])}),
              engine=NS(draft=NS(empty_cache=lambda *args:cache(0)),target=NS(gpt=NS(stop_mel_token=9))))
        core=NS(torch=torch,rt=rt,phase=lambda name,batch,fn:fn())
        with self.assertRaisesRegex(RuntimeError,'context projection failed'):
            StreamingCore.prepare_rows(core,[session],0)
        self.assertEqual(session['text'],'old');self.assertTrue(torch.equal(before,gen.get_state()))
        self.assertEqual(target.free,[1,0]);self.assertEqual(draft.free,[1,0])
        self.assertFalse(any(target.leased));self.assertFalse(any(draft.leased))


if __name__=='__main__':unittest.main()
