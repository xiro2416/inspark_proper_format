"""No CUDA: validate full-EOS/cancellation soak bookkeeping with a fake owner."""
import importlib.util
from contextlib import redirect_stdout
import hashlib
import io
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

import numpy as np

path=Path(__file__).resolve().parents[1]/'benchmarks/soak_requests.py'
spec=importlib.util.spec_from_file_location('request_soak',path)
soak=importlib.util.module_from_spec(spec);spec.loader.exec_module(soak)


def stats():
    return dict(sessions=0,active_rows=0,error_sessions=0,rng_policy='legacy_per_request',
                scheduler_max_batch=4,target_batch_counts={'1':1},
                strict_request_isolation=True,target_slots=dict(capacity=2,free=2,free_unique=2,leased=0,valid=True),
                draft_slots=None,memory=dict(cuda_allocated_bytes=0,cuda_reserved_bytes=0,
                cuda_peak_allocated_bytes=0,cuda_peak_reserved_bytes=0,rss_bytes=1024),**{key:0 for key in soak.COUNTERS})


def chunk(index,start,end,eos):
    return dict(index=index,sample_start=start,sample_end=end,pcm=np.zeros(end-start,dtype=np.int16),
                eos=eos,cfm_batch=1,vocoder_batch=1)


class FakePool:
    def __init__(self,config=None,gpu=None,workers=None,clock=None):
        self.owners={};self.sessions={};self.cancelled=[];self.released=[];self.clock=clock
        self.max_batch=(config or {}).get('max_batch',4);self.target_batches={}
    def __enter__(self):
        if self.clock:self.clock.advance(20)
        return self
    def __exit__(self,*args):pass
    def prepare_reference(self,*args):
        if self.clock:self.clock.advance(50)
    def prepare_deployment(self,plan):
        if self.clock:self.clock.advance(50)
        return [dict(requested=plan)]
    def create_session(self,identifier,voice,seed,emotion=None,arrival=None):
        if identifier in self.owners:raise ValueError('Duplicate')
        self.owners[identifier]=0
        self.sessions[identifier]=dict(complete=False,error=None,chunks=[],codes=[1,2],text='',seed=seed,ticks=0,kv_head_lengths=200)
    def push_text(self,identifier,text):self.sessions[identifier]['text']=text
    def finish_input(self,identifier):
        if not self.sessions[identifier]['text']:raise RuntimeError('No spoken text in request')
    def tick(self):
        if self.clock:self.clock.advance(1)
        active=[state for state in self.sessions.values() if state['text']][:self.max_batch]
        for state in active:state['ticks']+=1
        if active:
            key=str(len(active));self.target_batches[key]=self.target_batches.get(key,0)+1
        return []
    def request_rng_snapshot(self,identifier):
        state=self.sessions[identifier]
        digest=lambda text:hashlib.sha256(text.encode()).hexdigest()
        return dict(state_sha256=digest(f"state{state['seed']}-{state['ticks']}"),
            next_uniform_prefix_sha256=digest(f"prefix{state['seed']}-{state['ticks']}"),
            cfm_noise_prefix_sha256=digest(f"noise{state['seed']}"),seed=state['seed'],
            rounds=state['ticks'],code_count=2+state['ticks'],phase='active_row' if state['ticks'] else 'session_boundary')
    def run_ready(self):
        if self.clock:self.clock.advance(2)
        events=[]
        for identifier,state in self.sessions.items():
            state['chunks']=[chunk(0,0,44*256,False),chunk(1,44*256,50*256,True)]
            state['complete']=True
            events.append(dict(request_id=identifier,complete=True,chunk=state['chunks'][-1],received=self.clock() if self.clock else 1.))
        return events
    def result(self,identifier):return self.sessions[identifier]
    def release(self,identifier):
        self.released.append(identifier);self.owners.pop(identifier);self.sessions.pop(identifier)
    def cancel(self,identifier):
        self.cancelled.append(identifier);self.owners.pop(identifier);self.sessions.pop(identifier)
    def stats(self):
        value=stats();value.update(sessions=len(self.sessions),scheduler_max_batch=self.max_batch,
                                  target_batch_counts=dict(self.target_batches));return [value]


def args(**changes):
    values=dict(batch=8,gpu=0,reference=Path('/workspace/reference.wav'),seconds=10,warmups=1,
                request_timeout_seconds=300,cancel_every=11,strict_isolation=True,min_kv_length=129,
                max_live_growth_mib=64,max_rss_growth_mib=256,max_live_slope_mib_per_min=1,
                max_rss_slope_mib_per_min=16,memory_min_samples=3,memory_min_span_seconds=60)
    values.update(changes);return NS(**values)


class Clock:
    def __init__(self):self.now=0
    def __call__(self):return self.now
    def advance(self,value):self.now+=value


class RequestSoak(unittest.TestCase):
    def test_only_complete_eos_counts_as_success(self):
        state=dict(complete=True,error=None,codes=[1],chunks=[chunk(0,0,44*256,False)])
        with self.assertRaisesRegex(RuntimeError,'full EOS'):soak.request_record('r',state,0,1,.5,False)
        state['chunks'][-1]['eos']=True
        self.assertEqual(soak.request_record('r',state,0,1,.5,False)['outcome'],'complete_eos')

    def test_chunk_contiguity_and_size_are_checked(self):
        state=dict(complete=True,error=None,codes=[1],chunks=[chunk(0,1,257,True)])
        with self.assertRaisesRegex(RuntimeError,'contiguous'):soak.request_record('r',state,0,1,.5,False)

    def test_cancel_wave_reuses_ids_and_drains_all(self):
        cases=[dict(id='case',text='真实长句子。',seed=7,emotion=[0.]*8)]
        pool=FakePool()
        for wave in range(2):
            records,snapshot=soak.run_wave(pool,cases,4,wave,10,cancel_every=2,clock=lambda:1.)
            self.assertEqual(sum(row['outcome']=='complete_eos' for row in records),2)
            self.assertEqual(sum(row['outcome']=='cancelled' for row in records),2)
            self.assertFalse(pool.owners);soak.assert_drained(snapshot)
        self.assertEqual(len(pool.released),4);self.assertEqual(len(pool.cancelled),4)

    def test_worker_error_then_same_id_reuse(self):
        pool=FakePool();self.assertEqual(len(soak.check_expected_errors(pool)),2)
        self.assertFalse(pool.owners)

    def test_missing_metrics_and_leaked_slots_fail_closed(self):
        value=stats();value.pop('native_target_steps')
        with self.assertRaisesRegex(RuntimeError,'Missing'):soak.compact_stats(value)
        value=stats();value['target_slots'].update(leased=1,free=1,free_unique=1)
        with self.assertRaisesRegex(RuntimeError,'not returned'):soak.assert_drained(value)

    def test_strict_override_disables_all_changed_rng_paths(self):
        original=dict(device_round_b8=True,batched_proposal_rng=True,device_residual=True)
        plan,changes=soak.strict_plan(original,True)
        self.assertTrue(original['device_round_b8']);self.assertTrue(plan['strict_request_isolation'])
        for key in ('device_round_b8','batched_proposal_rng','device_residual','device_accept_plan'):
            self.assertFalse(plan[key])
        self.assertIn('device_round_b8',changes)

    def test_real_boundary_protocol_detects_admission_cancel_and_reuse(self):
        case=dict(id='case',text='真实测试句子。',seed=7,emotion=[0.]*8)
        pool=FakePool();checks=soak.check_runtime_rng(pool,case)
        self.assertTrue(checks['passed']);self.assertTrue(checks['active_row_observed'])
        self.assertFalse(pool.owners)
        before={key:'a'*64 for key in soak.RNG_HASHES};after=dict(before,cfm_noise_prefix_sha256='b'*64)
        change=soak.compare_rng_snapshots(before,after)
        self.assertFalse(change['passed']);self.assertEqual(change['changed_fields'],['cfm_noise_prefix_sha256'])

    def test_memory_trend_fails_even_when_final_growth_is_below_budget(self):
        before=stats();snapshots=[]
        for second in range(60,601,60):
            value=stats();value['memory']['cuda_allocated_bytes']=int(second*.1*1024**2)
            snapshots.append(dict(elapsed_s=second,**value))
        gate=soak.memory_gate(before,snapshots,args())
        self.assertTrue(gate['absolute_passed']);self.assertTrue(gate['trend_sufficient'])
        self.assertFalse(gate['passed'])
        self.assertAlmostEqual(gate['metrics']['cuda_allocated_bytes']['late_slope_mib_per_min'],6)

    def test_post_drain_memory_peak_cannot_hide_behind_final_recovery(self):
        snapshots=[]
        for index,second in enumerate(range(60,601,60)):
            value=stats();value['memory']['cuda_allocated_bytes']=(80*1024**2 if index==2 else 0)
            snapshots.append(dict(elapsed_s=second,**value))
        gate=soak.memory_gate(stats(),snapshots,args())
        self.assertFalse(gate['absolute_passed'])
        self.assertEqual(gate['metrics']['cuda_allocated_bytes']['final_growth_mib'],0)

    def test_smoke_excludes_setup_warmup_and_does_not_qualify_as_soak(self):
        case=dict(id='case',text='真实测试句子。',seed=7,emotion=[0.]*8);clock=Clock()
        factory=lambda config,gpu,workers:FakePool(config,gpu,workers,clock=clock)
        with redirect_stdout(io.StringIO()):
            result=soak.run_tier(args(seconds=4),{}, {},[case],16,factory,clock=clock)
        self.assertEqual(result['setup_seconds'],120);self.assertEqual(result['rng_probe_seconds'],2)
        self.assertEqual(result['warmup_seconds'],3);self.assertEqual(result['elapsed_s'],6)
        self.assertEqual(result['model_max_batch'],8);self.assertEqual(result['pass_gate']['microbatch']['observed_max'],8)
        self.assertTrue(result['pass_gate']['checks']['actual_concurrency'])
        self.assertEqual(result['status'],'smoke_passed');self.assertFalse(result['pass_gate']['soak_qualified'])
        self.assertFalse(result['pass_gate']['memory']['trend_sufficient'])

    def test_long_text_label_is_not_long_kv_evidence(self):
        case=dict(id='case',text='真实测试句子。',seed=7,emotion=[0.]*8);clock=Clock()
        factory=lambda config,gpu,workers:FakePool(config,gpu,workers,clock=clock)
        with redirect_stdout(io.StringIO()):
            result=soak.run_tier(args(seconds=4),{}, {},[case],1,factory,clock=clock)
        for record in result['requests']:record['kv_length']=128;record['long_text']=True
        gate=soak.pass_gate(result,args(seconds=4))
        self.assertFalse(gate['passed']);self.assertIn('long_sequence',gate['failed_checks'])

    def test_host_native_counts_remain_windowed_and_do_not_imply_all_trt_soak(self):
        class NativeHostPool(FakePool):
            def stats(self):
                value=super().stats()[0]
                calls=sum(self.target_batches.values())
                value.update(target_calls=calls,native_target_steps=calls,device_target_steps=0)
                return [value]
        case=dict(id='case',text='真实测试句子。',seed=7,emotion=[0.]*8);clock=Clock()
        factory=lambda config,gpu,workers:NativeHostPool(config,gpu,workers,clock=clock)
        with redirect_stdout(io.StringIO()):
            result=soak.run_tier(args(seconds=4),{}, {},[case],4,factory,clock=clock)
        delta=result['counter_delta'];batch_delta=result['pass_gate']['microbatch']['target_batch_counts_delta']
        self.assertGreater(result['before']['native_target_steps'],0)
        self.assertEqual(delta['native_target_steps'],2)
        self.assertEqual(delta['native_target_steps'],sum(batch_delta.values()))
        self.assertEqual(delta['target_calls'],delta['native_target_steps'])
        self.assertEqual(delta['device_target_steps'],0)
        # Full-EOS/lifecycle gates are intentionally not an all-native coverage gate.
        result['after']['native_target_steps']=result['before']['native_target_steps']
        self.assertTrue(soak.pass_gate(result,args(seconds=4))['passed'])

    def test_oom_skip_never_hides_required_tier_or_non_oom_failure(self):
        oom=RuntimeError('torch.OutOfMemoryError: CUDA out of memory')
        self.assertTrue(soak.may_skip_oom(oom,16,[16]))
        self.assertFalse(soak.may_skip_oom(oom,16,[]))
        self.assertFalse(soak.may_skip_oom(oom,8,[8,16]))
        self.assertFalse(soak.may_skip_oom(RuntimeError('Soak pass gate failed: memory_absolute'),16,[16]))
        self.assertFalse(soak.may_skip_oom(RuntimeError('CUDA illegal memory access'),16,[16]))


if __name__=='__main__':unittest.main()
