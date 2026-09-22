"""Pure CPU tests for fail-closed first-chunk evidence and timing accounting."""
import importlib.util
from pathlib import Path
import unittest


spec=importlib.util.spec_from_file_location('validate_trt113_first_chunks',
    Path(__file__).resolve().parents[1]/'scripts/validate_trt113_first_chunks.py')
audit=importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def snapshot(steps=0,replays=0):
    result={name:0 for name in audit.COUNTERS}
    result.update(device_target_steps=steps,device_draft_steps=steps,
                  native_target_steps=steps,native_draft_steps=steps,acoustic_routes={})
    for component in ('cfm','vocoder'):
        route=dict(component=component,kind='tensorrt',backend='tensorrt113',batch=4,
                   frames=310 if component=='cfm' else 52,plan=f'{component}.json',sha256='abc')
        result['acoustic_routes'][component]=dict(wrapper_direct=dict(calls=0,fallbacks=0),
            graph_routes=[dict(route=route,prepare=1,direct=0,replay=replays,fallback=0)])
    return result


def event(ident='request',frames=44,eos=False,received=1.1):
    return dict(request_id=ident,received=received,chunk=dict(index=0,sample_start=0,
        sample_end=frames*256,pcm=[0]*(frames*256),eos=eos,cfm_batch=4,vocoder_batch=4))


def state():
    return dict(chunks=[{}],complete=False,codes=list(range(31)),error=None,
                rounds=5,accepted=[6,5,6,5,4],kv_head_lengths=72)


class FirstChunkValidationTest(unittest.TestCase):
    def test_four_native_components_pass_using_replays_not_wrapper_counts(self):
        result=audit.validate_window(snapshot(),snapshot(steps=5,replays=1))
        self.assertTrue(result['passed'])
        self.assertEqual(result['components']['target']['total'],5)
        self.assertEqual(result['components']['draft']['total'],5)
        self.assertEqual(result['components']['cfm']['native'],1)
        self.assertEqual(result['components']['cfm']['wrapper_direct']['calls'],0)

    def test_missing_metric_fails_closed(self):
        after=snapshot(5,1);del after['device_target_steps']
        with self.assertRaisesRegex(audit.AuditError,'Missing metric'):
            audit.validate_window(snapshot(),after)

    def test_absent_route_metrics_fail_even_if_wrapper_claims_native(self):
        after=snapshot(5,1);del after['acoustic_routes']['vocoder']['graph_routes']
        with self.assertRaisesRegex(audit.AuditError,'Missing route metrics'):
            audit.validate_window(snapshot(),after)

    def test_eager_replay_does_not_count_as_native(self):
        before=snapshot();after=snapshot(5,1)
        for row in (before['acoustic_routes']['cfm']['graph_routes'][0],
                    after['acoustic_routes']['cfm']['graph_routes'][0]):
            row['route'].update(kind='eager',backend='eager')
        after['acoustic_routes']['cfm']['graph_routes'][0]['fallback']=1
        result=audit.validate_window(before,after)
        self.assertFalse(result['passed'])
        self.assertEqual(result['components']['cfm']['native'],0)
        self.assertEqual(result['components']['cfm']['non_trt'],1)

    def test_zero_native_executions_and_ar_fallback_fail(self):
        result=audit.validate_window(snapshot(),snapshot())
        self.assertFalse(result['passed'])
        self.assertEqual(len(result['errors']),4)
        after=snapshot(5,1);after['target_calls']=1;after['draft_backbone_calls']=1
        after['device_round_fallbacks']=1
        result=audit.validate_window(snapshot(),after)
        self.assertFalse(result['passed'])
        self.assertEqual(result['components']['target']['non_trt'],1)
        self.assertEqual(result['components']['draft']['non_trt'],1)

    def test_prepare_events_and_counter_reset_cannot_be_runtime_success(self):
        after=snapshot(5,1);after['acoustic_routes']['vocoder']['graph_routes'][0]['prepare']=2
        with self.assertRaisesRegex(audit.AuditError,'Online graph capture'):
            audit.validate_window(snapshot(),after)
        with self.assertRaisesRegex(audit.AuditError,'Counter decreased'):
            audit.validate_window(snapshot(6,1),snapshot(5,2))

    def test_unclassified_direct_calls_are_rejected(self):
        after=snapshot(5,1);after['acoustic_routes']['cfm']['wrapper_direct']['calls']=1
        with self.assertRaisesRegex(audit.AuditError,'Unaccounted direct'):
            audit.validate_window(snapshot(),after)

    def test_classified_direct_trt_route_is_supported(self):
        after=snapshot(5,1)
        component=after['acoustic_routes']['cfm']
        component['graph_routes'][0].update(replay=0,direct=1)
        component['wrapper_direct']['calls']=1
        result=audit.validate_window(snapshot(),after)
        self.assertTrue(result['passed'])
        self.assertEqual(result['components']['cfm']['direct'],1)

    def test_native_route_needs_matching_backend_and_engine_identity(self):
        before=snapshot();after=snapshot(5,1)
        for value in (before,after):value['acoustic_routes']['cfm']['graph_routes'][0]['route']['backend']='eager'
        with self.assertRaisesRegex(audit.AuditError,'Contradictory runtime route'):
            audit.validate_window(before,after)
        before=snapshot();after=snapshot(5,1)
        for value in (before,after):del value['acoustic_routes']['cfm']['graph_routes'][0]['route']['sha256']
        with self.assertRaisesRegex(audit.AuditError,'Missing native plan/hash'):
            audit.validate_window(before,after)

    def test_44_frames_and_short_early_eos_preserve_contract(self):
        self.assertEqual(audit.request_record('request',event(),state(),1.)['frame_count'],44)
        self.assertEqual(audit.request_record('request',event(frames=12,eos=True),state(),1.)['frame_count'],12)
        with self.assertRaisesRegex(audit.AuditError,'without an early EOS'):
            audit.request_record('request',event(frames=12),state(),1.)
        with self.assertRaisesRegex(audit.AuditError,'without an early EOS'):
            audit.request_record('request',event(frames=45,eos=True),state(),1.)

    def test_wrong_sample_count_or_extra_chunk_is_rejected(self):
        emitted=event();emitted['chunk']['pcm'].pop()
        with self.assertRaisesRegex(audit.AuditError,'invalid PCM'):
            audit.request_record('request',emitted,state(),1.)
        result=state();result['chunks'].append({})
        with self.assertRaisesRegex(audit.AuditError,'beyond the first chunk'):
            audit.request_record('request',event(),result,1.)

    def test_wave_starts_timing_before_admission_and_cleans_up(self):
        class FakePool:
            def __init__(self):self.arrivals={};self.cancelled=[];self.snapshots=iter((snapshot(),snapshot(5,1)))
            def stats(self):return [next(self.snapshots)]
            def create_session(self,ident,voice,seed,arrival):self.arrivals[ident]=arrival
            def push_text(self,*args):pass
            def finish_input(self,*args):pass
            def run_ready(self):
                received=max(self.arrivals.values())+.1
                return [event(ident=ident,received=received) for ident in self.arrivals]
            def result(self,ident):return state()
            def cancel(self,ident):self.cancelled.append(ident)
        pool=FakePool();result=audit.run_wave(pool,4,'test')
        self.assertTrue(result['audit']['passed'])
        self.assertEqual(len(pool.cancelled),4)
        self.assertEqual(len(result['requests']),4)
        self.assertGreaterEqual(result['all_first_chunks_ms'],100.)
        self.assertTrue(all(row['first_chunk_ms']>=99.99 for row in result['requests']))


if __name__=='__main__':unittest.main()
