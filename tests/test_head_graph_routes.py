"""CPU-only routing tests; fake capture never initializes CUDA."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from inspark_infer.runtime.graphs import HeadGraphs
from inspark_infer.runtime.pool import _engine_stats


class AcousticBackend:
    def __init__(self,component,batch=4,frames=55):
        self.component=component;self.batch=batch;self.frames=frames
        self.calls=0;self.fallbacks=0
        self.identity={'backend':'tensorrt113'}
        self.plan='plan.json';self.digest='original-engine-hash'

    def describe_route(self,*args):
        tensor=args[0]
        native=(tensor.shape[0]==self.batch and tensor.shape[-1]==self.frames and tensor.dtype==torch.float32)
        return dict(kind='tensorrt' if native else 'eager',backend='tensorrt113' if native else 'eager',
                    component=self.component,batch=int(tensor.shape[0]),frames=int(tensor.shape[-1]),
                    plan=self.plan,sha256=self.digest,plugins=['test-plugin'],
                    reason=None if native else 'shape_or_dtype',candidate_backend='tensorrt113')

    def __call__(self,*args):
        if self.describe_route(*args)['backend']=='tensorrt113':self.calls+=1
        else:self.fallbacks+=1
        return args[0].clone()

    def stats(self):
        return dict(backend='tensorrt113',calls=self.calls,fallbacks=self.fallbacks)


class FakeCaptured:
    def __init__(self,args,output):
        self.inputs=tuple(value.clone() for value in args)
        self.output=output
        self.replays=0

    def __call__(self,*args):
        self.replays+=1
        return self.output


def fake_capture(fn,args):
    for _ in range(3):fn(*args)
    return FakeCaptured(args,fn(*args))


def fake_engine():
    values={'voice.cache_mel':torch.zeros(1,80,3),
            'voice.cache_s2mel_prompt':torch.zeros(1,3,512),
            'voice.cache_s2mel_style':torch.zeros(1,192)}
    bank=SimpleNamespace(entries={'reference':object()},get=lambda key:{'values':values})
    return SimpleNamespace(sessions={},model=SimpleNamespace(bank=bank),config={'max_batch':4},
        student=AcousticBackend('cfm'),vocoder=AcousticBackend('vocoder',frames=52),
        rt=SimpleNamespace(proposal=SimpleNamespace(calls=0),target=SimpleNamespace(calls=0),
                           backbone=SimpleNamespace()),
        device_round_attempts=0,device_round_successes=0,device_round_fallbacks=0,
        head_graphs=None)


class HeadGraphRoutesTest(unittest.TestCase):
    def prepare(self):
        engine=fake_engine();bank=HeadGraphs()
        with patch('inspark_infer.runtime.graphs.capture',side_effect=fake_capture),\
             patch('inspark_infer.runtime.graphs.batches',return_value=(1,4)):
            bank.prepare(engine)
        engine.head_graphs=bank
        return engine,bank

    def test_freezes_real_route_and_distinguishes_eager_graph_replay(self):
        engine,bank=self.prepare()
        prepare_calls=engine.student.calls
        prepare_fallbacks=engine.student.fallbacks
        frozen=deepcopy(bank.routes['cfm'][4,3])
        engine.student.plan='changed-plan.json';engine.student.digest='changed-hash'
        bank.run_cfm(bank.cfm[1,3].inputs,3)
        bank.run_cfm(bank.cfm[4,3].inputs,3)
        self.assertEqual(engine.student.calls,prepare_calls)
        self.assertEqual(engine.student.fallbacks,prepare_fallbacks)
        self.assertEqual(bank.routes['cfm'][4,3],frozen)
        rows=[r for r in bank.stats()['route_counts'] if r['route']['component']=='cfm']
        eager=next(r for r in rows if r['route']['backend']=='eager')
        native=next(r for r in rows if r['route']['backend']=='tensorrt113')
        self.assertEqual((eager['prepare'],eager['replay'],eager['fallback']),(1,1,1))
        self.assertEqual((native['prepare'],native['replay'],native['fallback']),(1,1,0))
        self.assertEqual(native['route']['sha256'],'original-engine-hash')
        self.assertEqual(native['route']['plugins'],['test-plugin'])

    def test_missing_graph_uses_direct_route_without_counting_replay(self):
        engine,bank=self.prepare()
        mel=torch.zeros(2,80,52)
        output=bank.run_vocoder(mel)
        self.assertEqual(output.shape,mel.shape)
        self.assertEqual(bank.hits['vocoder'],0)
        row=next(r for r in bank.stats()['route_counts'] if r['direct'])
        self.assertEqual((row['direct'],row['replay'],row['fallback']),(1,0,1))
        self.assertEqual(row['route']['backend'],'eager')
        self.assertEqual(row['route']['graph_fallback'],'missing_graph')
        self.assertEqual(row['route']['batch'],2)

    def test_changed_tensor_signature_does_not_replay_static_graph(self):
        engine,bank=self.prepare()
        mel=torch.zeros(4,80,60)
        output=bank.run_vocoder(mel)
        self.assertEqual(output.shape,mel.shape)
        self.assertEqual(bank.vocoder[4].replays,0)
        row=next(r for r in bank.stats()['route_counts'] if r['direct'])
        self.assertEqual(row['route']['graph_fallback'],'input_signature')
        self.assertEqual(row['route']['frames'],60)

    def test_stats_snapshot_cannot_mutate_frozen_routes(self):
        _,bank=self.prepare()
        stats=bank.stats()
        stats['captured_routes']['cfm'][0]['route']['plugins'].append('mutation')
        stats['route_counts'][0]['route']['plan']='mutation'
        self.assertNotIn('mutation',bank.stats()['captured_routes']['cfm'][0]['route']['plugins'])
        self.assertNotEqual(bank.stats()['route_counts'][0]['route']['plan'],'mutation')

    def test_pool_stats_separates_prepare_direct_and_replay_for_both_components(self):
        engine,bank=self.prepare()
        engine.rt.device_target_steps=7
        engine.rt.backbone.device_steps=7
        engine.rt.backbone.calls=2
        bank.run_cfm(bank.cfm[1,3].inputs,3)
        bank.run_cfm(bank.cfm[4,3].inputs,3)
        bank.run_vocoder(torch.zeros(4,80,52))
        bank.run_vocoder(torch.zeros(2,80,52))
        engine.student(*bank.cfm[4,3].inputs)
        stats=_engine_stats(engine)
        self.assertEqual(stats['native_cfm_calls'],5)
        self.assertEqual(stats['native_vocoder_calls'],4)
        self.assertEqual(stats['acoustic_routes']['cfm']['wrapper_prepare'],dict(calls=4,fallbacks=4))
        self.assertEqual(stats['acoustic_routes']['cfm']['wrapper_direct'],dict(calls=1,fallbacks=0))
        self.assertEqual(stats['acoustic_routes']['vocoder']['wrapper_direct'],dict(calls=0,fallbacks=1))
        self.assertEqual(stats['head_graphs']['hits'],dict(cfm=2,vocoder=1))
        self.assertEqual(stats['native_vocoder_backend'],'tensorrt113')
        self.assertEqual(stats['device_target_steps'],7)
        self.assertEqual(stats['device_draft_steps'],7)
        self.assertEqual(stats['draft_backbone_calls'],2)

    def test_eager_backend_is_not_marked_as_a_backend_fallback(self):
        fn=lambda tensor:tensor
        route=HeadGraphs._describe_route('vocoder',fn,(torch.zeros(1,80,52),))
        self.assertEqual(route['backend'],'eager')
        self.assertFalse(route['fallback'])


if __name__=='__main__':unittest.main()
