"""Synthetic EOS and owner cleanup cases; not natural speech-quality evidence."""
from contextlib import nullcontext
from types import SimpleNamespace as NS

import pytest

from inspark_infer.runtime.engine import Engine


def fixture_engine():
    engine = Engine.__new__(Engine)
    log = []
    engine.closed = False
    engine.config = {'max_batch': 32}
    engine.sessions = {}
    engine.rt = NS(close=lambda: log.append('runtime-close'))
    engine.torch = NS(cuda=NS(stream=lambda _: nullcontext()), inference_mode=nullcontext)
    engine.model = NS(stream=None, _acquire=lambda _: log.append('acquire'),
                      _release=lambda: log.append('release'), close=lambda: log.append('model-close'))
    engine.unified_first_chunk = NS(failure_count=0)
    released = set()

    def release_row(row):
        if row is None:
            return
        if id(row) in released:
            raise RuntimeError('row released twice')
        released.add(id(row))
        log.append('row-release')
    engine._release_row = release_row
    return engine, log


def session(identifier, *, row=True):
    value = dict(case={'id': identifier}, chunks=[], error=None, complete=False,
                 parts=['synthetic'], input_closed=True, codes=[1], emitted=0,
                 rounds=0)
    if row:
        value['_row'] = NS()
    return value


def test_synthetic_short_eos_completes_and_suppresses_tail_work():
    engine, log = fixture_engine()
    value = session('terminal')
    canceled = []
    value.update(chunks=[{'eos': True}], emitted=256,
                 text_futures={1: ('unused tail', NS(cancel=lambda: canceled.append(True)))})
    engine._finish_or_drain(value)
    assert value['complete'] and value['parts'] == ['synthetic']
    assert canceled == [True] and 'text_futures' not in value
    assert log == []


@pytest.mark.parametrize('eos,emitted', [(False, 256), (True, 0)])
def test_synthetic_incomplete_pcm_drains_without_inventing_text(eos, emitted):
    engine, _ = fixture_engine()
    value = session('drain')
    value.update(chunks=[{'eos': eos}], emitted=emitted)
    engine._finish_or_drain(value)
    assert not value['complete']
    assert value['parts'] == ['synthetic', '']


def test_cancel_before_dispatch_and_after_preparation_release_owned_rows():
    engine, log = fixture_engine()
    canceled = []
    for identifier, owns_row in [('pending', False), ('prepared', True)]:
        value = session(identifier, row=owns_row)
        value['text_futures'] = {0: ('text', NS(cancel=lambda: canceled.append(True)))}
        engine.sessions[identifier] = value
        removed = engine.cancel(identifier)
        assert '_row' not in removed and not engine.sessions
    assert canceled == [True, True]
    assert log == ['row-release']


def test_cleanup_failure_retains_handle_and_allows_explicit_retry():
    engine, log = fixture_engine()
    value = session('retry'); row = value['_row']
    engine.sessions['retry'] = value
    release = engine._release_row
    engine._release_row = lambda _: (_ for _ in ()).throw(RuntimeError('synthetic cleanup error'))
    with pytest.raises(RuntimeError, match='cleanup error'):
        engine.cancel('retry')
    assert value['_row'] is row and value['error'] and engine.ready() == []
    engine._release_row = release
    engine.cancel('retry')
    assert engine.sessions == {} and log == ['row-release']


def test_scheduler_error_never_retries_tentative_rows_and_cancel_cleans():
    engine, log = fixture_engine()
    engine.sessions = {key: session(key) for key in ['left', 'right']}
    calls = []
    def fail(*_):
        calls.append(True)
        raise RuntimeError('synthetic status=4')
    engine.head_ready_pipeline = NS(run=fail)
    with pytest.raises(RuntimeError, match='status=4'):
        engine.run_ready()
    assert log == ['acquire', 'release']
    assert engine.unified_first_chunk.failure_count == 1
    assert all(s['error'] and '_row' in s for s in engine.sessions.values())
    assert engine.run_ready() == [] and calls == [True]
    for key in list(engine.sessions):
        engine.cancel(key)
    assert engine.sessions == {} and log.count('row-release') == 2


def test_owner_callback_error_after_partial_delivery_cleans_only_remaining_rows():
    engine, log = fixture_engine()
    engine.sessions = {key: session(key) for key in ['delivered', 'pending']}
    delivered = []
    def pipeline(rows, owner, callback):
        value = owner[id(rows[0])]
        value['chunks'].append({'index': 0, 'pcm': b'\0\0', 'eos': True})
        engine._release_row(rows[0]); value.pop('_row')
        callback({'request_id': value['case']['id'], 'chunk': value['chunks'][-1]})
        raise AssertionError('callback error must interrupt dispatch')
    engine.head_ready_pipeline = NS(run=pipeline)
    def callback(event):
        delivered.append(event['request_id'])
        raise RuntimeError('synthetic transport error')
    with pytest.raises(RuntimeError, match='transport error'):
        engine.run_ready(on_chunk=callback)
    assert delivered == ['delivered'] and engine.run_ready() == []
    for key in list(engine.sessions):
        engine.cancel(key)
    assert engine.sessions == {} and log.count('row-release') == 2
    assert log.count('release') == 1


def test_final_close_cleans_remaining_rows_and_private_pipeline_once():
    engine, log = fixture_engine()
    engine.sessions = {'active': session('active')}
    engine.head_ready_pipeline = NS(close=lambda: log.append('pipeline-close'))
    engine.close(); engine.close()
    assert engine.sessions == {} and engine.closed
    assert log == ['pipeline-close', 'row-release', 'runtime-close', 'model-close']
