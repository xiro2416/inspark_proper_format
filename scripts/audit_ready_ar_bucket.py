#!/usr/bin/env python3
"""Audit an AR-only ready bucket on noncontiguous real frozen B64 history.

This never constructs a synthetic full B48 deployment or runs its inherited
B32 prefix/acoustic plans. Floating differences are reported without a gate.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

import torch
from inspark_infer.runtime.bundle_paths import read_json
from inspark_infer.runtime.ready_scheduler import validate_component_identity
from scripts.audit_unified_ar import (_cpu, target_reference, draft_reference,
    conditional_rnn_reference, plan_model_sources)
from scripts.audit_unified_acoustics import digest, metrics


CACHE_FIELDS = {'target_cache', 'draft_cache', 'target_kv_append'}


def slice_frozen(frozen, indices):
    if (frozen.get('schema') != 1 or frozen.get('kind') != 'unified_frozen_first_ar_round'
            or frozen.get('batch') != 64):
        raise ValueError('Expected a real B64 frozen first AR round')
    if (indices.ndim != 1 or indices.dtype != torch.long or indices.numel() != 48
            or len(set(indices.tolist())) != 48 or indices.min() < 0 or indices.max() >= 64):
        raise ValueError('Expected 48 distinct B64 request indices')
    def select(values):
        result = {}
        for name, tensor in values.items():
            axis = 2 if name in CACHE_FIELDS else 0
            if tensor.shape[axis] != 64:
                raise ValueError('Frozen request axis differs for ' + name)
            result[name] = tensor.index_select(axis, indices).contiguous()
        return result
    return select(frozen['inputs']), select(frozen['actual']), [frozen['cases'][i] for i in indices.tolist()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--bucket', type=Path, required=True, help='AR-only B48 plan mapping')
    parser.add_argument('--config', required=True)
    parser.add_argument('--gpu', type=int, default=7)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--indices', help='48 comma-separated original B64 request slots; default descending63..16')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    frozen = torch.load(args.capture, map_location='cpu', weights_only=True)
    index = torch.tensor([int(x) for x in args.indices.split(',')] if args.indices else list(range(63, 15, -1)), dtype=torch.long)
    inputs_cpu, source_actual, cases = slice_frozen(frozen, index)
    bucket = read_json(args.bucket)
    if bucket.get('batch') != 48:
        raise ValueError('This audit requires exact B48 AR-only plans')
    source = frozen['deployment']['resolved']
    if digest(frozen['calibration']['path']) != frozen['calibration']['sha256']:
        raise ValueError('Frozen source calibration changed')
    plans = {c: read_json(bucket[c + '_plan']) for c in ('target', 'draft')}
    for component in plans:
        validate_component_identity(plans[component], frozen['plans'][component], bucket, source, component)
    from inspark_infer.runtime.config import load
    from inspark_infer.runtime.device import GPULease, select_gpu
    from inspark_infer.runtime.engine import Engine
    from inspark_infer.runtime.asset_identity import validate_model_identities
    from inspark_infer.runtime.unified_deployment import install_reference_recipe
    from inspark_infer.quantization.unified import load_artifact
    from inspark_infer.ops.tensorrt.unified_ar import StaticARProvider
    from inspark_infer.ops.trtllm.adapter import OfficialRNNProposal
    from inspark_infer.ops.trtllm.official import native_rnn_factory
    from inspark_infer.ops.trtllm.pcg import categorical
    from trt113_provenance import capture_provenance
    config = load(args.config)
    # Config is loaded from a supported full deployment; only the independent
    # raw Engine/AR provider below uses48. No prefix/suffix/head graphs run.
    config.update(max_batch=48, target_tf32=False)
    artifact = load_artifact(bucket['calibration'], bucket['precision'])
    report = dict(schema=1, audit_kind='B48_frozen_history_reporting_only', numerical_thresholds=None,
        batch=48, source_batch=64, source_slots=index.tolist(), cases=cases,
        capture_sha256=digest(args.capture), bucket_sha256=digest(args.bucket),
        cache_source='same calibrated real B64 prefix frozen caches; request batch sliced noncontiguously',
        scope='one exact B48 Draft/RNN/Target verification; no B48 prefix/acoustic or sampling-quality claim',
        inherited_cache=True, rnn_initial_state='zero_per_round', conditional_q_chain='actual_B48_sampled_previous_tokens')
    with GPULease(args.gpu):
        select_gpu(args.gpu)
        engine = Engine(config)
        try:
            validate_model_identities(engine, bucket)
            report['weight_identity'] = {}
            for component in plans:
                observed = capture_provenance(component, config, args.config, model=engine)
                lhs = {x['role']: x['sha256'] for x in observed['model_sources']}
                rhs = {x['role']: x['sha256'] for x in plan_model_sources(plans[component], 48)}
                if lhs != rhs:
                    raise ValueError('Reference checkpoint differs from B48 ' + component)
                report['weight_identity'][component] = dict(verified=True, sha256_by_role=lhs)
            with torch.cuda.stream(engine.model.stream), torch.inference_mode():
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
                inputs = {k: v.cuda() for k, v in inputs_cpu.items()}
                provider = StaticARProvider(engine.rt, bucket['target_plan'], bucket['draft_plan'], 48)
                provider.target_cache.copy_(inputs['target_cache'])
                provider.target_keep.copy_(inputs['target_keep'])
                provider.draft_cache.copy_(inputs['draft_cache'])
                provider.active.fill_(True)
                hidden, base = provider.draft(inputs['anchors'], inputs['draft_positions'], inputs['draft_lengths'])
                proposal = OfficialRNNProposal(engine.rt.engine.draft, native_rnn_factory(),
                    provenance='native TensorRT-LLM RNNHead').graph_rewrite(compile_graph=True)
                tokens, probability, logits = proposal.sample_uniform(hidden, base, inputs['proposal_uniform'], inputs['anchors'])
                reproduced = torch.stack([categorical(probability[:, j], inputs['proposal_uniform'][:, j]) for j in range(7)], 1)
                if not torch.equal(tokens, reproduced):
                    raise RuntimeError('B48 actual conditional q/uniforms do not reproduce sampled chain')
                inputs['verify_tokens'] = torch.cat((inputs['anchors'][:, None], tokens), 1)
                target_model = engine.rt.engine.target.model
                inputs['target_x'] = target_model.embeddings(inputs['verify_tokens']) + target_model.text_pos_embedding.emb(inputs['target_positions'])
                target_logits, selected, final = provider.target(inputs['verify_tokens'], inputs['target_positions'], inputs['target_lengths'])
                actual = _cpu(dict(draft_hidden=hidden, draft_base=base, proposal_tokens=tokens,
                    proposal_q=probability, proposal_logits=logits, target_logits=target_logits,
                    target_selected=selected, target_final=final, target_kv_append=provider.target_append))
                if provider.draft_engine.calls != 1 or provider.target_engine.calls != 1:
                    raise RuntimeError('Expected one exact native B48 enqueue per AR engine')
                def references(bf16):
                    draft = draft_reference(engine.rt.backbone, inputs)
                    target = target_reference(engine.rt.engine.target, inputs, bf16)
                    rnn = conditional_rnn_reference(engine.rt.engine.draft, draft['draft_hidden'], draft['draft_base'],
                                                    inputs['anchors'], actual['proposal_tokens'].cuda())
                    return _cpu(dict(draft, **target, **rnn))
                high = references(False)
                mapping = _cpu(conditional_rnn_reference(engine.rt.engine.draft, hidden, base,
                    inputs['anchors'], tokens))
                reference_roles = install_reference_recipe(engine, artifact)
                same = references(True)
            report['execution'] = dict(target_enqueues=1, draft_enqueues=1, engine_batch=48,
                kv_capacity=provider.capacity, prefix_enqueues=0, acoustic_enqueues=0, fallback=False,
                engines={c: dict(plan=str(Path(bucket[c + '_plan']).resolve()), sha256=plans[c]['sha256']) for c in plans})
            report['reference_roles'] = [r for r in reference_roles if r['path'].startswith(('target.', 'draft.'))]
            report['outputs'] = {name: dict(trt_vs_same_recipe=metrics(same[name], value),
                trt_vs_original_fp32_on_frozen_cache=metrics(high[name], value),
                same_recipe_vs_original_fp32=metrics(high[name], same[name]))
                for name, value in actual.items() if name != 'proposal_tokens'}
            report['rnn_adapter_on_identical_trt_hidden'] = {name: metrics(value, actual[name]) for name, value in mapping.items()}
            comparisons = [metric for row in report['outputs'].values() for metric in row.values()]
            comparisons += list(report['rnn_adapter_on_identical_trt_hidden'].values())
            if not all(m['finite'] for m in comparisons):
                raise RuntimeError('B48 audit contains nonfinite output/reference')
            report['conditional_q'] = dict(temperature=0.8, tokens=actual['proposal_tokens'].tolist(),
                probability_sums=actual['proposal_q'].sum(-1).tolist(), reproduced=True)
            report['source_shape_diagnostic'] = dict(
                draft_hidden=metrics(source_actual['draft_hidden'], actual['draft_hidden']),
                draft_base=metrics(source_actual['draft_base'], actual['draft_base']),
                proposal_tokens_equal=bool(torch.equal(source_actual['proposal_tokens'], actual['proposal_tokens'])),
                caveat='Only Draft backbone sees identical inputs; Target sampled chain may differ across shapes')
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
            print(json.dumps(dict(output=str(args.output), batch=48, finite=True, numerical_gate=False)), flush=True)
        finally:
            engine.close()


if __name__ == '__main__':
    main()
