#!/usr/bin/env python3
"""Controlled-input CPU/CUDA request-RNG audit, separate from model numerics.

Exercises actual Proposal, acceptance, residual (including enumeration fallback)
and correction sampling adapters. The deterministic tiny backbone/RNN and ASG
fixture deliberately remove neural floating-point/batch differences.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS


def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def run_scenario(device,batch,rounds,seed,shared=False,churn=False):
    import torch
    from acc_infer_clear.models.indextts2.pcg.asg import AcousticGroups
    from acc_infer_clear.models.indextts2.pcg.sampler import sample_pcg_residual_vectorized
    from acc_infer_clear.runtime.indextts2.batch_pcg import BatchedAcceptance,BatchedResidual
    from acc_infer_clear.runtime.indextts2.proposal import Proposal
    from acc_infer_clear.runtime.indextts2.runtime import Runtime
    vocab=8
    sparse=AcousticGroups(group_offsets=torch.arange(vocab+1),group_members=torch.arange(vocab),
                          token_offsets=torch.arange(vocab+1),token_groups=torch.arange(vocab),
                          membership_count=torch.ones(vocab,dtype=torch.long),metadata={})
    groups=sparse.dense(device)
    logits=torch.linspace(-1,1,vocab,device=device)
    model=NS(vocab_size=vocab,markov_state_size=2)
    model._prepare_optimized_linear_rnn=lambda hidden:(None,None,hidden.new_zeros(hidden.shape[0],7,2))
    model._optimized_linear_rnn_step=lambda state,previous,*args:(state,state.new_zeros(state.shape[0],vocab))
    proposal=Proposal.__new__(Proposal)
    proposal.model=model;proposal.graphs={};proposal.graph_hits=0;proposal.calls=0
    proposal.hidden_linear=None;proposal.state_linear=None;proposal.output_linear=None
    proposal.weight=None;proposal.table=None;proposal.out_weight=None
    proposal.batched_rng=shared;proposal.batched_rng_min_batch=1
    proposal.batch_generator=torch.Generator(device=device).manual_seed(0x5EED5EED)
    proposal.backbone=lambda jobs:[(torch.zeros(1,7,2,device=device),logits[None,None,:].expand(1,7,-1).clone()) for job in jobs]
    acceptance=BatchedAcceptance(groups,.8)
    residual=BatchedResidual(groups,sample_pcg_residual_vectorized)
    recorded={};watched_index=0
    original_proposal=proposal.math
    def proposal_math(base,terms,noise,previous):
        recorded['proposal_exponential']=tensor_hash(noise[watched_index])
        return original_proposal(base,terms,noise,previous)
    proposal.math=proposal_math
    original_accept=acceptance.tensor_body
    def acceptance_math(values,p,tokens,group_draws,accept_draws):
        recorded['accept_group_uniform']=tensor_hash(group_draws[watched_index])
        recorded['accept_uniform']=tensor_hash(accept_draws[watched_index])
        return original_accept(values,p,tokens,group_draws,accept_draws)
    acceptance.tensor_body=acceptance_math
    def task(value):
        gen=torch.Generator(device=device).manual_seed(value)
        return NS(generator=gen,state={'cuda_rng':gen.get_state()},codes=[torch.tensor([1],device=device)])
    watched=task(seed);others=[task(seed+index+100) for index in range(batch-1)]
    traces=[];tokens=[]
    for index in range(rounds):
        if churn:
            # Previous unrelated tasks are cancelled; their replacement may reuse the logical lane.
            others=[task(seed+index*31+offset+1000) for offset in range(batch-1)]
            torch.rand(257,device=device)  # unrelated global RNG consumption must not affect the request
        watched_index=index%batch if churn else batch-1
        tasks=others[:watched_index]+[watched]+others[watched_index:]
        recorded={}
        clone=torch.Generator(device=device);clone.set_state(watched.state['cuda_rng'])
        recorded['next_uniform_prefix']=tensor_hash(torch.rand(32,device=device,generator=clone))
        recorded['round_start_state']=tensor_hash(watched.state['cuda_rng'])
        jobs=[dict(cache=None,anchor_token=row.codes[-1],first_position=1,temperature=.8) for row in tasks]
        proposed=proposal(jobs,tasks)
        acceptance([(out[2][0],out[1][0],out[0][0]) for out in proposed],tasks)
        recorded['after_acceptance_state']=tensor_hash(watched.state['cuda_rng'])
        # Alternate true zero-residual enumeration with positive-residual thinning.
        q=torch.softmax(logits,dim=-1)
        p=q.clone() if index%2==0 else q.flip(0)
        selected=residual([((q,p,groups),dict(max_thinning_attempts=64)) for row in tasks],tasks)
        recorded['after_residual_state']=tensor_hash(watched.state['cuda_rng'])
        watched.generator.set_state(watched.state['cuda_rng'])
        correction=Runtime.sample(None,watched,logits[None])
        watched.codes.append(correction)
        recorded['round_end_state']=tensor_hash(watched.state['cuda_rng'])
        traces.append(dict(round=index,hashes=recorded))
        tokens.append(dict(proposal=proposed[watched_index][0].cpu().tolist(),
                           residual=int(selected[watched_index][0]),correction=int(correction[0]),
                           residual_enumeration=bool(selected[watched_index][2])))
    return dict(batch=batch,churn=churn,shared_rng=shared,traces=traces,controlled_tokens=tokens,
                residual_fallbacks=residual.fallbacks)


def audit(device='cpu',rounds=4,seed=123):
    import torch
    baseline=run_scenario(device,1,rounds,seed)
    results=[]
    for batch,churn in ((1,True),(4,False),(4,True),(8,False),(8,True)):
        value=run_scenario(device,batch,rounds,seed,churn=churn)
        value['random_stream_exact']=value['traces']==baseline['traces']
        value['controlled_tokens_exact']=value['controlled_tokens']==baseline['controlled_tokens']
        results.append(value)
    reused=run_scenario(device,1,rounds,seed)
    shared_one=run_scenario(device,1,rounds,seed,shared=True)
    shared_many=run_scenario(device,8,rounds,seed,shared=True,churn=True)
    negative_control=shared_one['traces']!=shared_many['traces']
    passed=all(row['random_stream_exact'] and row['controlled_tokens_exact'] for row in results)
    passed=passed and reused['traces']==baseline['traces'] and negative_control
    hardware=None
    if device=='cuda':
        major,minor=torch.cuda.get_device_capability(0)
        hardware=dict(name=torch.cuda.get_device_name(0),sm=major*10+minor,logical_device=0)
    return dict(scope='controlled_sampler_rng_isolation_not_real_model_numerical_parity',
        passed=passed,device=device,seed=seed,rounds=rounds,policy='legacy_per_request',
        torch_version=torch.__version__,cuda_version=torch.version.cuda,hardware=hardware,
        fixture_vocab_size=8,cuda_initialized=torch.cuda.is_initialized(),
        baseline=baseline,scenarios=results,cancel_recreate_same_seed_exact=reused['traces']==baseline['traces'],
        shared_rng_negative_control_detected=negative_control,
        interpretation='Fixed q/p/logits and identical branch schedule isolate RNG ownership. '
                       'Real eager/TRT floating-point differences can change acceptance, subsequent draw '
                       'consumption and tokens; those require separate numerical/codes audits.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    parser.add_argument('--gpu',type=int,default=None)
    parser.add_argument('--rounds',type=int,default=4)
    parser.add_argument('--seed',type=int,default=123)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.rounds<2:parser.error('At least two rounds are needed for both residual branches')
    if not args.output.resolve().is_relative_to('/workspace'):parser.error('Output must stay inside /workspace')
    if args.device=='cuda':
        if args.gpu is None:parser.error('--gpu is required for CUDA')
        from acc_infer_clear.runtime.device import GPULease,select_gpu
        select_gpu(args.gpu);lease=GPULease(args.gpu)
    else:
        os.environ['CUDA_VISIBLE_DEVICES']='';lease=nullcontext()
    with lease:
        report=audit(args.device,args.rounds,args.seed)
    report['runner_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    from acc_infer_clear.runtime.config import atomic_json
    atomic_json(args.output,report)
    print(json.dumps({key:report[key] for key in ('passed','device','policy','shared_rng_negative_control_detected')}))
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':main()
