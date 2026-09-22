#!/usr/bin/env python3
"""Same-input fused acceptance/prefix audit; not a full-model trajectory audit.

Only the actual ASG checkpoint is loaded. CPU --reference-only validates fixture
construction and existing reference logic; it never counts as a fused GPU pass.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]
SCENARIOS=('random','all_accept','reject_first','reject_middle','reject_last',
           'eos_first','eos_middle','eos_last','rejected_eos','eos_after_rejection',
           'remaining_zero','remaining_one','remaining_three','reject_at_cap',
           'eos_outside_budget','eos_at_budget_end','all_accept_exact_cap','mixed_rows')
MIXED=('all_accept','reject_first','eos_middle','remaining_zero',
       'remaining_three','reject_at_cap','eos_outside_budget','rejected_eos')
INPUTS=('logits','p','tokens','group_draws','accept_draws','remaining','current')


def file_hash(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def make_fixture(groups,batch,scenario,seed,eos,max_tokens=1500):
    """Generate on CPU once; both CUDA implementations receive these exact bits."""
    import torch
    if scenario not in SCENARIOS:raise ValueError('Unknown scenario')
    if batch not in (1,4,8):raise ValueError('Supported batches are 1/4/8')
    vocab=groups.token_group_counts.numel()
    if not 0<=eos<vocab or max_tokens<8:raise ValueError('Invalid EOS or total token limit')
    gen=torch.Generator().manual_seed(seed%(2**64))
    logits=torch.randn(batch,7,vocab,generator=gen).clamp(-2,2)
    p=torch.rand(batch,7,vocab,generator=gen)+.01
    tokens=torch.randint(vocab,(batch,7),generator=gen)
    tokens[tokens==eos]=(eos+1)%vocab
    group_draws=torch.rand(batch,7,generator=gen)
    group_draws[:,0]=0;group_draws[:,-1]=torch.nextafter(torch.tensor(1.),torch.tensor(0.))
    accept_draws=torch.rand(batch,7,generator=gen)
    remaining=torch.full((batch,),7,dtype=torch.int32)
    current=torch.full((batch,),min(10,max_tokens-8),dtype=torch.int32)
    expected_flags=torch.ones(batch,7,dtype=torch.bool)
    roles=[]
    for row in range(batch):
        role=MIXED[row%len(MIXED)] if scenario=='mixed_rows' else scenario
        roles.append(role)
        if role=='random':continue
        accept_draws[row].zero_()
        eos_position={'eos_first':0,'eos_middle':3,'eos_last':6,'rejected_eos':2,
                      'eos_after_rejection':5,'eos_outside_budget':3,'eos_at_budget_end':2}.get(role)
        if eos_position is not None:tokens[row,eos_position]=eos
        budget={'remaining_zero':0,'remaining_one':1,'remaining_three':3,'reject_at_cap':3,
                'eos_outside_budget':3,'eos_at_budget_end':3,'all_accept_exact_cap':7}.get(role)
        if budget is not None:remaining[row]=budget;current[row]=max_tokens-budget
        reject={'reject_first':0,'reject_middle':3,'reject_last':6,'rejected_eos':2,
                'eos_after_rejection':1,'reject_at_cap':2}.get(role)
        if reject is not None:
            token=tokens[row,reject]
            choice=(group_draws[row,reject]*groups.token_group_counts[token].float()).floor().long()
            group=groups.token_groups[token,choice]
            size=int(groups.group_sizes[group]);members=groups.group_members[group,:size]
            if size>=vocab:raise ValueError('Cannot force rejection for an all-vocabulary ASG group')
            # A comfortably separated decision, not an artificial ULP tie.
            logits[row,reject,members]=-8
            p[row,reject].fill_(1e-4);p[row,reject,members]=1
            accept_draws[row,reject]=.75;expected_flags[row,reject]=False
    p=p/p.sum(-1,keepdim=True)
    return dict(logits=logits,p=p,tokens=tokens,group_draws=group_draws,accept_draws=accept_draws,
                remaining=remaining,current=current,roles=roles,expected_flags=expected_flags)


def reference_prefix(packed,remaining,current,eos,max_tokens):
    import torch
    from acc_infer_clear.models.indextts2.dspark.logic import accepted_prefix
    result=[]
    for decisions,budget,used in zip(packed.detach().cpu().tolist(),remaining.cpu().tolist(),current.cpu().tolist()):
        n,ended=accepted_prefix(decisions,int(budget),eos)
        correction=not ended and int(used)+n<max_tokens
        result.append((n,ended,correction,correction and n<int(budget)))
    return (torch.tensor([row[0] for row in result],dtype=torch.int32),
            *(torch.tensor([row[index] for row in result],dtype=torch.bool) for index in range(1,4)))


def exact(expected,actual):
    import torch
    x,y=expected.detach().cpu(),actual.detach().cpu()
    result=dict(pass_gate=False,expected_shape=list(x.shape),actual_shape=list(y.shape))
    if x.shape!=y.shape:return dict(result,reason='shape mismatch')
    if not x.numel():return dict(result,reason='empty metadata')
    mismatch=x!=y;count=int(mismatch.sum())
    result.update(pass_gate=count==0,mismatched_elements=count)
    if count:
        index=tuple(int(value) for value in mismatch.nonzero()[0])
        result['first_mismatch']=dict(index=list(index),expected=x[index].item(),actual=y[index].item())
    return result


def plan_report(expected,actual):
    names=('n','eos','correction','residual')
    metrics={name:exact(x,y) for name,x,y in zip(names,expected,actual)}
    return dict(pass_gate=all(value['pass_gate'] for value in metrics.values()),metrics=metrics,
                expected={name:value.cpu().tolist() for name,value in zip(names,expected)},
                actual={name:value.detach().cpu().tolist() for name,value in zip(names,actual)})


def check_fixture(fixture,packed):
    expected=fixture['expected_flags'];actual=packed.detach().cpu()[:,:,0].bool()
    controlled=[index for index,role in enumerate(fixture['roles']) if role!='random']
    if controlled and not bool((actual[controlled]==expected[controlled]).all()):
        raise RuntimeError('Reference decisions did not realize the declared fixture scenario')


def audit_case(groups,fixture,eos,max_tokens,candidate=None,prefix=None):
    import torch
    from acc_infer_clear.guardrails.numerics import compare
    from acc_infer_clear.runtime.indextts2.batch_pcg import BatchedAcceptance
    args=tuple(fixture[name] for name in INPUTS[:5])
    before={name:tensor_hash(fixture[name]) for name in INPUTS}
    reference=BatchedAcceptance(groups,.8).tensor_body(*args)
    check_fixture(fixture,reference[2])
    expected_plan=reference_prefix(reference[2],fixture['remaining'],fixture['current'],eos,max_tokens)
    result=dict(input_hashes=before,roles=fixture['roles'],
                remaining=fixture['remaining'].cpu().tolist(),current=fixture['current'].cpu().tolist(),
                reference_flags=reference[2][:,:,0].cpu().tolist(),
                reference_prefix={name:value.tolist() for name,value in zip(('n','eos','correction','residual'),expected_plan)},
                reference_fixture_passed=True,fused_tested=candidate is not None)
    if candidate is None:return result
    actual=candidate(groups,.8,*args)
    floating={name:compare(x,y,'fp32') for name,x,y in (
        ('q',reference[0],actual[0]),('accept',reference[1],actual[1]),
        ('packed_accept',reference[2][:,:,3],actual[2][:,:,3]),
        ('packed_exact_token_accept',reference[2][:,:,4],actual[2][:,:,4]))}
    metadata={name:exact(reference[2][:,:,index],actual[2][:,:,index])
              for index,name in enumerate(('flags','tokens','group_sizes'))}
    prefix_reference=prefix(reference[2],fixture['remaining'],fixture['current'],eos,max_tokens)
    prefix_actual=prefix(actual[2],fixture['remaining'],fixture['current'],eos,max_tokens)
    actual_host=reference_prefix(actual[2],fixture['remaining'],fixture['current'],eos,max_tokens)
    plans=dict(reference_packed=plan_report(expected_plan,prefix_reference),
               candidate_packed=plan_report(actual_host,prefix_actual),
               end_to_end=plan_report(expected_plan,prefix_actual))
    result.update(floating=floating,discrete=metadata,prefix=plans,
                  inputs_unchanged=all(tensor_hash(fixture[name])==before[name] for name in INPUTS))
    result['pass_gate']=(all(value['pass_gate'] for value in (*floating.values(),*metadata.values(),*plans.values()))
                         and result['inputs_unchanged'])
    return result


def workspace_path(value):
    path=Path(value).resolve()
    if not path.is_relative_to('/workspace'):raise ValueError('Paths must stay inside /workspace')
    return path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu',type=int,help='Required for CUDA; this task uses physical GPU 6')
    parser.add_argument('--reference-only',action='store_true',help='CPU fixtures/reference only; never a fused CUDA audit pass')
    parser.add_argument('--asg',type=workspace_path,default=ROOT.parent/'models/asg/target_asg_threshold_0p49.safetensors')
    parser.add_argument('--batches',nargs='+',type=int,choices=(1,4,8),default=[1,4,8])
    parser.add_argument('--seed',type=int,default=20260923)
    parser.add_argument('--eos',type=int,default=8193,help='Actual IndexTTS2 stop_mel_token, not start token 8192')
    parser.add_argument('--max-tokens',type=int,default=1500)
    parser.add_argument('--output',type=workspace_path,required=True)
    args=parser.parse_args()
    if args.max_tokens<8:parser.error('--max-tokens must be at least 8')
    if args.reference_only:
        os.environ['CUDA_VISIBLE_DEVICES']='';lease=nullcontext();device='cpu'
    else:
        if args.gpu is None:parser.error('--gpu is required unless --reference-only is set')
        from acc_infer_clear.runtime.device import GPULease,select_gpu
        select_gpu(args.gpu);lease=GPULease(args.gpu);device='cuda'
    cache=ROOT.parent/'.cache/acceptance_metadata_audit';cache.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('TRITON_CACHE_DIR',str(cache/'triton'))
    os.environ['TMPDIR']=str(cache)
    from acc_infer_clear.runtime.config import atomic_json
    report=dict(scope='controlled_identical_inputs_acceptance_and_prefix_not_full_model_trajectory',
        status='running',reference_only=args.reference_only,seed=args.seed,eos=args.eos,max_tokens=args.max_tokens,
        temperature=.8,batches=args.batches,tolerances=dict(atol=1e-5,rtol=1e-4),
        exact_fields=['flags','tokens','group_sizes','n','eos','correction','residual'],cases=[])
    started=time.perf_counter()
    try:
        with lease:
            import torch
            from acc_infer_clear.models.indextts2.pcg.asg import AcousticGroups
            torch.set_num_threads(4)
            sparse=AcousticGroups.load(args.asg,device='cpu');cpu_groups=sparse.dense('cpu')
            groups=cpu_groups if args.reference_only else sparse.dense('cuda')
            report['asg']=dict(path=str(args.asg),sha256=file_hash(args.asg),vocab_size=sparse.vocab_size,
                               num_groups=sparse.num_groups,max_group_members=int(groups.group_members.shape[1]))
            report['software']=dict(torch=torch.__version__,cuda=torch.version.cuda,triton=importlib.metadata.version('triton'))
            report['source_hashes']={str(path.relative_to(ROOT)):file_hash(path) for path in (
                Path(__file__),ROOT/'src/acc_infer_clear/ops/triton/acceptance.py',
                ROOT/'src/acc_infer_clear/runtime/indextts2/batch_pcg.py',
                ROOT/'src/acc_infer_clear/models/indextts2/dspark/logic.py')}
            candidate=prefix=None
            if not args.reference_only:
                from acc_infer_clear.ops.triton.acceptance import acceptance,prefix_plan
                candidate,prefix=acceptance,prefix_plan
                major,minor=torch.cuda.get_device_capability(0)
                report['hardware']=dict(physical_gpu=args.gpu,name=torch.cuda.get_device_name(0),sm=major*10+minor)
            for batch in args.batches:
                for index,scenario in enumerate(SCENARIOS):
                    fixture=make_fixture(cpu_groups,batch,scenario,args.seed+batch*100+index*10000,args.eos,args.max_tokens)
                    for name in INPUTS:fixture[name]=fixture[name].to(device)
                    with torch.inference_mode():result=audit_case(groups,fixture,args.eos,args.max_tokens,candidate,prefix)
                    report['cases'].append(dict(batch=batch,scenario=scenario,**result))
            report['cuda_initialized']=torch.cuda.is_initialized()
        report['fixture_passed']=all(case['reference_fixture_passed'] for case in report['cases'])
        report['pass_gate']=None if args.reference_only else all(case['pass_gate'] for case in report['cases'])
        report['status']='reference_only_passed' if args.reference_only else ('passed' if report['pass_gate'] else 'failed')
    except Exception as error:
        report['status']='failed';report['pass_gate']=False
        report['error']=dict(type=type(error).__name__,message=str(error))
        raise
    finally:
        report['elapsed_s']=time.perf_counter()-started;atomic_json(args.output,report)
    print(json.dumps(dict(status=report['status'],cases=len(report['cases']),pass_gate=report['pass_gate'])))
    if report['status']=='failed':raise SystemExit(1)


if __name__=='__main__':main()
