"""Device-resident speculative token commit for fixed seven-slot proposals."""
import torch
import triton
import triton.language as tl

@triton.jit
def _commit(PROPOSED,CORRECTION,COUNTS,ENDS,HAS_CORRECTION,ACTIVE,BUFFER,LENGTHS,
            PAST,ACCEPTED,ROUND,DONE,COMMITTED,LAST,B:tl.constexpr,
            CAPACITY:tl.constexpr,EOS:tl.constexpr,MAX_TOKENS:tl.constexpr):
    row=tl.program_id(0);j=tl.arange(0,8);active=tl.load(ACTIVE+row);old=tl.load(LENGTHS+row);n=tl.load(COUNTS+row)
    valid=(j<n)&active;token=tl.load(PROPOSED+row*7+j,mask=j<7,other=0)
    tl.store(BUFFER+row*CAPACITY+old+j,token,mask=valid&(old+j<CAPACITY))
    correction=tl.load(HAS_CORRECTION+row)&active;ct=tl.load(CORRECTION+row)
    tl.store(BUFFER+row*CAPACITY+old+n,ct,mask=correction&(old+n<CAPACITY))
    n=tl.where(active,n,0);new_length=old+n+correction.to(tl.int32)
    prior_last=tl.load(LAST+row);accepted_last=tl.sum(tl.where(j==n-1,token,0),axis=0)
    last=tl.where(correction,ct,tl.where(n>0,accepted_last,prior_last))
    end=tl.load(ENDS+row)&active;prior_done=tl.load(DONE+row);done=prior_done|end|(active&((last==EOS)|(new_length>=MAX_TOKENS)))
    round=tl.load(ROUND+row)
    tl.store(ACCEPTED+row*CAPACITY+round,n,mask=active&(round<CAPACITY))
    tl.store(ROUND+row,round+active.to(tl.int32));tl.store(LENGTHS+row,new_length)
    committed=tl.where(active,n+1,0);tl.store(PAST+row,tl.load(PAST+row)+committed);tl.store(COMMITTED+row,committed)
    tl.store(LAST+row,last);tl.store(DONE+row,done)

def commit(proposed,correction,counts,ends,has_correction,token_buffer,
           token_lengths,past_lengths,accepted_history,rounds,done,committed,last,
           eos,max_tokens,active=None):
    b=proposed.shape[0]
    if proposed.shape!=(b,7) or token_buffer.shape[0]!=b:
        raise ValueError('Device commit shape mismatch')
    if active is None:active=torch.ones_like(done)
    _commit[(b,)](proposed,correction,counts,ends,has_correction,active,token_buffer,
        token_lengths,past_lengths,accepted_history,rounds,done,committed,last,
        b,token_buffer.shape[1],int(eos),int(max_tokens),num_warps=1)

@triton.jit
def _mark_keep(KEEP,SLOTS,LENGTHS,B:tl.constexpr,CAPACITY:tl.constexpr):
    row=tl.program_id(0);j=tl.arange(0,8);slot=tl.load(SLOTS+row);start=tl.load(LENGTHS+row)
    tl.store(KEEP+slot*CAPACITY+start+j,1,mask=start+j<CAPACITY)

def mark_keep(keep,slots,lengths):
    b=slots.numel();_mark_keep[(b,)](keep,slots,lengths,b,keep.shape[1],num_warps=1)

@triton.jit
def _status(READY,FAILURES,OUT,PAST,DRAFT,INITIAL:tl.constexpr,B:tl.constexpr,BLOCK:tl.constexpr,CHECK_CAPACITY:tl.constexpr):
    i=tl.arange(0,BLOCK);all_ready=tl.sum(tl.load(READY+i,mask=i<B,other=1).to(tl.int32),axis=0)==BLOCK
    failed=tl.load(FAILURES)>INITIAL
    capacity=tl.full((),0,tl.int32)
    if CHECK_CAPACITY:
        # The Target verification and Context commit can each write eight
        # positions, including for rows evaluated inside a fixed-batch graph.
        past=tl.load(PAST+i,mask=i<B,other=0)
        draft=tl.load(DRAFT+i,mask=i<B,other=0)
        capacity=tl.sum(((i<B)&((past+8>128)|(draft+8>128))).to(tl.int32),axis=0)>0
    tl.store(OUT,all_ready.to(tl.int32)|(failed.to(tl.int32)<<1)|(capacity.to(tl.int32)<<2))

def status(ready,failures,out,initial,past=None,draft_lengths=None):
    if (past is None)!=(draft_lengths is None):raise ValueError('Both cache lengths are required')
    check=past is not None
    block=triton.next_power_of_2(ready.numel())
    _status[(1,)](ready,failures,out,past if check else ready,
        draft_lengths if check else ready,int(initial),ready.numel(),block,check,num_warps=1)

def prepare(device,eos,max_tokens,batches=(8,)):
    """Compile selected Batch commit/keep/status signatures before admission."""
    capacity=max_tokens+64
    for b in batches:
        proposed=torch.zeros(b,7,device=device,dtype=torch.long);correction=torch.zeros(b,device=device,dtype=torch.long)
        counts=torch.zeros(b,device=device,dtype=torch.int32);flags=torch.zeros(b,device=device,dtype=torch.bool);active=torch.ones_like(flags)
        buffer=torch.zeros(b,capacity,device=device,dtype=torch.long);lengths=torch.ones(b,device=device,dtype=torch.int32)
        past=torch.ones_like(lengths);history=torch.zeros(b,capacity,device=device,dtype=torch.int32);rounds=torch.zeros_like(lengths)
        done=torch.zeros_like(flags);committed=torch.zeros_like(lengths);last=torch.zeros(b,device=device,dtype=torch.long)
        commit(proposed,correction,counts,flags,flags,buffer,lengths,past,history,rounds,done,committed,last,eos,max_tokens,active)
        keep=torch.zeros(2*max(b,8),2048,device=device,dtype=torch.int32);slots=torch.arange(b,device=device,dtype=torch.int32);mark_keep(keep,slots,past)
        state=torch.zeros((),device=device,dtype=torch.int32);status(done,torch.zeros_like(state),state,0)
        status(done,torch.zeros_like(state),state,0,past,past)
    torch.cuda.current_stream().synchronize()
    return dict(batches=list(batches),commit_capacity=capacity,keep_capacity=2048,online_compile=False)
