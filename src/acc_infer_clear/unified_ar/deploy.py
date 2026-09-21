"""Single production entry point for UnifiedQKV/Attention/Linear/ContextKV."""
def attach_target(engine):
    from acc_infer_clear.unified_ar.target import attach as attention
    from acc_infer_clear.unified_ar.linear import attach as linear
    attention(engine,mode=1);linear(engine)
    return dict(qk='fp8_block32',v='fp32',softmax='fp32',stage=2,
                batches=list(range(1,9)),online_tuning=False)

def attach_draft(engine):
    from acc_infer_clear.unified_ar.draft import attach
    return attach(engine)
