from contextlib import contextmanager
import time
from acc_infer_clear.runtime.indextts2.core import ARCore
from acc_infer_clear.models.indextts2.dspark.target import sample_logits
from acc_infer_clear.runtime.indextts2.batch_target import BatchedTarget
from acc_infer_clear.runtime.indextts2.batch_draft import BatchedDraftBackbone
from acc_infer_clear.runtime.indextts2.batch_context import BatchedContextAppend
from acc_infer_clear.runtime.indextts2.batch_latent import BatchedLatent
from acc_infer_clear.runtime.indextts2.batch_pcg import BatchedAcceptance, BatchedResidual
from acc_infer_clear.runtime.indextts2.proposal import Proposal
from acc_infer_clear.models.indextts2.batch_frontend import BatchFrontend
from acc_infer_clear.models.indextts2.pcg.sampler import sample_pcg_residual_vectorized

class Runtime(ARCore):
    def __init__(self,model):
        self.model=model;self.tts=model.tts;self.engine=model.engine
        self.device=next(self.engine.draft.parameters()).device
        self.frontend=BatchFrontend(model,workers=model.config['cpu_threads']);self.frontend.prime_voices()
        self.target=BatchedTarget(self.engine.target)
        self.backbone=BatchedDraftBackbone(self.engine.draft);self.proposal=Proposal(self.backbone)
        self.context=BatchedContextAppend(self.engine.draft)
        self.latent=BatchedLatent(self.tts.gpt,self.tts.gpt.get_logits)
        self.accept=BatchedAcceptance(self.engine.dense_groups,.8)
        self.residual=BatchedResidual(self.engine.dense_groups,sample_pcg_residual_vectorized)
        self.events=[]
        self.profile_cuda=False;self.profile_spans=[]
        self.native_target_steps=0
    @contextmanager
    def span(self,name,rows):
        start_host=time.perf_counter();cuda_start=cuda_end=None
        if getattr(self,'profile_cuda',False):
            import torch
            cuda_start=torch.cuda.Event(enable_timing=True);cuda_end=torch.cuda.Event(enable_timing=True);cuda_start.record()
        if getattr(self,'trace_ranges',False):
            import torch
            with torch.profiler.record_function(name):
                torch.cuda.nvtx.range_push(name)
                try:yield
                finally:torch.cuda.nvtx.range_pop()
        else:yield
        if cuda_end is not None:
            cuda_end.record();self.profile_spans.append(dict(name=name,batch=len(rows),start=cuda_start,end=cuda_end,
                host_ms=(time.perf_counter()-start_host)*1000,metadata={}))
    def sample(self,row,logits):
        token,_=sample_logits(logits,.8,generator=row.generator)
        row.state['cuda_rng']=row.generator.get_state();return token.reshape(1).clone()
    def close(self):self.frontend.close()
