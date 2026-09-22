"""Hash-pinned interval student deployment export; fixed two-step integration."""
import torch
from torch import nn
from acc_infer_clear.runtime.assets import sha256

class IntervalEmbedding(nn.Module):
    def __init__(self,original):
        super().__init__();self.original=original
        dim=original.mlp[-1].out_features
        self.fusion=nn.Linear(2*dim,dim,bias=False,device=next(original.parameters()).device)
    def forward(self,times):
        t,r=times.unbind(-1)
        return self.fusion(torch.cat((self.original(t),self.original(r)),dim=-1))

class Solver:
    def __init__(self,model,path,expected_sha,max_batch):
        actual=sha256(path)
        if actual!=expected_sha:raise ValueError('Student checkpoint hash mismatch')
        saved=torch.load(path,map_location='cpu',weights_only=True)
        if saved.get('format_version')!=1:raise ValueError('Expected deployment student export, not a training checkpoint')
        step=int(saved['step'])
        if step<0:raise ValueError('Invalid student training step')
        source=saved.get('source',dict(path=str(path),sha256=actual))
        for name in ('t_embedder','t_embedder2'):
            if hasattr(model,name):setattr(model,name,IntervalEmbedding(getattr(model,name)))
        model.load_state_dict(saved['student'],strict=True);del saved
        self.model=model.float().eval().requires_grad_(False);self.model.setup_caches(max_batch,8192)
        device=next(model.parameters()).device
        self.times=tuple(torch.tensor([[t,r]],device=device) for t,r in ((0.,.5),(.5,1.)))
        self.identity=dict(path=str(path),sha256=actual,source=source,step=step,intervals=[[0,.5],[.5,1]],cfg=0,precision='FP32')
        self.observer=None
    def __call__(self,x,prompt,lengths,style,mu,mask):
        x=x.float().masked_fill(mask,0)
        for interval,times in enumerate(self.times):
            if self.observer is not None:self.observer(interval)
            v=self.model(x,prompt,lengths,times.expand(x.shape[0],-1),style,mu)
            x=(x+.5*v.float()).masked_fill(mask,0)
        if self.observer is not None:self.observer(None)
        return x
