"""Weight-compatible DiT holder; inference solver lives in acc_infer_clear.cfm."""
from torch import nn
from .diffusion_transformer import DiT

class CFM(nn.Module):
    def __init__(self,args):
        super().__init__()
        if args.dit_type!='DiT':raise ValueError('Universal baseline requires DiT')
        self.estimator=DiT(args)
