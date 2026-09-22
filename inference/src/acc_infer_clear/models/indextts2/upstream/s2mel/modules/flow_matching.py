"""Weight-compatible DiT holder; solver lives in acc_infer_clear.models.indextts2.cfm."""
from torch import nn
from acc_infer_clear.models.indextts2.upstream.s2mel.modules.diffusion_transformer import DiT

class CFM(nn.Module):
    def __init__(self,args):
        super().__init__()
        if args.dit_type!='DiT':raise ValueError('Universal baseline requires DiT')
        self.estimator=DiT(args)
