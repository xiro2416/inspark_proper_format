"""Weight-compatible DiT holder; solver lives in inspark_infer.models.indextts2.cfm."""
from torch import nn
from inspark_infer.models.indextts2.upstream.s2mel.modules.diffusion_transformer import DiT

class CFM(nn.Module):
    def __init__(self,args):
        super().__init__()
        if args.dit_type!='DiT':raise ValueError('Universal baseline requires DiT')
        self.estimator=DiT(args)
