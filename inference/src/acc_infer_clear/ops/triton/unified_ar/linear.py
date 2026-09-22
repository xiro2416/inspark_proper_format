"""Role-fixed UnifiedLinear deployment for Target FP8 layers."""
import types
from acc_infer_clear.ops.triton.target_full_m.tiled import run_prequantized
from acc_infer_clear.ops.triton.target_norm_quant.residual import col_linear_residual
from acc_infer_clear.ops.triton.target_norm_quant.residual_down import run as down_residual

class UnifiedUp:
    def __init__(self,dense):self.col,self.scales,self.bias=dense.col,dense.scales,dense.bias
    def __call__(self,q,s,shape):
        m=q.shape[0];bm=max(16,1<<(m-1).bit_length());warps=4 if bm<=32 else 8
        plan=dict(mode='tiled',bm=bm,bn=64,bk=128,stages=6,warps=warps,wm=2,wn=warps//2,schedule='full',swizzle=True)
        return run_prequantized(q,s,shape,self.col,self.scales,self.bias,plan)

class UnifiedResidual:
    def __init__(self,old,role):
        self.role=role;self.scales=old.raw.scales;self.bias=old.raw.bias
        values=[x for x in old.bindings.values() if x is not None]
        if not values:raise RuntimeError('Unified residual requires N,K column weights')
        self.col=values[0]
    def __call__(self,x,residual):
        if self.role=='out':
            return col_linear_residual(x,residual,self.col,self.scales,self.bias,dict(bm=16,bn=64,bk=128,warps=4,stages=4))
        return down_residual(x,residual,self.col,self.scales,self.bias,dict(bm=16,bn=64,bk=128,stages=5,swizzle=True,inner=False))

def attach(engine):
    previous=engine.prepare_slot_target
    def prepare_slot_target(graphs=False):
        previous(graphs=False);target=engine.rt.target;changed=[]
        for index in range(len(target.target.model.transformer.h)):
            up=target._norm_quant_pairs.get((index,'up'))
            if up is not None:up.dense=UnifiedUp(up.dense);changed.append(f'target.{index}.up')
            for role in ('out','down'):
                old=target._residual_pairs.get((index,role))
                if old is not None:target._residual_pairs[index,role]=UnifiedResidual(old,role);changed.append(f'target.{index}.{role}')
        with engine.torch.cuda.stream(engine.model.stream),engine.torch.inference_mode():
            if graphs:target.prepare_graphs()
        target.unified_linear=True
        return dict(target.stats(),unified_linear=changed,role_schedules=dict(out='mtiled_bn64_s4',up='fullm_bn64_s6',down='mtiled_bn64_s5'),online_tuning=False)
    engine.prepare_slot_target=prepare_slot_target
