"""CFM-local elementwise/reduction fusions; unchanged high-precision interfaces."""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

@triton.jit(do_not_specialize=['T'])
def _rms_mod(X,W,P,Y,T,D:tl.constexpr,EPS:tl.constexpr,HAS_COND:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0);i=tl.arange(0,BLOCK);x=tl.load(X+row*D+i,i<D,0).to(tl.float32)
    # Match PyTorch's float mean vectorized-input reduction for D512, >=16 rows:
    # four separate accumulators per lane, then sequential combine + warp reduction.
    lane=tl.arange(0,32);a0=tl.full((32,),0,tl.float32);a1=a0;a2=a0;a3=a0
    for chunk in tl.static_range(D//128):
        base=X+row*D+chunk*128+lane*4
        v0=tl.load(base);v1=tl.load(base+1);v2=tl.load(base+2);v3=tl.load(base+3)
        a0+=v0*v0;a1+=v1*v1;a2+=v2*v2;a3+=v3*v3
    partial=((a0+a1)+a2)+a3
    for shift in tl.static_range(5):
        other=tl.inline_asm_elementwise('shfl.sync.bfly.b32 $0, $1, $2, 31, -1;',
            constraints='=r,r,r',args=[partial.to(tl.int32,bitcast=True),1<<shift],dtype=tl.int32,is_pure=True,pack=1).to(tl.float32,bitcast=True)
        partial=partial+other
    mean=tl.sum(tl.where(lane==0,partial,0.),0)/D;y=x*tl.rsqrt(mean+EPS);y=y*tl.load(W+i,i<D,0)
    if HAS_COND:
        b=row//T;gain=tl.load(P+b*2*D+i,i<D,0);bias=tl.load(P+b*2*D+D+i,i<D,0);y=gain*y+bias
    tl.store(Y+row*D+i,y,i<D)

@triton.jit(do_not_specialize=['COUNT'])
def _silu_mul(A,B,Y,COUNT,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);a=tl.load(A+i,i<COUNT,0);b=tl.load(B+i,i<COUNT,0)
    y=tl.div_rn(a,1.+libdevice.exp(-a))*b;tl.store(Y+i,y,i<COUNT)

@triton.jit(do_not_specialize=['T','TOTAL'])
def _rope_qkv(X,F,QKV,T,TOTAL,H:tl.constexpr,D:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);pair=i%(D//2);t=(i//(D//2))%T;h=(i//(D//2)//T)%H;b=i//(D//2)//T//H
    src=(b*T+t)*(3*H*D)+h*D+pair*2;valid=i<TOTAL
    co=tl.load(F+t*D+pair*2,valid,0);si=tl.load(F+t*D+pair*2+1,valid,0)
    q0=tl.load(X+src,valid,0);q1=tl.load(X+src+1,valid,0)
    k0=tl.load(X+src+H*D,valid,0);k1=tl.load(X+src+H*D+1,valid,0)
    v0=tl.load(X+src+2*H*D,valid,0);v1=tl.load(X+src+2*H*D+1,valid,0)
    tl.store(QKV+i*2,q0*co-q1*si,valid);tl.store(QKV+i*2+1,q1*co+q0*si,valid)
    tl.store(QKV+TOTAL*2+i*2,k0*co-k1*si,valid);tl.store(QKV+TOTAL*2+i*2+1,k1*co+k0*si,valid)
    tl.store(QKV+TOTAL*4+i*2,v0,valid);tl.store(QKV+TOTAL*4+i*2+1,v1,valid)

class AdaptiveRMS(torch.nn.Module):
    def __init__(self,old):
        super().__init__();self.project_layer=old.project_layer;self.norm=old.norm;self.eps=old.eps
    def forward(self,x,embedding=None):
        if x.dtype!=torch.float32 or x.ndim!=3:raise ValueError('Validated FP32 B,T,D interface only')
        x=x.contiguous();b,t,d=x.shape
        if d!=512 or b*t<16:
            if embedding is None:return self.norm(x)
            gain,bias=self.project_layer(embedding).chunk(2,dim=-1)
            return gain*self.norm(x)+bias
        p=self.project_layer(embedding).contiguous() if embedding is not None else self.norm.weight
        y=torch.empty_like(x)
        _rms_mod[(b*t,)](x,self.norm.weight,p,y,t,d,self.eps,embedding is not None,triton.next_power_of_2(d),enable_fp_fusion=False)
        return y

class GatedFFN(torch.nn.Module):
    def __init__(self,old):super().__init__();self.w1=old.w1;self.w3=old.w3;self.w2=old.w2
    def forward(self,x):
        a,b=self.w1(x),self.w3(x);y=torch.empty_like(a);n=a.numel()
        _silu_mul[(triton.cdiv(n,512),)](a,b,y,n,512,enable_fp_fusion=False)
        return self.w2(y)

class ProjectionRope(torch.nn.Module):
    def __init__(self,projection,heads,dim):super().__init__();self.projection=projection;self.heads=heads;self.dim=dim
    def forward(self,x,freqs):
        y=self.projection(x).contiguous();b,t,_=y.shape;h,d=self.heads,self.dim
        if y.dtype!=torch.float32:raise ValueError('FP32 QKV expected')
        f=freqs.float().contiguous();out=torch.empty((3,b,h,t,d),device=y.device,dtype=y.dtype);n=b*h*t*(d//2)
        _rope_qkv[(triton.cdiv(n,256),)](y,f,out,t,n,h,d,256,enable_fp_fusion=False)
        return out[0],out[1],out[2]

def install(model,parts=('norm','gate','rope')):
    requested=set(parts)
    if not requested or not requested<= {'norm','gate','rope'}:
        raise ValueError('parts must be a non-empty subset of norm/gate/rope')
    changed=[];installed=set()
    def walk(module,prefix):
        for name,child in list(module.named_children()):
            path=prefix+'.'+name
            if 'norm' in parts and type(child).__name__=='AdaptiveLayerNorm' and type(child.norm).__name__=='RMSNorm':
                module.add_module(name,AdaptiveRMS(child));changed.append(path);installed.add('norm')
            elif 'gate' in parts and type(child).__name__=='FeedForward':module.add_module(name,GatedFFN(child));changed.append(path);installed.add('gate')
            elif 'rope' in parts and type(child).__name__=='Attention' and hasattr(child,'wqkv') and child.n_head==child.n_local_heads:
                if getattr(child,'_acc_qkv_rope',None) is not None:raise RuntimeError('Existing QKV hook must not be overwritten')
                child._acc_qkv_rope=ProjectionRope(child.wqkv,child.n_head,child.head_dim);changed.append(path+'._acc_qkv_rope');installed.add('rope')
            else:walk(child,path)
    walk(model,'cfm')
    missing=requested-installed
    if missing:raise RuntimeError('Requested CFM fusion parts were not installed: '+','.join(sorted(missing)))
    return changed
