import triton
import triton.language as tl
@triton.jit
def fused(A,W,Bias,R,Y,Q,M,N:tl.constexpr,K:tl.constexpr,Alpha:tl.constexpr,Inv:tl.constexpr,BM,BN:tl.constexpr,BK:tl.constexpr):
 pid=tl.program_id(0);nr=tl.cdiv(M,BM);nc=tl.cdiv(N,BN);group=pid//(8*nc);first=group*8;size=tl.minimum(nr-first,8);pm=first+(pid%(8*nc))%size;pn=(pid%(8*nc))//size
 rows=pm*BM+tl.arange(0,BM);cols=pn*BN+tl.arange(0,BN);ks=tl.arange(0,BK);acc=tl.zeros((BM,BN),tl.float32)
 for off in range(tl.cdiv(K,BK)):
  kk=off*BK+ks;x=tl.load(A+rows[:,None]*K+kk[None,:],(rows[:,None]<M)&(kk[None,:]<K),0.0);w=tl.load(W+cols[None,:]*K+kk[:,None],(cols[None,:]<N)&(kk[:,None]<K),0.0);acc=tl.dot(x,w,acc,max_num_imprecise_acc=0)
 z=acc*Alpha+tl.load(Bias+cols,cols<N,0)[None,:];z=z+tl.load(R+rows[:,None]*N+cols[None,:],(rows[:,None]<M)&(cols[None,:]<N),0)
 q=tl.maximum(tl.minimum(z*Inv,448.),-448.).to(tl.float8e4nv,fp_downcast_rounding="rtne")
 tl.store(Y+rows[:,None]*N+cols[None,:],z,(rows[:,None]<M)&(cols[None,:]<N));tl.store(Q+rows[:,None]*N+cols[None,:],q,(rows[:,None]<M)&(cols[None,:]<N))

@triton.jit
def residual_aot(A,W,Bias,R,M,Y,Q,N:tl.constexpr,K:tl.constexpr,Alpha:tl.constexpr,Inv:tl.constexpr):
 fused(A,W,Bias,R,Y,Q,M,N,K,Alpha,Inv,64,64,64)
