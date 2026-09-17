// Welford expressions/order adapted from PyTorch v2.8.0 layer_norm_kernel.cu
// and the validated local exact_norm mode4. PyTorch BSD-style license applies.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAMathCompat.h>
struct WData {float mean,sigma2,count;};
struct alignas(16) Vec {float val[4];};
__device__ WData online_sum(float val,const WData& s){
 float delta=val-s.mean,count=s.count+1.f;
 float mean=s.mean+delta*(1.f/count);
 return {mean,s.sigma2+delta*(val-mean),count};
}
__device__ WData combine(WData b,WData a){
 float delta=b.mean-a.mean,count=a.count+b.count,mean,sigma2;
 if(count>0.f){auto coef=1.f/count;auto na=a.count*coef;auto nb=b.count*coef;
 mean=na*a.mean+nb*b.mean;sigma2=a.sigma2+b.sigma2+delta*delta*a.count*nb;
 }else{mean=0.f;sigma2=0.f;}return {mean,sigma2,count};
}
__device__ __forceinline__ float full_div(float x,float y){
 float r;asm("div.full.f32 %0, %1, %2;":"=f"(r):"f"(x),"f"(y));return r;
}
__device__ __forceinline__ float approx_rcp(float x){
 float r;asm("rcp.approx.ftz.f32 %0, %1;":"=f"(r):"f"(x));return r;
}
__device__ __forceinline__ unsigned short pack_fp8(float lo,float hi){
 unsigned short r;asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;":"=h"(r):"f"(hi),"f"(lo));return r;
}
template<bool EMIT_Y,int DIVISION,int SYNC_STRATEGY>
__global__ void norm_quant(int n,float eps,const float* __restrict__ x,
 const float* __restrict__ gamma,const float* __restrict__ beta,float* y,unsigned char* q,float* scales){
 __shared__ float buf[6];
 __shared__ float amax_shared[4];
 int row=blockIdx.x,tid=threadIdx.x+threadIdx.y*blockDim.x;
 auto xv=reinterpret_cast<const Vec*>(x+row*n);
 auto gv=reinterpret_cast<const Vec*>(gamma);auto bv=reinterpret_cast<const Vec*>(beta);
 Vec cx[3],cg[3],cb[3],out[3];WData wd{0.f,0.f,0.f};
 #pragma unroll
 for(int t=0;t<3;++t){int i=tid+t*128;if(i<320){Vec data=xv[i];cx[t]=data;cg[t]=gv[i];cb[t]=bv[i];
 #pragma unroll
 for(int j=0;j<4;++j)wd=online_sum(data.val[j],wd);}}
 for(int offset=16;offset>0;offset>>=1){
 WData b{__shfl_down_sync(0xffffffff,wd.mean,offset),__shfl_down_sync(0xffffffff,wd.sigma2,offset),__shfl_down_sync(0xffffffff,wd.count,offset)};wd=combine(wd,b);}
 float* ms=buf;float* counts=buf+blockDim.y;
 for(int offset=blockDim.y/2;offset>0;offset/=2){
 if(threadIdx.x==0&&threadIdx.y>=offset&&threadIdx.y<2*offset){int w=threadIdx.y-offset;ms[2*w]=wd.mean;ms[2*w+1]=wd.sigma2;counts[w]=wd.count;}
 __syncthreads();
 if(threadIdx.x==0&&threadIdx.y<offset){WData b{ms[2*threadIdx.y],ms[2*threadIdx.y+1],counts[threadIdx.y]};wd=combine(wd,b);}
 __syncthreads();}
 if(threadIdx.x==0&&threadIdx.y==0){ms[0]=wd.mean;ms[1]=wd.sigma2/float(n);}
 __syncthreads();
 float mean=ms[0];float rstd=c10::cuda::compat::rsqrt(ms[1]+eps),amax=0.f;
 #pragma unroll
 for(int t=0;t<3;++t){int i=tid+t*128;if(i<320){
 #pragma unroll
 for(int j=0;j<4;++j){out[t].val[j]=cg[t].val[j]*(rstd*(cx[t].val[j]-mean))+cb[t].val[j];amax=fmaxf(amax,fabsf(out[t].val[j]));}
 if(EMIT_Y)reinterpret_cast<Vec*>(y+row*n)[i]=out[t];}}
 float scale;
 if constexpr(SYNC_STRATEGY==0){
   // Control: protect stats readers, then publish amax and broadcast scale.
   __syncthreads();
   for(int offset=16;offset>0;offset>>=1)amax=fmaxf(amax,__shfl_down_sync(0xffffffff,amax,offset));
   if(threadIdx.x==0)buf[threadIdx.y]=amax;
   __syncthreads();
   if(tid==0){float a=fmaxf(fmaxf(buf[0],buf[1]),fmaxf(buf[2],buf[3]));buf[4]=fmaxf(full_div(a,448.f),1e-12f);scales[row]=buf[4];}
   __syncthreads();scale=buf[4];
 }else{
   // Separate storage cannot race with remaining Welford-buffer readers.
   // One barrier publishes four warp maxima; each lane computes its own scale.
   for(int offset=16;offset>0;offset>>=1)amax=fmaxf(amax,__shfl_down_sync(0xffffffff,amax,offset));
   if(threadIdx.x==0)amax_shared[threadIdx.y]=amax;
   __syncthreads();
   float a=fmaxf(fmaxf(amax_shared[0],amax_shared[1]),fmaxf(amax_shared[2],amax_shared[3]));
   scale=fmaxf(full_div(a,448.f),1e-12f);
   if(tid==0)scales[row]=scale;
 }
 float inv=0.f;
 if(DIVISION==1)inv=approx_rcp(scale);
 #pragma unroll
 for(int t=0;t<3;++t){int i=tid+t*128;if(i<320){float z[4];
 #pragma unroll
 for(int j=0;j<4;++j)z[j]=DIVISION==0?full_div(out[t].val[j],scale):__fmul_rn(out[t].val[j],inv);
 auto qp=reinterpret_cast<unsigned short*>(q+row*n+i*4);qp[0]=pack_fp8(z[0],z[1]);qp[1]=pack_fp8(z[2],z[3]);}}
}
std::vector<torch::Tensor> fused(torch::Tensor x,torch::Tensor w,torch::Tensor b,double eps,bool emit_y,int division,int sync_strategy){
 TORCH_CHECK(x.is_cuda()&&w.is_cuda()&&b.is_cuda(),"CUDA only");
 TORCH_CHECK(x.scalar_type()==torch::kFloat32&&w.scalar_type()==torch::kFloat32&&b.scalar_type()==torch::kFloat32,"FP32 only");
 TORCH_CHECK(x.dim()>0&&x.size(-1)==1280&&w.numel()==1280&&b.numel()==1280,"N1280 only");
 TORCH_CHECK(x.is_contiguous()&&w.is_contiguous()&&b.is_contiguous(),"contiguous only");
 TORCH_CHECK(x.device()==w.device()&&x.device()==b.device(),"same GPU required");
 TORCH_CHECK(x.numel()>0&&division>=0&&division<=1,"invalid input/mode");
 TORCH_CHECK(sync_strategy>=0&&sync_strategy<=1,"invalid synchronization strategy");
 c10::cuda::CUDAGuard guard(x.device());int m=x.numel()/1280;
 auto q=torch::empty({m,1280},x.options().dtype(torch::kUInt8));auto s=torch::empty({m},x.options());
 auto y=emit_y?torch::empty_like(x):torch::empty({0},x.options());
 auto stream=at::cuda::getCurrentCUDAStream();dim3 grid(m),block(32,4);
 #define LAUNCH(E,D,S) norm_quant<E,D,S><<<grid,block,0,stream>>>(1280,float(eps),x.data_ptr<float>(),w.data_ptr<float>(),b.data_ptr<float>(),y.data_ptr<float>(),q.data_ptr<unsigned char>(),s.data_ptr<float>())
 #define DISPATCH(S) if(emit_y){if(division==0){LAUNCH(true,0,S);}else{LAUNCH(true,1,S);}}else{if(division==0){LAUNCH(false,0,S);}else{LAUNCH(false,1,S);}}
 if(sync_strategy==0){DISPATCH(0);}else{DISPATCH(1);}
 #undef DISPATCH
 #undef LAUNCH
 C10_CUDA_KERNEL_LAUNCH_CHECK();return {q,s,y};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("forward",&fused);}
