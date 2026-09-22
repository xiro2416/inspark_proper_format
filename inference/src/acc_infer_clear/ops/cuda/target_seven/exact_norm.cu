// Target-only FP32 N=1280 experiment. Welford arithmetic/reduction structure
// derived from PyTorch v2.8.0 aten/src/ATen/native/cuda/layer_norm_kernel.cu
// (PyTorch BSD-style license). No fast math; default NVCC FMA contraction.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAMathCompat.h>

struct WData {float mean,sigma2,count;};
__device__ WData online_sum(float val,const WData& s) {
  float delta=val-s.mean;
  float count=s.count+1.f;
  float mean=s.mean+delta*(1.f/count);
  return {mean,s.sigma2+delta*(val-mean),count};
}
// Parameter order intentionally follows cuWelfordCombine(dataB,dataA).
__device__ WData combine(WData b,WData a) {
  float delta=b.mean-a.mean;
  float count=a.count+b.count;
  float mean,sigma2;
  if(count>0.f) {
    auto coef=1.f/count;
    auto na=a.count*coef;
    auto nb=b.count*coef;
    mean=na*a.mean+nb*b.mean;
    sigma2=a.sigma2+b.sigma2+delta*delta*a.count*nb;
  } else {mean=0.f;sigma2=0.f;}
  return {mean,sigma2,count};
}
struct alignas(16) Vec {float val[4];};

template<bool CACHE_X,bool PREFETCH_AFFINE,bool UNROLL_CACHE=false>
__global__ void exact_norm_kernel(int n,float eps,const float* __restrict__ x,
 const float* __restrict__ gamma,const float* __restrict__ beta,float* y) {
  extern __shared__ float buf[];
  int row=blockIdx.x;
  int numx=blockDim.x*blockDim.y;
  int tid=threadIdx.x+threadIdx.y*blockDim.x;
  int nv=n/4;
  auto xv=reinterpret_cast<const Vec*>(x+row*n);
  auto gv=reinterpret_cast<const Vec*>(gamma);
  auto bv=reinterpret_cast<const Vec*>(beta);
  auto yv=reinterpret_cast<Vec*>(y+row*n);
  Vec cached_x[3],cached_g[3],cached_b[3];
  WData wd{0.f,0.f,0.f};
  int k=0;
  if constexpr(UNROLL_CACHE) {
    // Same per-thread input order as upstream. Compile-time array subscripts
    // allow scalar register allocation instead of dynamic local arrays.
    #pragma unroll
    for(int t=0;t<3;++t) {
      int i=tid+t*128;
      if(i<320) {
        Vec data=xv[i];cached_x[t]=data;
        if(PREFETCH_AFFINE){cached_g[t]=gv[i];cached_b[t]=bv[i];}
        #pragma unroll
        for(int j=0;j<4;++j)wd=online_sum(data.val[j],wd);
      }
    }
  } else {
    for(int i=tid;i<nv;i+=numx,++k) {
      Vec data=xv[i];
      if(CACHE_X)cached_x[k]=data;
      if(PREFETCH_AFFINE){cached_g[k]=gv[i];cached_b[k]=bv[i];}
      #pragma unroll
      for(int j=0;j<4;++j)wd=online_sum(data.val[j],wd);
    }
  }
  for(int offset=16;offset>0;offset>>=1) {
    WData b{__shfl_down_sync(0xffffffff,wd.mean,offset),
      __shfl_down_sync(0xffffffff,wd.sigma2,offset),
      __shfl_down_sync(0xffffffff,wd.count,offset)};
    wd=combine(wd,b);
  }
  float* ms=buf;float* counts=buf+blockDim.y;
  for(int offset=blockDim.y/2;offset>0;offset/=2) {
    if(threadIdx.x==0&&threadIdx.y>=offset&&threadIdx.y<2*offset) {
      int w=threadIdx.y-offset;ms[2*w]=wd.mean;ms[2*w+1]=wd.sigma2;counts[w]=wd.count;
    }
    __syncthreads();
    if(threadIdx.x==0&&threadIdx.y<offset) {
      WData b{ms[2*threadIdx.y],ms[2*threadIdx.y+1],counts[threadIdx.y]};wd=combine(wd,b);
    }
    __syncthreads();
  }
  if(threadIdx.x==0&&threadIdx.y==0){ms[0]=wd.mean;ms[1]=wd.sigma2/float(n);}
  __syncthreads();
  float mean=ms[0];float rstd=c10::cuda::compat::rsqrt(ms[1]+eps);
  k=0;
  if constexpr(UNROLL_CACHE) {
    #pragma unroll
    for(int t=0;t<3;++t) {
      int i=tid+t*128;
      if(i<320) {
        Vec data=cached_x[t];
        Vec g=PREFETCH_AFFINE?cached_g[t]:gv[i];
        Vec b=PREFETCH_AFFINE?cached_b[t]:bv[i];Vec out;
        #pragma unroll
        for(int j=0;j<4;++j)out.val[j]=g.val[j]*(rstd*(data.val[j]-mean))+b.val[j];
        yv[i]=out;
      }
    }
  } else {
    for(int i=tid;i<nv;i+=numx,++k) {
      Vec data=CACHE_X?cached_x[k]:xv[i];
      Vec g=PREFETCH_AFFINE?cached_g[k]:gv[i];
      Vec b=PREFETCH_AFFINE?cached_b[k]:bv[i];Vec out;
      #pragma unroll
      for(int j=0;j<4;++j)out.val[j]=g.val[j]*(rstd*(data.val[j]-mean))+b.val[j];
      yv[i]=out;
    }
  }
}

torch::Tensor exact_norm(torch::Tensor x,torch::Tensor w,torch::Tensor b,double eps,int mode) {
  TORCH_CHECK(x.is_cuda()&&w.is_cuda()&&b.is_cuda(),"CUDA only");
  TORCH_CHECK(x.scalar_type()==torch::kFloat32&&w.scalar_type()==torch::kFloat32&&b.scalar_type()==torch::kFloat32,"FP32 only");
  TORCH_CHECK(x.is_contiguous()&&w.is_contiguous()&&b.is_contiguous(),"contiguous only");
  TORCH_CHECK(x.size(-1)==1280&&w.numel()==1280&&b.numel()==1280,"N1280 only");
  TORCH_CHECK(x.device()==w.device()&&x.device()==b.device(),"same device required");
  TORCH_CHECK(mode>=0&&mode<=4,"mode0=clone,1=cacheX,2=cacheX+prefetchAffine,3=unrolledX,4=unrolledX+affine");
  c10::cuda::CUDAGuard guard(x.device());auto y=torch::empty_like(x);
  dim3 grid(x.numel()/1280),block(32,4);auto stream=at::cuda::getCurrentCUDAStream();
  if(mode==0)exact_norm_kernel<false,false><<<grid,block,24,stream>>>(1280,float(eps),x.data_ptr<float>(),w.data_ptr<float>(),b.data_ptr<float>(),y.data_ptr<float>());
  if(mode==1)exact_norm_kernel<true,false><<<grid,block,24,stream>>>(1280,float(eps),x.data_ptr<float>(),w.data_ptr<float>(),b.data_ptr<float>(),y.data_ptr<float>());
  if(mode==2)exact_norm_kernel<true,true><<<grid,block,24,stream>>>(1280,float(eps),x.data_ptr<float>(),w.data_ptr<float>(),b.data_ptr<float>(),y.data_ptr<float>());
  if(mode==3)exact_norm_kernel<true,false,true><<<grid,block,24,stream>>>(1280,float(eps),x.data_ptr<float>(),w.data_ptr<float>(),b.data_ptr<float>(),y.data_ptr<float>());
  if(mode==4)exact_norm_kernel<true,true,true><<<grid,block,24,stream>>>(1280,float(eps),x.data_ptr<float>(),w.data_ptr<float>(),b.data_ptr<float>(),y.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();return y;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("forward",&exact_norm);}
