// Native SM75 inference pointwise operations; FP32, no fast math or FMA.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

__global__ void unary(const float* x,float* y,int64_t n,int op) {
    int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=n) return;
    float v=x[i];
    if(isnan(v)) { y[i]=v; return; }
    if(op==0) {
        float b=fminf(fmaxf(v,-4.f),4.f);
        float inner=-0.055908203125f*fabsf(b)+0.447265625f;
        y[i]=v*(b*inner+0.89453125f);
    } else {
        bool global=op==2;
        float a=global?0.08953857421875f:0.044921875f;
        float b=global?1.708984375f:1.30078125f;
        float lo=global?1.439453125f:1.03125f;
        float hi=global?1.9775390625f:1.5693359375f;
        float shift=global?11.090354888959125f:22.18070977791825f;
        float z=global?0.083984375f:0.025390625f;
        y[i]=z*expf((fminf(fmaxf(v*a+b,lo),hi)-b)*shift);
    }
}
__global__ void residual(const float* x,const float* skip,const float* s,float* y,int64_t n,int64_t channels) {
    int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<n) y[i]=x[i]+skip[i]*s[i%channels];
}
extern "C" int nr_unary(const float* x,float* y,int64_t n,int op,cudaStream_t stream) {
    if(n>0) unary<<<(n+255)/256,256,0,stream>>>(x,y,n,op);
    return cudaGetLastError();
}
extern "C" int nr_residual(const float* x,const float* skip,const float* s,float* y,int64_t n,int64_t channels,cudaStream_t stream) {
    if(n>0) residual<<<(n+255)/256,256,0,stream>>>(x,skip,s,y,n,channels);
    return cudaGetLastError();
}
