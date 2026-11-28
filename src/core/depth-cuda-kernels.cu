#include <math.h>
struct DepthRange { float lo,hi; unsigned invalid; };
extern "C" __global__ void depth_pack(const unsigned char* rgb,float* nchw,unsigned n){
    unsigned p=blockIdx.x*blockDim.x+threadIdx.x;
    if(p>=n)return;
    const float mean[3]={.485f,.456f,.406f},dev[3]={.229f,.224f,.225f};
    for(unsigned c=0;c<3;++c)nchw[c*n+p]=(float(rgb[p*3+c])/255.f-mean[c])/dev[c];
}
__device__ void reduce_range(float* lo,float* hi,unsigned* bad,unsigned lane){
    for(unsigned stride=blockDim.x/2;stride;stride>>=1){
        __syncthreads();
        if(lane<stride){
            lo[lane]=fminf(lo[lane],lo[lane+stride]);
            hi[lane]=fmaxf(hi[lane],hi[lane+stride]);
            bad[lane]|=bad[lane+stride];
        }
    }
    __syncthreads();
}
extern "C" __global__ void depth_partial(const float* values,DepthRange* blocks,unsigned n){
    __shared__ float lo[256],hi[256];__shared__ unsigned bad[256];
    unsigned lane=threadIdx.x,p=blockIdx.x*blockDim.x+lane;
    float value=p<n?values[p]:0.f;
    bool valid=p<n&&isfinite(value);
    lo[lane]=valid?value:INFINITY;hi[lane]=valid?value:-INFINITY;
    bad[lane]=p<n&&!valid;
    reduce_range(lo,hi,bad,lane);
    if(!lane)blocks[blockIdx.x]={lo[0],hi[0],bad[0]};
}
extern "C" __global__ void depth_range(const DepthRange* blocks,DepthRange* total,unsigned count){
    __shared__ float lo[256],hi[256];__shared__ unsigned bad[256];
    unsigned lane=threadIdx.x;
    float mn=INFINITY,mx=-INFINITY;unsigned invalid=0;
    for(unsigned i=lane;i<count;i+=blockDim.x){
        mn=fminf(mn,blocks[i].lo);mx=fmaxf(mx,blocks[i].hi);invalid|=blocks[i].invalid;
    }
    lo[lane]=mn;hi[lane]=mx;bad[lane]=invalid;
    reduce_range(lo,hi,bad,lane);
    if(!lane)*total={lo[0],hi[0],bad[0]};
}
extern "C" __global__ void depth_unpack(const float* values,const DepthRange* range,unsigned char* out,unsigned n){
    unsigned p=blockIdx.x*blockDim.x+threadIdx.x;
    if(p>=n)return;
    float span=range->hi-range->lo;
    out[p]=range->invalid||span<1e-6f?0:
        static_cast<unsigned char>(fminf(fmaxf((values[p]-range->lo)*(255.f/span),0.f),255.f));
}
