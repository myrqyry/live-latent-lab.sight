#include <math.h>
extern "C" __global__ void edge_pack(const unsigned char* bytes,float* chw,unsigned n,unsigned probability){
    unsigned p=blockIdx.x*blockDim.x+threadIdx.x;
    if(p>=n)return;
    for(unsigned c=0;c<3;++c){
        const float v=float(bytes[p*3+c]);
        chw[c*n+p]=probability?v/255.f:v;
    }
}
extern "C" __global__ void edge_combine(const float* a,const float* b,const float* c,const float* d,
    unsigned char* out,unsigned* invalid,unsigned n,unsigned steps,unsigned probability){
    unsigned p=blockIdx.x*blockDim.x+threadIdx.x;
    if(p>=n)return;
    if(probability){
        float value=a[p];
        if(!isfinite(value)){atomicOr(invalid,1u);out[p]=0;return;}
        float adjusted=steps?float(int(value*3.f))/2.f:value;
        out[p]=static_cast<unsigned char>(fminf(fmaxf(adjusted*255.f,0.f),255.f));
    }else{
        if(!isfinite(a[p])||!isfinite(b[p])||!isfinite(c[p])||!isfinite(d[p])){
            atomicOr(invalid,1u);out[p]=0;return;
        }
        float logits=((a[p]+b[p])+c[p]+d[p])*.25f;
        double sigmoid=1.0/(1.0+exp(-double(logits)));
        double value=steps?double(float(int(float(sigmoid)*float(steps+1)))/float(steps)):sigmoid;
        out[p]=static_cast<unsigned char>(fmin(fmax(value*255.0,0.0),255.0));
    }
}
