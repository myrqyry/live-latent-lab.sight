#include <math.h>
extern "C" __global__ void cutout_pack(const unsigned char* rgb,float* chw,
    unsigned w,unsigned h,unsigned iw,unsigned ih,unsigned mode){
    unsigned p=blockIdx.x*blockDim.x+threadIdx.x;
    if(p>=iw*ih)return;
    unsigned x=p%iw,y=p/iw;
    float sx=(float(w-1)/float(iw-1))*x,sy=(float(h-1)/float(ih-1))*y;
    unsigned xl=unsigned(sx),yl=unsigned(sy),xh=min(xl+1,w-1),yh=min(yl+1,h-1);
    float ax=sx-xl,ay=sy-yl;
    const float mean[]={.485f,.456f,.406f},dev[]={.229f,.224f,.225f};
    for(unsigned c=0;c<3;++c){
        float v=rgb[(yl*w+xl)*3+c]*(1-ax)*(1-ay)+rgb[(yl*w+xh)*3+c]*ax*(1-ay)+
            rgb[(yh*w+xl)*3+c]*(1-ax)*ay+rgb[(yh*w+xh)*3+c]*ax*ay;
        float f=float(static_cast<unsigned char>(v))/255.f;
        chw[c*iw*ih+p]=mode==2?f:mode==1?(f-.5f)/.5f:(f-mean[c])/dev[c];
    }
}
extern "C" __global__ void cutout_validate(const float* alpha,unsigned n,unsigned* invalid){
    unsigned p=blockIdx.x*blockDim.x+threadIdx.x;
    if(p<n&&!isfinite(alpha[p]))atomicOr(invalid,1u);
}
extern "C" __global__ void cutout_unpack(const float* alpha,unsigned char* mask,
    unsigned w,unsigned h,unsigned iw,unsigned ih){
    unsigned p=blockIdx.x*blockDim.x+threadIdx.x;
    if(p>=w*h)return;
    unsigned x=p%w,y=p/w;
    float sx=x*(w>1?float(iw-1)/float(w-1):0.f),sy=y*(h>1?float(ih-1)/float(h-1):0.f);
    unsigned xl=unsigned(sx),yl=unsigned(sy),xh=min(xl+1,iw-1),yh=min(yl+1,ih-1);
    float ax=sx-xl,ay=sy-yl;
    float v=alpha[yl*iw+xl]*(1-ax)*(1-ay)+alpha[yl*iw+xh]*ax*(1-ay)+
        alpha[yh*iw+xl]*(1-ax)*ay+alpha[yh*iw+xh]*ax*ay;
    mask[p]=static_cast<unsigned char>(fminf(fmaxf(v,0.f),1.f)*255.f);
}
