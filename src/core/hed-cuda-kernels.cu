#include <math.h>
extern "C" __global__ void hed_pack(const unsigned char* rgb,float* chw,unsigned n){
    unsigned p=blockIdx.x*blockDim.x+threadIdx.x;
    if(p>=n)return;
    for(unsigned c=0;c<3;++c)chw[c*n+p]=float(rgb[p*3+c]);
}
__device__ float sample(const float* map,unsigned x,unsigned y,unsigned w,unsigned h,unsigned scale){
    unsigned iw=w>>scale,ih=h>>scale;
    // Same half-pixel coordinate convention as OpenCV INTER_LINEAR.
    float sx=float((double(x)+0.5)/double(1u<<scale)-0.5);
    float sy=float((double(y)+0.5)/double(1u<<scale)-0.5);
    int ix=int(floorf(sx)),iy=int(floorf(sy));
    float ax=sx-ix,ay=sy-iy;
    if(ix<0){ix=0;ax=0;}if(ix>=int(iw)-1){ix=iw-1;ax=0;}
    if(iy<0){iy=0;ay=0;}if(iy>=int(ih)-1){iy=ih-1;ay=0;}
    unsigned x1=min(unsigned(ix+1),iw-1),y1=min(unsigned(iy+1),ih-1);
    float a=map[iy*iw+ix]*(1.f-ax)+map[iy*iw+x1]*ax;
    float b=map[y1*iw+ix]*(1.f-ax)+map[y1*iw+x1]*ax;
    return a*(1.f-ay)+b*ay;
}
extern "C" __global__ void hed_combine(const float* s1,const float* s2,const float* s3,const float* s4,const float* s5,
    unsigned char* out,unsigned* invalid,unsigned w,unsigned h,unsigned safe){
    unsigned p=blockIdx.x*blockDim.x+threadIdx.x;
    if(p>=w*h)return;
    unsigned x=p%w,y=p/w;
    const float v[]={s1[p],sample(s2,x,y,w,h,1),sample(s3,x,y,w,h,2),sample(s4,x,y,w,h,3),sample(s5,x,y,w,h,4)};
    float sum=0;
    for(unsigned i=0;i<5;++i){
        if(!isfinite(v[i])){atomicOr(invalid,1u);out[p]=0;return;}
        sum+=v[i];
    }
    float average=sum/5.f;
    double sigmoid=1.0/(1.0+exp(-double(average)));
    double value=safe?double(float(int(float(sigmoid)*3.f))/2.f):sigmoid;
    out[p]=static_cast<unsigned char>(fmin(fmax(value*255.0,0.0),255.0));
}
