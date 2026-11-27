#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace sight {
// A bounded, exposure-compensated texture-change estimate. Affine luma
// differences are cheap to recognize and should not be warped as motion.
// Conservative thresholds deliberately choose flow for ambiguous inputs.
inline bool flow_stationary(const std::vector<uint8_t>&prev,const std::vector<uint8_t>&cur,
                            unsigned w,unsigned h)
{
    const size_t n=size_t(w)*h;
    if(!w||!h||w>1280||h>1280||prev.size()!=3*n||cur.size()!=3*n)return false;
    const unsigned gw=std::min(w,64u),gh=std::min(h,64u);
    const unsigned count=gw*gh;
    std::array<float,4096> a,b;
    double sa=0,sb=0,saa=0,sab=0;
    unsigned i=0;
    for(unsigned y=0;y<gh;++y)for(unsigned x=0;x<gw;++x){
        const size_t sy=uint64_t(2*y+1)*h/(2*gh),sx=uint64_t(2*x+1)*w/(2*gw);
        const auto j=(sy*w+sx)*3;
        a[i]=(prev[j]+prev[j+1]+prev[j+2])/3.f;
        b[i]=(cur[j]+cur[j+1]+cur[j+2])/3.f;
        sa+=a[i];sb+=b[i];saa+=a[i]*a[i];sab+=a[i]*b[i];++i;
    }
    const double variance=saa-sa*sa/count;
    const double slope=variance>1?std::clamp((sab-sa*sb/count)/variance,.5,1.5):1.;
    const double offset=(sb-slope*sa)/count;
    // Big global jumps are not considered ordinary lighting changes.
    if(std::abs(offset)>32)return false;
    double sum=0;unsigned changed=0;
    for(i=0;i<count;++i){
        const double err=std::abs(b[i]-(a[i]*slope+offset));
        sum+=err;changed+=err>6;
    }
    return sum/count<=1.25&&double(changed)/count<=.01;
}

// Sample a planar flow field or interleaved image using pixel coordinates.
template<class T>
inline float flow_sample(const T* p,unsigned w,unsigned h,float x,float y,
                         unsigned stride=1,unsigned channel=0)
{
    x=std::clamp(x,0.f,float(w-1));y=std::clamp(y,0.f,float(h-1));
    const unsigned x0=unsigned(x),y0=unsigned(y),x1=std::min(x0+1,w-1),y1=std::min(y0+1,h-1);
    const float dx=x-x0,dy=y-y0;
    auto at=[&](unsigned a,unsigned b){return float(p[(size_t(b)*w+a)*stride+channel]);};
    return (at(x0,y0)*(1-dx)+at(x1,y0)*dx)*(1-dy)+(at(x0,y1)*(1-dx)+at(x1,y1)*dx)*dy;
}

// Longpipe backward flow uses BASE resolution pixels, not grid or canvas
// pixels. This conservative, independently authored gate is not SDK parity.
// Fresh alpha is authoritative: reject out-of-bounds, divergent motion,
// incompatible RGB or large matte disagreement. Never propagate alone.
inline size_t warp_matte(std::vector<uint8_t>&alpha,const std::vector<uint8_t>&rgb,
    const std::vector<uint8_t>&prev,const std::vector<uint8_t>&prgb,
    unsigned w,unsigned h,const std::vector<float>&flow,unsigned fw,unsigned fh,
    unsigned bw,unsigned bh,float retention)
{
    const size_t n=size_t(w)*h,fn=size_t(fw)*fh;
    if(!w||!h||!fw||!fh||!bw||!bh||w>1280||h>1280||fw>320||fh>320||
       !(retention>0)||alpha.size()!=n||prev.size()!=n||rgb.size()!=3*n||
       prgb.size()!=3*n||flow.size()!=2*fn||
       !std::all_of(flow.begin(),flow.end(),[](float v){return std::isfinite(v);}))return 0;
    retention=std::min(retention,.9f);
    const float sx=float(w)/bw,sy=float(h)/bh;
    // Prepare bounded flow-grid gates once, not a divergence calculation per
    // high-resolution output pixel.
    std::vector<float> gate(fn);
    const auto*dx=flow.data();const auto*dy=dx+fn;
    for(unsigned y=0;y<fh;++y)for(unsigned x=0;x<fw;++x){
        const size_t i=size_t(y)*fw+x;
        unsigned l=x?x-1:x,r=std::min(x+1,fw-1),u=y?y-1:y,d=std::min(y+1,fh-1);
        const float divx=r==l?0:(dx[size_t(y)*fw+r]-dx[size_t(y)*fw+l])*float(fw)/bw/(r-l);
        const float divy=d==u?0:(dy[size_t(d)*fw+x]-dy[size_t(u)*fw+x])*float(fh)/bh/(d-u);
        const float magnitude=std::hypot(dx[i],dy[i]);
        const float divergence=std::abs(divx)+std::abs(divy);
        gate[i]=std::max(0.f,1-magnitude/12)*std::max(0.f,1-divergence);
    }
    size_t changed=0;
    for(unsigned y=0;y<h;++y)for(unsigned x=0;x<w;++x){
        const size_t i=size_t(y)*w+x;
        const float gx=(x+.5f)*fw/w-.5f,gy=(y+.5f)*fh/h-.5f;
        const float weight=retention*flow_sample(gate.data(),fw,fh,gx,gy);
        if(!(weight>0))continue;
        const float px=x+sx*flow_sample(dx,fw,fh,gx,gy),py=y+sy*flow_sample(dy,fw,fh,gx,gy);
        if(!(px>=0&&py>=0&&px<=w-1&&py<=h-1))continue;
        const float old=flow_sample(prev.data(),w,h,px,py);
        if(std::abs(old-alpha[i])>64)continue;
        const auto value=uint8_t(std::clamp(std::lround(alpha[i]*(1-weight)+old*weight),0L,255L));
        if(value==alpha[i])continue; // no need to sample three RGB planes
        float delta=0;
        for(unsigned c=0;c<3;++c)
            delta=std::max(delta,std::abs(rgb[3*i+c]-flow_sample(prgb.data(),w,h,px,py,3,c)));
        if(delta>24)continue;
        ++changed;alpha[i]=value;
    }
    return changed;
}
}
