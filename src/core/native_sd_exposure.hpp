#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
namespace sight::sd {
struct ExposureResult {
    bool applied=false;
    float before_error=0,after_error=0;
};
class ExposureAnchor {
    bool valid_=false;
    double mean_=0;
    static double mean(const std::vector<uint8_t>&p,unsigned channels){
        uint64_t sum=0;
        for(size_t i=0;i<p.size();i+=channels)
            sum+=54u*p[i]+183u*p[i+1]+19u*p[i+2];
        return double(sum)/(256.*double(p.size()/channels));
    }
public:
    void reset(){valid_=false;mean_=0;}
    ExposureResult apply(std::vector<uint8_t>&p,unsigned channels,bool new_anchor,float strength){
        if((channels!=3&&channels!=4)||p.empty()||p.size()%channels||!std::isfinite(strength))return {};
        if(new_anchor)reset();
        if(!valid_){mean_=mean(p,channels);valid_=true;return {};}
        if(strength<=0)return {};
        const double current=mean(p,channels);
        const int shift=int(std::lround(std::clamp(mean_-current,-16.,16.)*std::clamp(strength,0.f,1.f)));
        ExposureResult r{false,float(std::abs(mean_-current)),float(std::abs(mean_-current))};
        if(!shift)return r;
        for(size_t i=0;i<p.size();i+=channels){
            // Common channel offset, constrained by the pixel's gamut:
            // preserve RGB channel differences and alpha exactly.
            const int lo=std::min({p[i],p[i+1],p[i+2]}),hi=std::max({p[i],p[i+1],p[i+2]});
            const int safe=std::clamp(shift,-lo,255-hi);
            for(unsigned c=0;c<3;++c)p[i+c]=uint8_t(int(p[i+c])+safe);
            r.applied|=safe!=0;
        }
        r.after_error=float(std::abs(mean_-mean(p,channels)));
        return r;
    }
};
}
