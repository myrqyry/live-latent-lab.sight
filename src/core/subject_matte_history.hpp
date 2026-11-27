#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace sight {
// Common unwarped stabilization kernel; callers own history and validate
// geometry/age. Preserves the original motion and alpha disagreement gates.
inline size_t blend_stationary_matte(std::vector<uint8_t>&alpha,const std::vector<uint8_t>&rgb,
    const std::vector<uint8_t>&previous_alpha,const std::vector<uint8_t>&previous_rgb,float retention)
{
    const size_t count=alpha.size();
    if(!(retention>0)||previous_alpha.size()!=count||rgb.size()!=count*3||previous_rgb.size()!=count*3)return 0;
    retention=std::min(retention,.9f);
    size_t blended=0;
    for(size_t i=0;i<count;++i){
        int motion=0;
        for(size_t c=0;c<3;++c)
            motion=std::max(motion,std::abs(int(rgb[i*3+c])-int(previous_rgb[i*3+c])));
        if(motion<=24&&std::abs(int(alpha[i])-int(previous_alpha[i]))<=64){
            ++blended;
            alpha[i]=uint8_t(std::lround(alpha[i]*(1.f-retention)+previous_alpha[i]*retention));
        }
    }
    return blended;
}

// Worker-owned, bounded history at inference resolution. Never blends over
// large mask changes or changed source pixels; no graphics readback needed.
class SubjectMatteHistory {
    std::vector<uint8_t> previous_alpha_, previous_rgb_;
    uint32_t width_ = 0, height_ = 0;
public:
    void reset() {
        previous_alpha_.clear();
        previous_rgb_.clear();
        width_ = height_ = 0;
    }

    size_t apply(std::vector<uint8_t>& alpha, const std::vector<uint8_t>& rgb,
               uint32_t width, uint32_t height, float retention) {
        const size_t count = size_t(width) * height;
        if (!(retention > 0.f) || !count || alpha.size()!=count || rgb.size()!=count*3) {
            reset();
            return 0;
        }
        size_t blended=0;
        if(width_==width&&height_==height)
            blended=blend_stationary_matte(alpha,rgb,previous_alpha_,previous_rgb_,retention);
        width_=width; height_=height;
        previous_alpha_=alpha;
        previous_rgb_=rgb;
        return blended;
    }
};
}
