#pragma once
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include "sight_matte_v1.h"

namespace sight {
struct MotionSample { bool valid=false; float energy=0.f; };

// Inference-cadence mean absolute luma difference, not optical flow or
// time-normalized velocity. Reuse the existing worker RGB capture.
class SubjectMotion {
    std::vector<uint8_t> previous_;
    uint32_t w_=0,h_=0,sw_=0,sh_=0;
    std::chrono::steady_clock::time_point previous_time_{};
public:
    void reset() { previous_.clear(); w_=h_=sw_=sh_=0; }
    MotionSample update(const std::vector<uint8_t>& rgb, uint32_t w, uint32_t h,
                        uint32_t sw, uint32_t sh,
                        std::chrono::steady_clock::time_point captured) {
        if (!w || !h || !sw || !sh || w>SIGHT_MATTE_MAX_EDGE || h>SIGHT_MATTE_MAX_EDGE ||
            rgb.size()!=size_t(w)*h*3) {reset(); return {};}
        const bool comparable=w_==w && h_==h && sw_==sw && sh_==sh &&
            captured>previous_time_ &&
            captured-previous_time_<=std::chrono::seconds(1);
        const size_t count=size_t(w)*h;
        previous_.resize(count);
        uint64_t difference=0;
        for (size_t i=0;i<count;++i) {
            const auto* p=&rgb[i*3];
            // Integer BT.601 luma approximation (coefficients sum to 256).
            const uint8_t y=(77u*p[0]+150u*p[1]+29u*p[2]+128u)>>8;
            if (comparable) difference+=std::abs(int(y)-int(previous_[i]));
            previous_[i]=y;
        }
        w_=w; h_=h; sw_=sw; sh_=sh; previous_time_=captured;
        return {comparable,comparable?float(difference)/float(count*255u):0.f};
    }
};
}
