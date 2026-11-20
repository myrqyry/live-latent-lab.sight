#pragma once
// Optional, owned copy of MediaPipe Holistic's pose segmentation mask.
// This is NOT part of TrackingSignalBus; consumer snapshots stay raw.
#include "tasks_tracker.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace sight {
struct HolisticPersonMask {
    uint32_t width=0,height=0;
    std::vector<float> values;

    bool valid() const {
        return width>=8 && height>=8 && width<=4096 && height<=4096 &&
               values.size()==size_t(width)*height;
    }
    // Conservative neighborhood test. Segmentation sometimes undercovers
    // wrists/hair: allow a generous ~6-9% image-edge buffer.
    bool foreground_near(float x,float y,float radius=.075f) const {
        if(!valid()||!std::isfinite(x)||!std::isfinite(y))return false;
        radius=std::clamp(radius,0.f,.15f);
        for(float dy:{-1.f,0.f,1.f})for(float dx:{-1.f,0.f,1.f}) {
            const float sx=x+dx*radius, sy=y+dy*radius;
            if(sx<0.f||sx>1.f||sy<0.f||sy>1.f)continue;
            const auto ix=std::min(width-1,uint32_t(sx*float(width-1)));
            const auto iy=std::min(height-1,uint32_t(sy*float(height-1)));
            const float value=values[size_t(iy)*width+ix];
            if(std::isfinite(value)&&value>=.30f)return true;
        }
        return false;
    }
    bool has_credible_person(const TrackingData& body) const {
        if(!valid()||body.landmarks.size()!=33)return false;
        // A blank, entirely foreground, or corrupted mask should never
        // cause missing limbs. A coarse grid bounds work per frame.
        int hits=0,total=0;
        for(uint32_t y=0;y<height;y+=std::max(1u,height/16))
            for(uint32_t x=0;x<width;x+=std::max(1u,width/16)) {
                const float v=values[size_t(y)*width+x];
                if(!std::isfinite(v)||v<0.f||v>1.f)return false;
                hits+=v>.50f; ++total;
            }
        if(!total || hits<total/100 || hits>total*9/10)return false;
        int torso=0;
        for(int i:{11,12,23,24}) {
            const auto& p=body.landmarks[size_t(i)];
            if(p.confidence>.45f && foreground_near(p.x,p.y,.045f))
                ++torso;
        }
        return torso>=2;
    }
};
} // namespace sight
