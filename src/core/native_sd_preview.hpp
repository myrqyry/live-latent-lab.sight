#pragma once
#include "common/image-utils.hpp"
#include <cstdint>
#include <cmath>
#include <algorithm>

namespace sight::sd {
struct PreviewKey {
    uint64_t epoch=0,frame=0;
    uint32_t width=0,height=0;
    uint64_t completed_ns=0;
    bool recursive=false;
    TransformParams transform{};
};
inline bool can_interpolate(const PreviewKey&a,const PreviewKey&b){
    return a.width&&a.height&&a.width==b.width&&a.height==b.height&&a.epoch==b.epoch&&
        b.recursive&&b.frame>a.frame&&b.frame-a.frame==1&&b.completed_ns>a.completed_ns;
}
inline uint64_t preview_duration_ns(const PreviewKey&a,const PreviewKey&b,unsigned max_ms){
    if(!can_interpolate(a,b)||!max_ms)return 0;
    return std::min(b.completed_ns-a.completed_ns,uint64_t(max_ms)*uint64_t(1000000));
}
inline float preview_phase(uint64_t elapsed,uint64_t duration){
    return duration?float(std::min(1.,double(elapsed)/double(duration))):1.f;
}
struct PreviewMapping {
    float old_scale,old_angle,old_pan_x,old_pan_y;
    float new_scale,new_angle,new_pan_x,new_pan_y;
};
inline PreviewMapping preview_mapping(const TransformParams&t,float phase){
    phase=std::clamp(phase,0.f,1.f);
    const float remain=1.f-phase;
    constexpr float radians=3.14159265358979323846f/180.f;
    // Map old -> intermediate, then intermediate -> new via the exact
    // composition T_full * inverse(T_partial). Translation is in pixels.
    PreviewMapping m{std::pow(t.zoom_factor,phase),t.rotation_deg*radians*phase,t.pan_x*phase,t.pan_y*phase,
        std::pow(t.zoom_factor,remain),t.rotation_deg*radians*remain,0,0};
    const float c=std::cos(m.new_angle),s=std::sin(m.new_angle);
    m.new_pan_x=t.pan_x-m.new_scale*(c*m.old_pan_x-s*m.old_pan_y);
    m.new_pan_y=t.pan_y-m.new_scale*(s*m.old_pan_x+c*m.old_pan_y);
    return m;
}
}
