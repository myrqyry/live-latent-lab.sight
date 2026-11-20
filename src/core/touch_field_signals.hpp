#pragma once

#include "tracking_signal_bus.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cmath>

namespace sight {
// Owned by one OBS consumer. No inferences, allocations or scene transforms.
// Each point is normalized to the producing native OBS video, not the
// downscaled inference image or OBS canvas coordinates.
struct TouchPoint {
    float x=0.f, y=0.f, pinch=0.f, active=0.f;
};
struct TouchFields {
    std::array<TouchPoint,2> fingers{};
    unsigned count=0;
};
inline TouchFields select_touch_fingertips(const TrackingSignal& signal,
                                          uint32_t target_width,
                                          uint32_t target_height) {
    TouchFields result;
    if(signal.kind!=TrackingKind::Hand||!target_width||!target_height||
       signal.source_width!=target_width||signal.source_height!=target_height)
        return result;
    for(const auto& hand:signal.data.hands) {
        if(result.count==2)break;
        if(hand.landmarks.size()!=21)continue;
        const auto& index=hand.landmarks[8];
        if(!std::isfinite(index.x)||!std::isfinite(index.y)||
           index.x<0.f||index.x>1.f||index.y<0.f||index.y>1.f||
           !std::isfinite(hand.pinch))
            continue;
        result.fingers[result.count++]={index.x,index.y,
            std::clamp(hand.pinch,0.f,1.f),1.f};
    }
    return result;
}
}
