#include "hand_signals.hpp"
#include <algorithm>
#include <cmath>
namespace sight {
float normalized_pinch(const std::vector<TrackedPoint>& hand) {
    if(hand.size()!=21)return 0;
    const auto distance=[&](unsigned a,unsigned b) {
        const auto& x=hand[a];
        const auto& y=hand[b];
        return std::hypot(double(x.x)-y.x,double(x.y)-y.y);
    };
    // Thumb tip (4) to index tip (8), scaled by a stable hand-size
    // reference (wrist 0 to middle MCP 9). 1 = pinched, 0 = open.
    const auto palm=distance(0,9);
    const auto fingertips=distance(4,8);
    if(!std::isfinite(palm)||!std::isfinite(fingertips)||palm<.02)
        return 0;
    return float(1.-std::clamp((fingertips/palm-.15)/.85,0.,1.));
}
HandGesture parse_hand_gesture(std::string_view name) {
    if(name=="Closed_Fist")return HandGesture::ClosedFist;
    if(name=="Open_Palm")return HandGesture::OpenPalm;
    if(name=="Pointing_Up")return HandGesture::PointingUp;
    if(name=="Thumb_Down")return HandGesture::ThumbDown;
    if(name=="Thumb_Up")return HandGesture::ThumbUp;
    if(name=="Victory")return HandGesture::Victory;
    if(name=="ILoveYou")return HandGesture::ILoveYou;
    return HandGesture::None;
}
float gesture_match(const HandData& hand,HandGesture target) {
    if(target==HandGesture::None)return 0;
    return parse_hand_gesture(hand.gesture)==target?
        std::clamp(hand.gesture_score,0.f,1.f):0.f;
}
}
