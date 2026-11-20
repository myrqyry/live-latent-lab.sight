#pragma once
#include <cstring>
#include <string>
namespace sight {
enum class TrackingMode { None, Face, Body, Hand, Holistic, Object, Rtmw };
inline constexpr TrackingMode tracking_modes[] = {
    TrackingMode::Face, TrackingMode::Body, TrackingMode::Hand, TrackingMode::Holistic,
    TrackingMode::Object,
#ifdef SIGHT_HAVE_RTMW
    TrackingMode::Rtmw,
#endif
};
inline const char* tracking_mode_id(TrackingMode mode) {
    switch(mode) {
        case TrackingMode::Face:return "face";
        case TrackingMode::Body:return "body";
        case TrackingMode::Hand:return "hand";
        case TrackingMode::Holistic:return "holistic";
        case TrackingMode::Object:return "object";
        case TrackingMode::Rtmw:return "rtmw";
        default:return "none";
    }
}
inline TrackingMode parse_tracking_mode(const char* id) {
    for(auto mode:tracking_modes)if(id&&!strcmp(id,tracking_mode_id(mode)))return mode;
    return TrackingMode::None;
}
inline std::string tracking_mode_key(TrackingMode mode,const char* key) {
    return std::string(tracking_mode_id(mode))+"_"+key;
}
}
