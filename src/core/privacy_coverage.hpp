#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <charconv>
#include <string>
#include <string_view>
#include <vector>
#include "fast-face-detector.hpp"
namespace sight::privacy {
struct Rect { float x0=0,y0=0,x1=0,y1=0; };
// Bounding boxes are from the *actual* native multi-face detector, not a
// single face landmarker; expand for hair/edges/short-lived motion.
inline std::vector<Rect> coverage(const std::vector<FastFace>& faces,
                                  unsigned width,unsigned height) {
    std::vector<Rect> boxes;
    if(!width||!height)return boxes;
    for(const auto& f:faces) {
        if(f.w<=0||f.h<=0||f.x<0||f.y<0||
           f.x+f.w>int(width)||f.y+f.h>int(height))continue;
        const float cx=(f.x+f.w*.5f)/width,cy=(f.y+f.h*.5f)/height;
        const float rw=std::max(.09f,f.w/float(width)*1.6f);
        const float rh=std::max(.12f,f.h/float(height)*1.75f);
        boxes.push_back({std::clamp(cx-rw*.5f,0.f,1.f),
                         std::clamp(cy-rh*.5f,0.f,1.f),
                         std::clamp(cx+rw*.5f,0.f,1.f),
                         std::clamp(cy+rh*.5f,0.f,1.f)});
    }
    return boxes;
}
inline void paint_mask(std::vector<uint8_t>& mask,unsigned w,unsigned h,
                       const std::vector<Rect>& faces) {
    mask.assign(size_t(w)*h,0);
    for(const auto& f:faces) {
        unsigned x0=std::min(w,unsigned(f.x0*w)),x1=std::min(w,unsigned(std::ceil(f.x1*w)));
        unsigned y0=std::min(h,unsigned(f.y0*h)),y1=std::min(h,unsigned(std::ceil(f.y1*h)));
        for(unsigned y=y0;y<y1;++y)
            std::fill_n(mask.begin()+size_t(y)*w+x0,x1-x0,uint8_t(255));
    }
}
// ROI contract for a future ONNX privacy classifier + OCR pipeline.
// A text classifier produces character spans, NOT screen coordinates.
// Caller must supply a genuine OCR word/line box; conceal the full line
// to avoid reconstructing sensitive fragments from partially covered glyphs.
struct TextSpan {size_t start=0,end=0;};
struct TextEvidence {
    std::string text;
    Rect box;
    std::vector<TextSpan> spans;
};
inline bool valid_rect(const Rect& r) {
    return std::isfinite(r.x0)&&std::isfinite(r.y0)&&
        std::isfinite(r.x1)&&std::isfinite(r.y1)&&
        r.x0>=0.f&&r.y0>=0.f&&r.x1<=1.f&&r.y1<=1.f&&
        r.x0<r.x1&&r.y0<r.y1;
}
inline std::vector<Rect> text_coverage(const std::vector<TextEvidence>& lines) {
    std::vector<Rect> zones;
    for(const auto& line:lines) {
        if(!valid_rect(line.box)||line.text.empty())continue;
        bool contains_valid_detection=false;
        for(const auto& span:line.spans) {
            if(span.start<span.end&&span.end<=line.text.size()) {
                contains_valid_detection=true;break;
            }
        }
        if(!contains_valid_detection)continue;
        constexpr float padding=.012f;
        zones.push_back({
            std::max(0.f,line.box.x0-padding),
            std::max(0.f,line.box.y0-padding),
            std::min(1.f,line.box.x1+padding),
            std::min(1.f,line.box.y1+padding),
        });
    }
    return zones;
}
inline std::string_view trim_zone(std::string_view s) {
    const auto first=s.find_first_not_of(" \r\n\t");
    if(first==std::string_view::npos)return {};
    return s.substr(first,s.find_last_not_of(" \r\n\t")-first+1);
}
// Optional manually protected text ROI list for scenes without an OCR provider.
// A failed parse preserves the existing valid zones, never partially installs
// malformed rectangles. Up to 32 normalized "x0,y0,x1,y1;..." entries.
inline bool parse_text_zones(std::string_view specification,std::vector<Rect>& zones) {
    std::vector<Rect> parsed;
    specification=trim_zone(specification);
    if(specification.empty()){zones.clear();return true;}
    while(!specification.empty()) {
        if(parsed.size()>=32)return false;
        const size_t separator=specification.find(';');
        const auto chunk=trim_zone(specification.substr(0,separator));
        if(chunk.empty())return false;
        float numbers[4]{};
        auto rest=chunk;
        for(int n=0;n<4;++n) {
            const auto comma=rest.find(',');
            if((n<3&&comma==std::string_view::npos)||
               (n==3&&comma!=std::string_view::npos))return false;
            const auto part=trim_zone(rest.substr(0,comma));
            if(part.empty())return false;
            auto [end,ec]=std::from_chars(part.data(),part.data()+part.size(),
                                           numbers[n],std::chars_format::general);
            if(ec!=std::errc{}||end!=part.data()+part.size()||!std::isfinite(numbers[n]))
                return false;
            if(n<3)rest.remove_prefix(comma+1);
        }
        Rect rect{numbers[0],numbers[1],numbers[2],numbers[3]};
        if(!valid_rect(rect))return false;
        parsed.push_back(rect);
        if(separator==std::string_view::npos)break;
        specification=trim_zone(specification.substr(separator+1));
        if(specification.empty())return false;
    }
    zones=std::move(parsed);
    return true;
}
// Retain an *already protected* frame across transient detector dropouts.
// This is a maximum time since approval, not since capture: model inference
// can take longer than one frame. No unapproved camera pixels may be shown.
inline constexpr uint64_t kMaxProtectedHoldMs=1800;
// An extremely late detection result is not useful for a live video filter,
// even though its source frame could still be safely anonymized.
inline constexpr uint64_t kMaxInferenceLatencyMs=4000;
inline bool hide_entire_frame(bool ready,bool approved,bool mask_ready,
                              uint64_t now_ms,uint64_t approved_at_ms) {
    return !ready||!approved||!mask_ready||!approved_at_ms||
           now_ms<approved_at_ms||
           now_ms-approved_at_ms>kMaxProtectedHoldMs;
}
// Zero faces is a successful detection outcome, not a reason to conceal.
inline bool detection_result_usable(bool valid,
                                   uint64_t now_ms,uint64_t captured_at_ms) {
    return valid&&captured_at_ms>0&&now_ms>=captured_at_ms&&
           now_ms-captured_at_ms<=kMaxInferenceLatencyMs;
}
} // namespace sight::privacy
