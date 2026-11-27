#pragma once
#include "common/image-utils.hpp"
#include "core/preprocessors_processor.hpp"
#include <cstdint>
#include <vector>
#include <string_view>

namespace sight::sd {
inline bool valid_feedback_size(uint32_t w,uint32_t h){
    return w>=64&&h>=64&&w<=1024&&h<=1024&&w%64==0&&h%64==0;
}
inline int64_t feedback_seed(int64_t base,uint64_t frame,uint32_t stride){
    return int64_t((uint64_t(base)+frame*uint64_t(stride))&uint64_t(INT64_MAX));
}
class Feedback {
    uint32_t w_=0,h_=0;
    std::vector<uint8_t> rgba_;
public:
    void reset(){rgba_.clear();w_=h_=0;}
    bool accept(const std::vector<uint8_t>&pixels,uint32_t w,uint32_t h,uint32_t channels){
        if(!valid_feedback_size(w,h)||(channels!=3&&channels!=4)||pixels.size()!=size_t(w)*h*channels)return false;
        rgba_.resize(size_t(w)*h*4);
        for(size_t i=0;i<size_t(w)*h;++i){
            for(unsigned c=0;c<3;++c)rgba_[i*4+c]=pixels[i*channels+c];
            rgba_[i*4+3]=255;
        }
        w_=w;h_=h;return true;
    }
    const std::vector<uint8_t>&rgba()const{return rgba_;}
    std::vector<uint8_t> init_rgb(uint32_t w,uint32_t h,const TransformParams&t)const{
        if(w!=w_||h!=h_||rgba_.empty()||
           !std::isfinite(t.zoom_factor)||t.zoom_factor<.9f||t.zoom_factor>1.1f||
           !std::isfinite(t.pan_x)||std::abs(t.pan_x)>100||
           !std::isfinite(t.pan_y)||std::abs(t.pan_y)>100||
           !std::isfinite(t.rotation_deg)||std::abs(t.rotation_deg)>15)return {};
        std::vector<uint8_t> rgb(size_t(w)*h*3);
        if(t.zoom_factor==1.f&&t.pan_x==0.f&&t.pan_y==0.f&&t.rotation_deg==0.f){
            lll::convert_rgba_to_rgb(rgba_.data(),rgb.data(),size_t(w)*h);
        }else{
            std::vector<uint8_t> transformed(rgba_.size());
            if(!transform_image(rgba_.data(),rgba_.size(),w,h,transformed.data(),transformed.size(),t))return {};
            lll::convert_rgba_to_rgb(transformed.data(),rgb.data(),size_t(w)*h);
        }
        return rgb;
    }
};
inline bool make_native_guide(const std::vector<uint8_t>&rgb,uint32_t w,uint32_t h,
                              std::string_view mode,std::vector<uint8_t>&guide){
    guide.clear();
    if(!valid_feedback_size(w,h)||rgb.size()!=size_t(w)*h*3||
       (mode!="canny"&&mode!="tile"))return false;
    std::vector<uint8_t> rgba(size_t(w)*h*4),out;
    lll::convert_rgb_to_rgba(rgb.data(),rgba.data(),size_t(w)*h);
    if(!preprocessors::process_rgba(rgba.data(),w,h,w*4,mode,{},out)||
       out.size()!=rgba.size())return false;
    guide.resize(rgb.size());
    lll::convert_rgba_to_rgb(out.data(),guide.data(),size_t(w)*h);
    return true;
}
}
