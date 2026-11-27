#pragma once
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

namespace sight::sd {
// The SD1.x pipeline aligns spatial dimensions to 64 pixels. Do not
// allow upstream's implicit alignment to crop a live video source.
// Other model families must supply their own spatial multiple.
struct FrameGeometry {
    const uint32_t source_width,source_height,width,height,left,top;
    FrameGeometry(uint32_t w,uint32_t h,uint32_t multiple)
        :source_width(w),source_height(h),width(aligned(w,multiple)),
         height(aligned(h,multiple)),left((width-w)/2),top((height-h)/2){
        if(uint64_t(width)*height>4096ull*4096)
            throw std::invalid_argument("Diffusion frame exceeds pixel budget");
    }
    std::vector<uint8_t> pad_rgb(const std::vector<uint8_t>& rgb) const {
        if(rgb.size()!=size_t(source_width)*source_height*3)
            throw std::invalid_argument("Source RGB size mismatch");
        if(width==source_width&&height==source_height)return rgb;
        std::vector<uint8_t> out(size_t(width)*height*3);
        for(uint32_t y=0;y<height;++y){
            const auto sy=uint32_t(std::clamp(int(y)-int(top),0,int(source_height)-1));
            auto*dst=out.data()+size_t(y)*width*3;
            const auto*src=rgb.data()+size_t(sy)*source_width*3;
            for(uint32_t x=0;x<left;++x)std::memcpy(dst+x*3,src,3);
            std::memcpy(dst+left*3,src,size_t(source_width)*3);
            for(uint32_t x=left+source_width;x<width;++x)
                std::memcpy(dst+x*3,src+(source_width-1)*3,3);
        }
        return out;
    }
    std::vector<uint8_t> crop_rgb(const std::vector<uint8_t>& rgb) const {
        if(rgb.size()!=size_t(width)*height*3)
            throw std::invalid_argument("Aligned RGB size mismatch");
        if(width==source_width&&height==source_height)return rgb;
        std::vector<uint8_t> out(size_t(source_width)*source_height*3);
        for(uint32_t y=0;y<source_height;++y)
            std::memcpy(out.data()+size_t(y)*source_width*3,
                rgb.data()+(size_t(y+top)*width+left)*3,size_t(source_width)*3);
        return out;
    }
private:
    static uint32_t aligned(uint32_t n,uint32_t multiple){
        if(!n||!multiple||n>4096||multiple>4096)
            throw std::invalid_argument("Invalid diffusion frame dimensions");
        const auto result=((n+multiple-1)/multiple)*multiple;
        if(result>4096)throw std::invalid_argument("Aligned dimension exceeds limit");
        return result;
    }
};
}
