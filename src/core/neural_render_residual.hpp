#pragma once
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace sight::nr {
// NRF1: four-byte magic followed by little-endian float32 interleaved RGB.
// The renderer uploads RGBA32F; alpha of the signed correction is zero.
inline bool decode_float_residual(const std::vector<uint8_t>& bytes,size_t pixels,
                                  std::vector<float>& delta){
    delta.clear();
    if(pixels==0||pixels>(std::numeric_limits<size_t>::max()-4)/12||
       bytes.size()!=4+pixels*12||bytes[0]!='N'||bytes[1]!='R'||bytes[2]!='F'||bytes[3]!='1')return false;
    delta.resize(pixels*4,0.f);
    for(size_t i=0;i<pixels;++i)for(unsigned c=0;c<3;++c){
        size_t k=4+(i*3+c)*4;
        uint32_t u=uint32_t(bytes[k])|(uint32_t(bytes[k+1])<<8)|
            (uint32_t(bytes[k+2])<<16)|(uint32_t(bytes[k+3])<<24);
        float v=std::bit_cast<float>(u);
        if(!std::isfinite(v)||std::abs(v)>16.f){delta.clear();return false;}
        delta[i*4+c]=v;
    }
    return true;
}
}
