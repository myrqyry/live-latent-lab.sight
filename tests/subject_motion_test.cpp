#include "core/subject_motion.hpp"
#include <cmath>
#include <iostream>
using namespace std::chrono_literals;
int main() {
    sight::SubjectMotion motion;
    auto t=std::chrono::steady_clock::time_point{}+1s;
    std::vector<uint8_t> black(12,0), white(12,255), half=black;
    for (size_t i=0;i<6;++i) half[i]=255;
    auto sample=[&](const auto& rgb,auto delta, uint32_t w=2,uint32_t h=2,uint32_t sw=640,uint32_t sh=480) {
        t+=delta; return motion.update(rgb,w,h,sw,sh,t);
    };
    auto close=[](auto r,float e){return r.valid && std::abs(r.energy-e)<.001f;};
    bool first=!sample(black,0ms).valid;
    bool still=close(sample(black,200ms),0);
    bool moving=close(sample(half,200ms),.5f)&&close(sample(white,200ms),.5f);
    bool full=close(sample(black,200ms),1);
    bool gap=!sample(white,1500ms).valid && close(sample(white,200ms),0);
    bool resized=!sample(black,200ms,2,2,800,600).valid &&
        close(sample(black,200ms,2,2,800,600),0) &&
        !sample(black,200ms,1,4,800,600).valid;
    motion.reset();
    bool reset=!sample(white,200ms).valid && !sample(black,0ms).valid;
    bool malformed=!sample(std::vector<uint8_t>{1},200ms).valid &&
        !sample(white,200ms).valid;
    // Larger inference canvases must preserve motion without growing the
    // bounded analysis history or changing the original source geometry.
    motion.reset();
    constexpr uint32_t W=1280,H=768;
    std::vector<uint8_t> big(size_t(W)*H*3,0),big_half=big;
    for(uint32_t y=0;y<H;++y)
        for(uint32_t x=0;x<W/2;++x)
            for(unsigned c=0;c<3;++c)big_half[(size_t(y)*W+x)*3+c]=255;
    bool big_ok=!sample(big,200ms,W,H,W,H).valid &&
        close(sample(big,200ms,W,H,W,H),0) &&
        close(sample(big_half,200ms,W,H,W,H),.5f) &&
        close(sample(big_half,200ms,W,H,W,H),0) &&
        !sample(big_half,200ms,W,H,H,W).valid &&
        close(sample(big_half,200ms,W,H,H,W),0);
    std::cout<<"large_canvas="<<big_ok<<"\n";
    std::cout<<"first="<<first<<" still="<<still<<" moving="<<moving<<" full="<<full
        <<" gap="<<gap<<" resize="<<resized<<" reset="<<reset<<" malformed="<<malformed<<"\n";
    return !(first&&still&&moving&&full&&gap&&resized&&reset&&malformed&&big_ok);
}
