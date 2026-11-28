#pragma once
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace sight {
// Owned CPU staging data. Swap the frame between the render-side scratch,
// pending slot, and worker; no consumer sees mapped GPU memory.
struct FlowCaptureFrame {
    std::vector<uint8_t> rgba;
    unsigned w=0,h=0,sw=0,sh=0;
    uint64_t revision=0;
    std::chrono::steady_clock::time_point captured{};

    bool copy_rgba(const uint8_t* bytes,size_t pitch,unsigned width,unsigned height) {
        if(!bytes||!width||!height||width>640||height>640||
           pitch<size_t(width)*4)return false;
        const size_t row=size_t(width)*4;
        rgba.resize(row*height);
        for(unsigned y=0;y<height;++y)
            std::memcpy(rgba.data()+size_t(y)*row,bytes+size_t(y)*pitch,row);
        w=width;h=height;
        return true;
    }
};
}
