#include "core/flow_capture_buffer.hpp"
#include <array>
#include <cstdio>
#include <utility>
#define CHECK(x) do{if(!(x)){std::fprintf(stderr,"failed: %s\n",#x);return 1;}}while(0)
int main(){
    sight::FlowCaptureFrame render, pending, worker;
    std::array<uint8_t,40> padded{};
    for(unsigned i=0;i<padded.size();++i)padded[i]=uint8_t(i);
    const uint8_t *buffers[2]{};
    for(unsigned i=0;i<200;++i){
        padded[0]=uint8_t(i);
        CHECK(render.copy_rgba(padded.data(),20,4,2));
        render.sw=1920;render.sh=1080;render.revision=i+1;
        if(i<2)buffers[i]=render.rgba.data();
        CHECK(render.rgba.data()==buffers[i%2]);
        std::swap(render,pending);
        std::swap(worker,pending);
        CHECK(worker.revision==i+1&&worker.sw==1920&&worker.sh==1080);
        CHECK(worker.rgba.size()==32&&worker.rgba[0]==i&&worker.rgba[16]==20&&worker.rgba[31]==35);
        // Return the storage only after processing; no aliasing with render.
        if(!render.rgba.empty())CHECK(render.rgba.data()!=worker.rgba.data());
        std::swap(worker,pending);
    }
    CHECK(buffers[0]!=buffers[1]);
    CHECK(!render.copy_rgba(nullptr,20,4,2));
    CHECK(!render.copy_rgba(padded.data(),15,4,2));
    CHECK(!render.copy_rgba(padded.data(),20,0,2));
    CHECK(!render.copy_rgba(padded.data(),20,641,2));
    CHECK(render.copy_rgba(padded.data(),20,3,1)&&render.rgba.size()==12);
    CHECK(render.copy_rgba(padded.data(),20,4,2)&&render.rgba.size()==32);
    std::puts("FLOW_CAPTURE_BUFFER reused=200 padded_stride=1 independent=1 invalid=1 resize=1");
}
