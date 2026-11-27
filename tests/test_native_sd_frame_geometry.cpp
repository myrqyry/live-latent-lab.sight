#include "core/native_sd_frame_geometry.hpp"
#include <cassert>
#include <iostream>
using namespace sight::sd;
int main(){
    for(auto [w,h]: {std::pair{512u,288u},std::pair{288u,512u},std::pair{257u,259u},std::pair{256u,256u}}){
        const FrameGeometry g(w,h,64);
        assert(g.width%64==0&&g.height%64==0);
        assert(g.width>=w&&g.height>=h&&g.width-w<64&&g.height-h<64);
        std::vector<uint8_t> input(size_t(w)*h*3);
        for(size_t i=0;i<input.size();++i)input[i]=uint8_t(i*17+i/23);
        auto aligned=g.pad_rgb(input);
        assert(aligned.size()==size_t(g.width)*g.height*3);
        assert(g.crop_rgb(aligned)==input);
        // Replicate border pixels, not black bands, so padding does not
        // introduce artificial contrast into the diffusion input.
        assert(aligned[0]==input[0]);
        assert(aligned.back()==input.back());
        bool rejected=false;
        try{g.crop_rgb(std::vector<uint8_t>(aligned.size()-1));}catch(const std::invalid_argument&){rejected=true;}
        assert(rejected);
    }
    for(auto [w,h]: {std::pair{0u,32u},std::pair{32u,0u},std::pair{UINT32_MAX,32u}}){
        bool rejected=false;
        try{FrameGeometry g(w,h,64);}catch(const std::invalid_argument&){rejected=true;}
        assert(rejected);
    }
    std::cout<<"PASS native diffusion frame geometry: exact round trip, edge replication, invalid dimensions\n";
}
