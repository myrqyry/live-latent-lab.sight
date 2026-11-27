#include "core/neural_render_residual.hpp"
#include "core/neural_render_process.hpp"
#include <bit>
#include <iostream>
#include <limits>

int main(int argc,char**argv){
    using namespace sight::nr;
    std::vector<uint8_t> packet{'N','R','F','1'};
    auto append=[&](float v){
        uint32_t u=std::bit_cast<uint32_t>(v);
        for(int i=0;i<4;++i)packet.push_back(uint8_t(u>>(8*i)));
    };
    append(.0007f);append(-.012f);append(1.25f);
    if(argc==2){
        char rgb[3];
        if(!std::cin.read(rgb,3))return 2;
        std::cout.write(reinterpret_cast<const char*>(packet.data()),packet.size());
        std::cout.flush();return 0;
    }
    std::vector<float> delta;
    const auto good=decode_float_residual(packet,1,delta)&&
        delta==std::vector<float>{.0007f,-.012f,1.25f,0.f};
    auto invalid=packet;invalid[0]='X';
    bool reject= !decode_float_residual(invalid,1,delta)&&delta.empty();
    invalid=packet;invalid.pop_back();
    reject &= !decode_float_residual(invalid,1,delta);
    packet.resize(4);append(std::numeric_limits<float>::quiet_NaN());append(0);append(0);
    reject &= !decode_float_residual(packet,1,delta);
    packet.resize(4);append(17.f);append(0);append(0);
    reject &= !decode_float_residual(packet,1,delta);
    Process process;
    std::vector<uint8_t> response;
    bool transport=process.start({"/proc/self/exe","--child"})&&
        process.exchange({255,128,0},response,2000,[]{return false;},16)&&
        decode_float_residual(response,1,delta)&&delta[0]==.0007f;
    std::cout<<"fractional_signed="<<good<<" reject_malformed="<<reject<<" sized_transport="<<transport<<"\n";
    return good&&reject&&transport?0:1;
}
