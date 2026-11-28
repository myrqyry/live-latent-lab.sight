// Real model, matched CUDA provider on both sides. No CPU fallback.
#include "core/onnx_depth_processor.hpp"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <array>
#include <stdexcept>
using Clock=std::chrono::steady_clock;
struct Fixture{
    unsigned w,h;
    std::vector<uint8_t> rgb,reference;
    double host_ms=0;
    Fixture(unsigned width,unsigned height):w(width),h(height),rgb(size_t(w)*h*3){
        for(unsigned y=0;y<h;++y)for(unsigned x=0;x<w;++x){
            auto i=(size_t(y)*w+x)*3;
            rgb[i]=(x*3+y)&255;rgb[i+1]=(y*4)&255;rgb[i+2]=(x+y*2)&255;
        }
    }
};
static double run(lll::OnnxDepthProcessor& model,Fixture& f,std::vector<uint8_t>&out){
    std::vector<double> times;
    for(int i=0;i<13;++i){
        const auto begin=Clock::now();
        if(!model.estimate(f.rgb.data(),f.w,f.h,out)||out.size()!=size_t(f.w)*f.h)
            throw std::runtime_error("Inference: "+model.last_error());
        if(i>=3)times.push_back(std::chrono::duration<double,std::milli>(Clock::now()-begin).count());
    }
    std::sort(times.begin(),times.end());return times[times.size()/2];
}
int main(int argc,char**argv){
    if(argc!=2)return 2;
    try{
        std::array<Fixture,3> cases={Fixture(252,140),Fixture(336,196),Fixture(518,294)};
        setenv("SIGHT_ONNX_DEVICE","cuda",1);
        setenv("SIGHT_DEPTH_CUDA_IO","host",1);
        {
            lll::OnnxDepthProcessor model;
            if(!model.load(argv[1])||!model.configured_provider()||
               std::string(model.configured_provider()).find("CUDA")==std::string::npos)
                throw std::runtime_error("Baseline CUDA unavailable: "+model.last_error());
            for(auto&f:cases)f.host_ms=run(model,f,f.reference);
        }
        setenv("SIGHT_DEPTH_CUDA_IO","device",1);
        lll::OnnxDepthProcessor model;
        if(!model.load(argv[1]))throw std::runtime_error(model.last_error());
        std::vector<uint8_t> result;
        for(auto&f:cases){
            auto ms=run(model,f,result);
            auto before=model.tensor_io_stats();
            if(!model.estimate(f.rgb.data(),f.w,f.h,result))return 1;
            auto after=model.tensor_io_stats();
            uint64_t error=0;unsigned maximum=0;
            for(size_t i=0;i<result.size();++i){
                const unsigned e=std::abs(int(result[i])-int(f.reference[i]));error+=e;maximum=std::max(maximum,e);
            }
            const auto n=size_t(f.w)*f.h;
            const bool ok=after.frames==before.frames+1&&after.prepares==before.prepares&&
                after.upload_bytes-before.upload_bytes==n*3&&after.download_bytes-before.download_bytes==n+4&&
                maximum<=2&&error<=n/100;
            std::cout<<"SIGHT_DEPTH_CUDA_IO "<<f.w<<"x"<<f.h<<" host_ms="<<f.host_ms<<" device_ms="<<ms
                <<" upload="<<after.upload_bytes-before.upload_bytes<<" download="<<after.download_bytes-before.download_bytes
                <<" prepares="<<after.prepares<<" frames="<<after.frames<<" mae="<<double(error)/n
                <<" max="<<maximum<<" pass="<<ok<<"\n";
            if(!ok)return 1;
        }
        if(model.tensor_io_stats().prepares!=3)return 1;
        const auto old=result;
        if(model.estimate(cases[0].rgb.data(),251,140,result)||result!=old)return 1;
        // Revisit first geometry, not another model load.
        if(!model.estimate(cases[0].rgb.data(),252,140,result)||model.tensor_io_stats().prepares!=4)return 1;
        // Repeated loading and invalid path must release old bindings/session.
        if(model.load("")||model.ready()||!model.load(argv[1])||
           !model.estimate(cases[0].rgb.data(),252,140,result)||model.tensor_io_stats().frames!=1)return 1;
        setenv("SIGHT_ONNX_DEVICE","cpu",1);
        if(model.load(argv[1])||model.ready())return 1;
        setenv("SIGHT_DEPTH_CUDA_IO","host",1);
        if(!model.load(argv[1])||!model.estimate(cases[0].rgb.data(),252,140,result)||
           model.tensor_io_stats().frames!=0||
           std::string(model.configured_provider()).find("CPU")==std::string::npos)return 1;
        std::cout<<"SIGHT_DEPTH_CUDA_IO_PASS=1\n";
    }catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}
}
