#include "core/segmentation_processor.hpp"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <array>
#include <stdexcept>
using Clock=std::chrono::steady_clock;
struct Fixture{
    int w,h;
    std::vector<uint8_t> rgb,reference;
    std::vector<float> canvas;
    double host_ms=0;
    Fixture(int width,int height):w(width),h(height),rgb(size_t(w)*h*3),reference(size_t(w)*h){
        for(int y=0;y<h;++y)for(int x=0;x<w;++x){
            size_t i=(size_t(y)*w+x)*3;
            rgb[i]=(x*3+y)&255;rgb[i+1]=(y*4)&255;rgb[i+2]=(x+y*2)&255;
        }
    }
};
double run(lll::SegmentationProcessor& m,Fixture& f,std::vector<uint8_t>&out){
    out.resize(size_t(f.w)*f.h);std::vector<double> times;
    for(int i=0;i<11;++i){
        auto begin=Clock::now();
        if(!m.process(f.rgb.data(),f.w,f.h,out.data()))throw std::runtime_error(m.last_error());
        if(i>=3)times.push_back(std::chrono::duration<double,std::milli>(Clock::now()-begin).count());
    }
    std::sort(times.begin(),times.end());return times[times.size()/2];
}
int main(int argc,char**argv){
    if(argc!=3)return 2;
    auto type=lll::SegModelType::U2Net;
    std::string family=argv[1];
    if(family=="modnet")type=lll::SegModelType::MODNet;
    else if(family=="longpipe")type=lll::SegModelType::Longpipe;
    else if(family!="u2net")return 2;
    try{
        std::array<Fixture,3> fixtures={Fixture(320,180),Fixture(512,512),Fixture(384,224)};
        setenv("SIGHT_ONNX_DEVICE","cuda",1);setenv("SIGHT_CUTOUT_CUDA_IO","host",1);
        {
            lll::SegmentationProcessor m;
            if(!m.load(argv[2],type,true,true))throw std::runtime_error(m.last_error());
            for(auto& f:fixtures){f.host_ms=run(m,f,f.reference);if(family=="longpipe")f.canvas=m.canvas_rgb();}
        }
        setenv("SIGHT_CUTOUT_CUDA_IO","device",1);
        lll::SegmentationProcessor m;
        if(!m.load(argv[2],type,true,true))throw std::runtime_error(m.last_error());
        std::vector<uint8_t> out;
        for(auto& f:fixtures){
            auto ms=run(m,f,out);auto before=m.tensor_io_stats();
            if(!m.process(f.rgb.data(),f.w,f.h,out.data()))return 1;
            auto after=m.tensor_io_stats();size_t n=out.size();
            uint64_t err=0;unsigned mx=0;
            for(size_t i=0;i<n;++i){unsigned e=std::abs(int(out[i])-int(f.reference[i]));err+=e;mx=std::max(mx,e);}
            bool canvas=family!="longpipe"||m.canvas_rgb()==f.canvas;
            bool ok=after.frames==before.frames+1&&after.prepares==before.prepares&&after.upload_bytes-before.upload_bytes==3*n&&after.download_bytes-before.download_bytes==n+4&&mx<=1&&err<=n/100&&canvas;
            std::cout<<"SIGHT_CUTOUT_CUDA_IO "<<family<<" "<<f.w<<"x"<<f.h<<" host_ms="<<f.host_ms<<" device_ms="<<ms<<" upload="<<after.upload_bytes-before.upload_bytes<<" download="<<after.download_bytes-before.download_bytes<<" mae="<<double(err)/n<<" max="<<mx<<" canvas="<<canvas<<" pass="<<ok<<std::endl;
            if(!ok)return 1;
        }
        if(m.tensor_io_stats().prepares!=3)return 1;
        auto& f=fixtures[0];out.resize(f.reference.size());
        if(!m.process(f.rgb.data(),f.w,f.h,out.data())||m.tensor_io_stats().prepares!=4)return 1;
        if(m.load("",type,true,true)||m.is_ready()||!m.load(argv[2],type,true,true)||!m.process(f.rgb.data(),f.w,f.h,out.data())||m.tensor_io_stats().frames!=1)return 1;
        setenv("SIGHT_ONNX_DEVICE","cpu",1);
        if(m.load(argv[2],type,false,false)||m.is_ready())return 1;
        setenv("SIGHT_CUTOUT_CUDA_IO","host",1);
        if(!m.load(argv[2],type,false,false)||!m.process(f.rgb.data(),f.w,f.h,out.data())||m.tensor_io_stats().frames)return 1;
        std::cout<<"SIGHT_CUTOUT_CUDA_IO_PASS "<<family<<std::endl;
    }catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}
}
