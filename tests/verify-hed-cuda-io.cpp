#include "core/onnx_hed_processor.hpp"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <array>
#include <stdexcept>
using Clock=std::chrono::steady_clock;
struct Fixture {
    unsigned w,h;
    std::vector<uint8_t> rgb,soft,safe;
    double host_ms=0;
    Fixture(unsigned width,unsigned height):w(width),h(height),rgb(size_t(w)*h*3){
        for(unsigned y=0;y<h;++y)for(unsigned x=0;x<w;++x){
            auto i=(size_t(y)*w+x)*3;
            rgb[i]=x>w/5&&x<w*4/5?245:35;
            rgb[i+1]=y>h/6&&y<h*5/6?25:210;
            rgb[i+2]=(x+y)%256;
        }
    }
};
static double run(lll::OnnxHedProcessor& m,Fixture& f,bool safe,std::vector<uint8_t>&out){
    std::vector<double> times;
    for(int i=0;i<11;++i){
        auto begin=Clock::now();
        if(!m.estimate(f.rgb.data(),f.w,f.h,safe,out)||out.size()!=size_t(f.w)*f.h)
            throw std::runtime_error(m.last_error());
        if(i>=3)times.push_back(std::chrono::duration<double,std::milli>(Clock::now()-begin).count());
    }
    std::sort(times.begin(),times.end());return times[times.size()/2];
}
int main(int argc,char**argv){
    if(argc!=2)return 2;
    try{
        std::array<Fixture,3> cases={Fixture(192,128),Fixture(320,192),Fixture(512,320)};
        setenv("SIGHT_ONNX_DEVICE","cuda",1);setenv("SIGHT_HED_CUDA_IO","host",1);
        {
            lll::OnnxHedProcessor m;
            if(!m.load(argv[1]))throw std::runtime_error(m.last_error());
            for(auto& f:cases){f.host_ms=run(m,f,false,f.soft);run(m,f,true,f.safe);}
        }
        setenv("SIGHT_HED_CUDA_IO","device",1);
        lll::OnnxHedProcessor m;
        if(!m.load(argv[1]))throw std::runtime_error(m.last_error());
        std::vector<uint8_t> output;
        for(auto& f:cases){
            auto ms=run(m,f,false,output);
            auto before=m.tensor_io_stats();
            if(!m.estimate(f.rgb.data(),f.w,f.h,false,output))return 1;
            auto after=m.tensor_io_stats();
            uint64_t error=0;unsigned maximum=0;
            for(size_t i=0;i<output.size();++i){unsigned e=std::abs(int(output[i])-int(f.soft[i]));error+=e;maximum=std::max(maximum,e);}
            const size_t n=size_t(f.w)*f.h;
            const bool safe_ok=m.estimate(f.rgb.data(),f.w,f.h,true,output)&&output==f.safe;
            const bool ok=after.frames==before.frames+1&&after.prepares==before.prepares&&
                after.upload_bytes-before.upload_bytes==3*n&&after.download_bytes-before.download_bytes==n+4&&
                maximum<=1&&error<=n/100&&safe_ok;
            std::cout<<"SIGHT_HED_CUDA_IO "<<f.w<<"x"<<f.h<<" host_ms="<<f.host_ms<<" device_ms="<<ms
                <<" upload="<<after.upload_bytes-before.upload_bytes<<" download="<<after.download_bytes-before.download_bytes
                <<" prepares="<<after.prepares<<" mae="<<double(error)/n<<" max="<<maximum<<" safe_exact="<<safe_ok<<" pass="<<ok<<std::endl;
            if(!ok)return 1;
        }
        auto old=output;
        if(m.estimate(cases[0].rgb.data(),191,128,false,output)||output!=old)return 1;
        if(!m.estimate(cases[0].rgb.data(),192,128,false,output)||m.tensor_io_stats().prepares!=4)return 1;
        if(m.load("")||m.ready()||!m.load(argv[1])||!m.estimate(cases[0].rgb.data(),192,128,false,output)||m.tensor_io_stats().frames!=1)return 1;
        setenv("SIGHT_ONNX_DEVICE","cpu",1);
        if(m.load(argv[1])||m.ready())return 1;
        setenv("SIGHT_HED_CUDA_IO","host",1);
        if(!m.load(argv[1])||!m.estimate(cases[0].rgb.data(),192,128,false,output)||m.tensor_io_stats().frames)return 1;
        std::cout<<"SIGHT_HED_CUDA_IO_PASS=1\n";
    }catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}
}
