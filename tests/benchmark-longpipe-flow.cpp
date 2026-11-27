// Real production processors, controlled translated/relit RGB input.
// Reference is the first predicted matte, NOT ground truth.
#include "core/segmentation_processor.hpp"
#include "core/longpipe-flow.hpp"
#include "core/subject_matte_history.hpp"
#include <array>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <numeric>
using Clock=std::chrono::steady_clock;
using namespace std::chrono_literals;
using Bytes=std::vector<uint8_t>;
struct Metric {
    double temporal=0,reference=0;size_t n=0,edges=0;double edge=0;
    std::vector<double> times;
    Bytes previous;
    int previous_shift=0;
    void sample(const Bytes&a,const Bytes&ref,unsigned w,unsigned h,int shift,double ms){
        times.push_back(ms);
        // Measure the overlapping central domain in original coordinates.
        // Ignore translated/padded borders; compare exactly aligned pixels.
        for(unsigned y=16;y+16<h;++y)for(unsigned x=40;x+40<w;++x){
            const auto i=size_t(y)*w+x;
            const int v=a[size_t(y)*w+x+shift];
            reference+=std::abs(v-int(ref[i]));++n;
            if(ref[i]>12&&ref[i]<243){edge+=std::abs(v-int(ref[i]));++edges;}
            if(!previous.empty())temporal+=std::abs(v-int(previous[size_t(y)*w+x+previous_shift]));
        }
        previous=a;previous_shift=shift;
    }
    void print(const char*mode,const char*sequence){
        auto sorted=times;std::sort(sorted.begin(),sorted.end());
        const double tn=double(n)*(times.size()-1)/times.size();
        std::cout<<"RESULT "<<sequence<<" "<<mode<<" reference_mae="<<reference/(255*n)
          <<" edge_mae="<<(edges?edge/(255*edges):0)
          <<" aligned_temporal_mae="<<(tn?temporal/(255*tn):0)
          <<" extra_p50_ms="<<sorted[sorted.size()/2]<<"\n";
    }
};
int main(int argc,char**argv){
    if(argc!=7){std::cerr<<"usage: benchmark matting.onnx flow.onnx cpu|cuda width height image.rgb\n";return 2;}
    const bool gpu=std::string(argv[3])=="cuda";
    unsigned w=std::stoul(argv[4]),h=std::stoul(argv[5]);
    if(w<100||h<100||w>1280||h>1280)return 2;
    std::ifstream f(argv[6],std::ios::binary);Bytes input((std::istreambuf_iterator<char>(f)),{});
    const size_t n=size_t(w)*h;if(input.size()!=3*n)return 2;
    lll::SegmentationProcessor matte;
    if(!matte.load(argv[1],lll::SegModelType::Longpipe,gpu,gpu))return 1;
    sight::LongpipeFlow flow;
    flow.load(argv[2],matte.input_width(),matte.input_height(),gpu,gpu);
    if(!flow.ready()){std::cerr<<flow.status()<<"\n";return 1;}
    Bytes ref(n),raw(n),rgb(3*n);
    for(int i=0;i<8;++i)if(!matte.process(input.data(),w,h,ref.data()))return 1;
    auto [lo,hi]=std::minmax_element(ref.begin(),ref.end());if(*hi<*lo+20)return 1;
    std::cout<<std::fixed<<std::setprecision(6);
    for(auto sequence:{"relight","translate"}){
        flow.reset();sight::SubjectMatteHistory ordinary;Metric mr,mo,mf;
        size_t changed=0;std::vector<double> base_ms;
        auto t=Clock::now();
        for(int k=0;k<24;++k){
            const int shift=std::string(sequence)=="translate"?2*(k<12?k:24-k):0;
            const float gain=1.f+float((k%3)-1)*.035f;
            for(unsigned y=0;y<h;++y)for(unsigned x=0;x<w;++x)for(unsigned c=0;c<3;++c){
                int src=int(x)-shift;
                const auto val=src>=0&&src<int(w)?input[(size_t(y)*w+src)*3+c]:80;
                rgb[(size_t(y)*w+x)*3+c]=uint8_t(std::clamp(std::lround(val*gain),0L,255L));
            }
            auto start=Clock::now();if(!matte.process(rgb.data(),w,h,raw.data()))return 1;
            base_ms.push_back(std::chrono::duration<double,std::milli>(Clock::now()-start).count());
            auto a=raw,b=raw;
            start=Clock::now();ordinary.apply(a,rgb,w,h,.6f);
            double ordinary_ms=std::chrono::duration<double,std::milli>(Clock::now()-start).count();
            start=Clock::now();changed+=flow.apply(b,rgb,matte.canvas_rgb(),w,h,w,h,t+k*33ms,.6f);
            double flow_ms=std::chrono::duration<double,std::milli>(Clock::now()-start).count();
            if(!flow.ready())return 1;
            if(k>=4){mr.sample(raw,ref,w,h,shift,0);mo.sample(a,ref,w,h,shift,ordinary_ms);mf.sample(b,ref,w,h,shift,flow_ms);}
        }
        std::sort(base_ms.begin()+4,base_ms.end());
        mr.print("raw",sequence);mo.print("ordinary",sequence);mf.print("flow",sequence);
        std::cout<<"RUN "<<sequence<<" matte_p50_ms="<<base_ms[14]<<" blended="<<changed<<" graph_runs="<<flow.runs()<<" "<<flow.status()<<"\n";
        if(!changed)return 1;
        if(std::string(sequence)=="relight"&&flow.runs()!=0)return 1;
        if(std::string(sequence)=="translate"&&flow.runs()==0)return 1;
        // Abrupt replacement and capture gap: no old matte should survive
        // a known gap >500ms even if scene appearance would pass a gate.
        std::fill(rgb.begin(),rgb.end(),80);
        if(!matte.process(rgb.data(),w,h,raw.data()))return 1;
        auto cut=raw;flow.apply(cut,rgb,matte.canvas_rgb(),w,h,w,h,t+2s,.6f);
        if(cut!=raw)return 1;
    }
    std::cout<<"SIGHT_FLOW_SEQUENCE_20261126_W812 PASS\n";
}
