#include "core/neural_render_motion.hpp"
#include <cmath>
#include <iostream>

int main(){
    constexpr int w=320,h=180;
    cv::Mat rgb(h,w,CV_8UC3);
    cv::RNG rng(327);
    rng.fill(rgb,cv::RNG::UNIFORM,0,256);
    cv::GaussianBlur(rgb,rgb,{5,5},1);
    std::vector<uint8_t> bytes(rgb.data,rgb.data+rgb.total()*3);
    auto ref=sight::nr::make_reference(71,bytes,w,h);
    cv::Mat moved,rgba;
    cv::Mat transform=(cv::Mat_<double>(2,3)<<1,0,8,0,1,-4);
    cv::warpAffine(rgb,moved,transform,rgb.size(),cv::INTER_LINEAR,cv::BORDER_REFLECT);
    cv::cvtColor(moved,rgba,cv::COLOR_RGB2RGBA);
    sight::nr::MotionSync sync;
    auto wait=[&](uint64_t seq,sight::nr::MotionResult&out){
        for(int i=0;i<100;++i){
            if(sync.latest(seq,out))return true;
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        return false;
    };
    // Simulate a higher resolution capture than the bounded flow proxy.
    cv::Mat large;
    cv::resize(rgba,large,{w*2,h*2},0,0,cv::INTER_NEAREST);
    sync.submit(ref,large,sight::nr::SyncClock::now());
    sight::nr::MotionResult out;
    if(!wait(71,out)){std::cerr<<"motion result unavailable\n";return 1;}
    unsigned count=0;double err=0.;
    for(unsigned y=2;y<16;++y)for(unsigned x=2;x<30;++x){
        const auto c=out.flow.cells[y*SIGHT_FLOW_WIDTH+x];
        if(c.confidence>.5){
            ++count;err+=std::hypot(c.dx*w+8,c.dy*h-4);
        }
    }
    const bool translation=count>250&&err/std::max(1u,count)<1.;
    const bool wrong_ref_rejected=!sync.latest(72,out);
    rng.fill(rgb,cv::RNG::UNIFORM,0,256);
    cv::cvtColor(rgb,rgba,cv::COLOR_RGB2RGBA);
    auto ref2=sight::nr::make_reference(72,bytes,w,h);
    sync.submit(ref2,rgba,sight::nr::SyncClock::now());
    const bool cut=wait(72,out)&&out.flow.coverage==0.f;
    std::this_thread::sleep_for(std::chrono::milliseconds(160));
    const bool stale=!sync.latest(72,out);
    std::cout<<"translation="<<translation<<" confident_cells="<<count
             <<" endpoint_error_px="<<err/std::max(1u,count)
             <<" cut_rejected="<<cut<<" wrong_reference_rejected="<<wrong_ref_rejected
             <<" stale_rejected="<<stale<<"\n";
    return translation&&cut&&wrong_ref_rejected&&stale?0:1;
}
