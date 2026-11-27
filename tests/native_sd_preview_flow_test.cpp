#include "core/native_sd_preview_flow.hpp"
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <chrono>
int main(){
    constexpr int W=320,H=192;
    cv::Mat a(H,W,CV_8UC1),b;
    cv::RNG rng(42);rng.fill(a,cv::RNG::UNIFORM,0,256);cv::GaussianBlur(a,a,{5,5},1);
    cv::Mat m=(cv::Mat_<double>(2,3)<<1,0,6,0,1,-4);cv::warpAffine(a,b,m,a.size(),cv::INTER_LINEAR,cv::BORDER_REFLECT);
    cv::Mat rgb,rgba;cv::cvtColor(a,rgb,cv::COLOR_GRAY2RGB);cv::cvtColor(b,rgba,cv::COLOR_GRAY2RGBA);
    std::vector<uint8_t> init(rgb.data,rgb.data+rgb.total()*3),next(rgba.data,rgba.data+rgba.total()*4);
    sight::sd::PreviewFlow estimator;
    auto start=std::chrono::steady_clock::now();
    auto flow=estimator.estimate(init,next,W,H);
    double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count(),error=0;
    unsigned n=0;
    for(unsigned y=2;y<16;++y)for(unsigned x=2;x<30;++x){
        size_t i=(y*32+x)*4;float c=flow.grid[i+2];
        if(c>.5f){error+=std::hypot(flow.grid[i]*W/c+6,flow.grid[i+1]*H/c-4);++n;}
    }
    bool translated=flow.coverage>.5f&&n>300&&error/std::max(n,1u)<.5;
    std::fill(next.begin(),next.end(),90);
    auto cut=estimator.estimate(init,next,W,H);
    auto bad=estimator.estimate({},next,W,H);
    bool rejected=cut.coverage==0&&bad.coverage==0;
    std::cout<<"coverage="<<flow.coverage<<" confident="<<n<<" endpoint_error="<<error/std::max(n,1u)<<" ms="<<ms<<" rejected="<<rejected<<"\n";
    return translated&&rejected?0:1;
}
