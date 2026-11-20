#include "core/dense_motion.hpp"
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <cmath>
#include <chrono>
int main(){
    cv::Mat a(179,319,CV_8U),b,cut;
    cv::RNG rng(42);rng.fill(a,cv::RNG::UNIFORM,0,256);cv::GaussianBlur(a,a,{5,5},1);
    cv::Mat M=(cv::Mat_<double>(2,3)<<1,0,5,0,1,-3);
    cv::warpAffine(a,b,M,a.size(),cv::INTER_LINEAR,cv::BORDER_REFLECT);
    sight::DenseMotion flow;sight_flow_v1 out{};
    bool moved=flow.estimate(a,b,out);double err=0;unsigned n=0;
    for(unsigned y=2;y<16;++y)for(unsigned x=2;x<30;++x){
        auto c=out.cells[y*32+x];if(c.confidence>.5){err+=std::hypot(c.dx*a.cols+5,c.dy*a.rows-3);++n;}
    }
    moved=moved&&n>300&&err/std::max(1u,n)<.5;
    std::cout<<"translation n="<<n<<" endpoint_error="<<err/std::max(1u,n)<<" coverage="<<out.coverage<<"\n";
    cv::Mat local=a.clone();
    // Separate regions shift left and right; ignore the seam and borders.
    for(int half=0;half<2;++half){
        cv::Rect region(half?160:0,0,half?159:160,179);
        cv::Mat translated;
        cv::Mat transform=(cv::Mat_<double>(2,3)<<1,0,half?-5:5,0,1,0);
        cv::warpAffine(a(region),translated,transform,region.size(),cv::INTER_LINEAR,cv::BORDER_REFLECT);
        translated.copyTo(local(region));
    }
    bool spatial=flow.estimate(a,local,out);
    double left=0,right=0;unsigned nl=0,nr=0;
    for(unsigned y=2;y<16;++y)for(unsigned x=2;x<30;++x){
        const auto c=out.cells[y*32+x];if(c.confidence<.5)continue;
        if(x<13){left+=c.dx*a.cols;++nl;}
        if(x>18){right+=c.dx*a.cols;++nr;}
    }
    spatial=spatial&&nl>100&&nr>100&&std::abs(left/std::max(1u,nl)+5)<.5&&std::abs(right/std::max(1u,nr)-5)<.5;
    std::cout<<"local_backward_dx left="<<left/std::max(1u,nl)<<" right="<<right/std::max(1u,nr)<<"\n";
    auto begin=std::chrono::steady_clock::now();
    for(int i=0;i<20;++i)flow.estimate(a,b,out);
    std::cout<<"bidirectional_ms="<<std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count()/20<<"\n";
    bool still=flow.estimate(a,a,out);
    for(auto c:out.cells)still&=std::hypot(c.dx,c.dy)<.001;
    cv::Mat flat(a.size(),CV_8U,cv::Scalar(90));
    bool blank=!flow.estimate(flat,flat,out);
    rng.fill(cut=cv::Mat(a.size(),CV_8U),cv::RNG::UNIFORM,0,256);
    bool scene_cut=!flow.estimate(a,cut,out);
    bool bad=!flow.estimate(a,cv::Mat(),out)&&!flow.estimate(a,cv::Mat(20,20,CV_8U),out);
    std::cout<<"DENSE_MOTION translate="<<moved<<" spatial="<<spatial<<" still="<<still<<" flat="<<blank<<" cut="<<scene_cut<<" invalid="<<bad<<"\n";
    return !(moved&&spatial&&still&&blank&&scene_cut&&bad);
}
