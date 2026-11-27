#include "core/native_sd_temporal.hpp"
#include <iostream>
#include <cmath>
int main(){
    using namespace sight::sd;
    constexpr int w=256,h=144;
    cv::Mat rgba(h,w,CV_8UC4);
    cv::RNG rng(71);
    rng.fill(rgba,cv::RNG::UNIFORM,20,210);
    cv::GaussianBlur(rgba,rgba,{5,5},1);
    for(int y=0;y<h;++y)for(int x=0;x<w;++x)rgba.at<cv::Vec4b>(y,x)[3]=255;
    cv::Mat moved;
    cv::Mat transform=(cv::Mat_<double>(2,3)<<1,0,8,0,1,-4);
    cv::warpAffine(rgba,moved,transform,rgba.size(),cv::INTER_LINEAR,cv::BORDER_REPLICATE);
    std::vector<float> old(w*h*4),fresh(w*h*4,0.f);
    for(int y=0;y<h;++y)for(int x=0;x<w;++x)
        for(int c=0;c<3;++c)old[(y*w+x)*4+c]=float(rgba.at<cv::Vec4b>(y,x)[c])/255.f-.5f;
    TemporalHistory history;
    auto now=TemporalHistory::Clock::now();
    history.apply(1,rgba,old,.5f,now);
    auto unchanged=fresh;
    auto static_stats=history.apply(1,rgba,unchanged,.5f,now+std::chrono::milliseconds(100));
    bool static_ok=static_stats.coverage>.99f;
    for(size_t i=0;i<old.size();++i)static_ok&=std::abs(unchanged[i]-.5f*old[i])<1e-6f;
    auto third=fresh;
    history.apply(1,rgba,third,.5f,now+std::chrono::milliseconds(200));
    bool nonrecursive=third==fresh;
    history.reset();history.apply(1,rgba,old,.5f,now);
    auto occluded=rgba.clone();
    const cv::Rect region(96,48,64,48);
    occluded(region).setTo(cv::Scalar(255,255,255,255));
    auto occlusion_delta=fresh;
    auto occlusion_stats=history.apply(1,occluded,occlusion_delta,.5f,now+std::chrono::milliseconds(100));
    bool occlusion_ok=occlusion_stats.coverage>.5f;
    for(int y=region.y;y<region.y+region.height;++y)
        for(int x=region.x;x<region.x+region.width;++x)
            for(int c=0;c<3;++c)occlusion_ok&=occlusion_delta[(y*w+x)*4+c]==0.f;
    history.reset();history.apply(1,rgba,old,.5f,now);
    auto translated=fresh;
    auto stats=history.apply(1,moved,translated,.5f,now+std::chrono::milliseconds(100));
    double warped=0,unwarped=0;
    for(int y=16;y<h-16;++y)for(int x=16;x<w-16;++x)for(int c=0;c<3;++c){
        auto i=(y*w+x)*4+c;
        warped+=std::abs(translated[i]-.5f*old[((y+4)*w+x-8)*4+c]);
        unwarped+=std::abs(translated[i]-.5f*old[i]);
    }
    cv::Mat cut(h,w,CV_8UC4,cv::Scalar(255,255,255,255));
    auto cut_delta=fresh;
    auto cut_stats=history.apply(1,cut,cut_delta,.5f,now+std::chrono::milliseconds(200));
    bool cut_ok=cut_delta==fresh&&cut_stats.coverage==0;
    history.reset();history.apply(1,rgba,old,.5f,now);
    auto other=fresh;
    history.apply(2,rgba,other,.5f,now+std::chrono::milliseconds(100));
    bool reset_ok=other==fresh;
    history.reset();history.apply(1,rgba,old,.5f,now);
    history.apply(1,rgba,other,.5f,now+std::chrono::milliseconds(501));
    bool expiry=other==fresh;
    history.reset();history.apply(1,rgba,old,.5f,now);
    history.apply(1,rgba,other,0.f,now+std::chrono::milliseconds(100));
    bool neutral=other==fresh;
    std::cout<<"static="<<static_ok<<" coverage="<<stats.coverage<<" warped_error="<<warped
        <<" unwarped_error="<<unwarped<<" nonrecursive="<<nonrecursive<<" occlusion="<<occlusion_ok<<" cut="<<cut_ok<<" reset="<<reset_ok<<" expiry="<<expiry<<" neutral="<<neutral<<"\n";
    // Neural renderer supports a 640px proxy; diffusion keeps its 512px cap.
    cv::Mat wide(360,640,CV_8UC4,cv::Scalar(110,120,130,255));
    std::vector<float> wide_old(wide.total()*4,.001f),wide_new(wide.total()*4,0.f);
    for(size_t i=3;i<wide_old.size();i+=4)wide_old[i]=0.f;
    TemporalHistory neural(640);
    neural.apply(1,wide,wide_old,.5f,now);
    auto wide_stats=neural.apply(1,wide,wide_new,.5f,now+std::chrono::milliseconds(33));
    bool wide_ok=wide_stats.coverage==1.f&&std::abs(wide_new[0]-.0005f)<1e-8f;
    history.reset();
    history.apply(1,wide,wide_old,.5f,now);
    auto capped=history.apply(1,wide,wide_new,.5f,now+std::chrono::milliseconds(33));
    wide_ok &= capped.coverage==0.f;
    neural.reset();
    neural.apply(1,wide,wide_old,.5f,now);
    cv::Mat transparent=wide.clone();
    const cv::Rect clear_area(100,100,20,20);
    transparent(clear_area).setTo(cv::Scalar(0,0,0,0));
    auto clear_delta=std::vector<float>(wide.total()*4,0.f);
    auto clear_stats=neural.apply(1,transparent,clear_delta,.5f,now+std::chrono::milliseconds(33));
    bool alpha_ok=clear_stats.coverage>.5f;
    for(int y=100;y<120;++y)for(int x=100;x<120;++x)
        for(int c=0;c<3;++c)alpha_ok &= clear_delta[(y*640+x)*4+c]==0.f;
    wide_ok &= alpha_ok;
    std::cout<<"neural_640_fractional="<<wide_ok<<" transparent_rejection="<<alpha_ok<<"\n";
    return wide_ok&&static_ok&&stats.coverage>.5f&&warped<unwarped*.65&&cut_ok&&reset_ok&&expiry&&neutral&&nonrecursive&&occlusion_ok?0:1;
}
