#include "dense_motion.hpp"
#include <algorithm>
#include <cmath>
namespace sight {
bool DenseMotion::estimate(const cv::Mat& previous,const cv::Mat& current,sight_flow_v1& out) {
    out={};
    if(previous.empty()||current.empty()||previous.size()!=current.size()||
       previous.type()!=CV_8UC1||current.type()!=CV_8UC1||
       current.cols<32||current.rows<32||current.cols>640||current.rows>640)return false;
    try {
        if(!engine_)engine_=cv::DISOpticalFlow::create(cv::DISOpticalFlow::PRESET_MEDIUM);
        cv::Mat backward,forward;
        engine_->calc(current,previous,backward);
        engine_->calc(previous,current,forward);
        const int w=current.cols,h=current.rows;
        unsigned accepted=0;
        for(unsigned y=0;y<SIGHT_FLOW_HEIGHT;++y)for(unsigned x=0;x<SIGHT_FLOW_WIDTH;++x) {
            const int cx=int((x+.5)*w/SIGHT_FLOW_WIDTH),cy=int((y+.5)*h/SIGHT_FLOW_HEIGHT);
            const auto v=backward.at<cv::Vec2f>(cy,cx);
            if(!std::isfinite(v[0])||!std::isfinite(v[1])||std::abs(v[0])>w/4.f||std::abs(v[1])>h/4.f)continue;
            const int px=int(std::lround(cx+v[0])),py=int(std::lround(cy+v[1]));
            if(px<3||py<3||px>=w-3||py>=h-3||cx<3||cy<3||cx>=w-3||cy>=h-3)continue;
            const auto f=forward.at<cv::Vec2f>(py,px);
            const float consistency=std::hypot(v[0]+f[0],v[1]+f[1]);
            if(!std::isfinite(consistency)||consistency>1.5f)continue;
            int lo=255,hi=0,residual=0;
            for(int j=-2;j<=2;++j)for(int i=-2;i<=2;++i){
                int a=current.at<uint8_t>(cy+j,cx+i),b=previous.at<uint8_t>(py+j,px+i);
                lo=std::min(lo,a);hi=std::max(hi,a);residual+=std::abs(a-b);
            }
            const float mean=residual/25.f;
            if(hi-lo<14||mean>18.f)continue;
            auto& cell=out.cells[y*SIGHT_FLOW_WIDTH+x];
            cell={v[0]/w,v[1]/h,std::clamp(1.f-std::max(consistency/3.f,mean/36.f),0.f,1.f),0};
            ++accepted;
        }
        out.coverage=float(accepted)/SIGHT_FLOW_CELLS;
        if(out.coverage<.2f){out={};return false;}
        return true;
    } catch(const cv::Exception&) {out={};engine_.release();return false;}
}
}
