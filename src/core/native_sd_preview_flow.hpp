#pragma once
#include "core/sight_flow_v1.h"
#include <array>
#include <vector>
#include <cstdint>
#ifdef SIGHT_HAVE_DENSE_FLOW
#include "core/dense_motion.hpp"
#include <opencv2/imgproc.hpp>
#endif
namespace sight::sd {
struct PreviewFlowResult {
    // Confidence-premultiplied UV displacement, confidence, unused alpha.
    std::array<float,SIGHT_FLOW_CELLS*4> grid{};
    float coverage=0;
};
class PreviewFlow {
#ifdef SIGHT_HAVE_DENSE_FLOW
    sight::DenseMotion engine_;
#endif
public:
    PreviewFlowResult estimate(const std::vector<uint8_t>&init,const std::vector<uint8_t>&next,int w,int h){
        PreviewFlowResult result;
#ifdef SIGHT_HAVE_DENSE_FLOW
        if(w<32||h<32||w>1024||h>1024||init.size()!=size_t(w)*h*3||next.size()!=size_t(w)*h*4)return result;
        try{
            const double scale=std::min(1.,320./std::max(w,h));
            cv::Size size(std::max(1,int(w*scale)),std::max(1,int(h*scale)));
            if(size.width<32||size.height<32)return result;
            cv::Mat a,b,small;
            cv::resize(cv::Mat(h,w,CV_8UC3,const_cast<uint8_t*>(init.data())),small,size,0,0,cv::INTER_AREA);
            cv::cvtColor(small,a,cv::COLOR_RGB2GRAY);
            cv::resize(cv::Mat(h,w,CV_8UC4,const_cast<uint8_t*>(next.data())),small,size,0,0,cv::INTER_AREA);
            cv::cvtColor(small,b,cv::COLOR_RGBA2GRAY);
            sight_flow_v1 flow{};
            if(!engine_.estimate(a,b,flow))return result;
            result.coverage=flow.coverage;
            for(size_t i=0;i<SIGHT_FLOW_CELLS;++i){
                const auto&c=flow.cells[i];
                result.grid[i*4]=c.dx*c.confidence;
                result.grid[i*4+1]=c.dy*c.confidence;
                result.grid[i*4+2]=c.confidence;
            }
        }catch(const cv::Exception&){return {};}
#else
        (void)init;(void)next;(void)w;(void)h;
#endif
        return result;
    }
};
}
