#pragma once
#include "sight_flow_v1.h"
#ifdef SIGHT_HAVE_DENSE_FLOW
#include "dense_motion.hpp"
#endif
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <vector>

namespace sight::sd {
// A two-keyframe filter: always store the RAW newest correction. Never
// recursively accumulate synthetic pixels or carry history across settings.
struct TemporalStats {float coverage=0.f,mean_weight=0.f;};
class TemporalHistory {
public:
    using Clock=std::chrono::steady_clock;
    explicit TemporalHistory(int max_edge=512):max_edge_(std::clamp(max_edge,1,640)){}
    void reset(){previous_.release();raw_.clear();generation_=0;at_={};}
    TemporalStats apply(uint64_t generation,const cv::Mat& rgba,std::vector<float>& delta,
                        float strength,Clock::time_point captured){
        if(!std::isfinite(strength)||strength<=0.f||rgba.empty()||rgba.type()!=CV_8UC4||
           rgba.cols>max_edge_||rgba.rows>max_edge_||delta.size()!=rgba.total()*4){
            reset();return {};
        }
        strength=std::min(strength,.9f);
        TemporalStats stats;
        const auto age=captured-at_;
        const bool compatible=generation==generation_&&previous_.size()==rgba.size()&&
            raw_.size()==delta.size()&&age>=Clock::duration::zero()&&age<=std::chrono::milliseconds(500);
        if(compatible){
            cv::Mat old_delta(rgba.rows,rgba.cols,CV_32FC4,raw_.data());
            cv::Mat warped_ref,warped_delta,confidence;
#ifdef SIGHT_HAVE_DENSE_FLOW
            if(cv::norm(previous_,rgba,cv::NORM_INF)>0){
                cv::Mat a,b;
                cv::cvtColor(previous_,a,cv::COLOR_RGBA2GRAY);
                cv::cvtColor(rgba,b,cv::COLOR_RGBA2GRAY);
                const int edge=std::max(a.cols,a.rows);
                if(edge>320){
                    cv::Size size(std::max(1,a.cols*320/edge),std::max(1,a.rows*320/edge));
                    cv::resize(a,a,size,0,0,cv::INTER_AREA);cv::resize(b,b,size,0,0,cv::INTER_AREA);
                }
                sight_flow_v1 flow{};
                if(engine_.estimate(a,b,flow)){
                    cv::Mat grid(SIGHT_FLOW_HEIGHT,SIGHT_FLOW_WIDTH,CV_32FC3);
                    for(unsigned y=0;y<SIGHT_FLOW_HEIGHT;++y)for(unsigned x=0;x<SIGHT_FLOW_WIDTH;++x){
                        const auto& c=flow.cells[y*SIGHT_FLOW_WIDTH+x];
                        grid.at<cv::Vec3f>(y,x)={c.dx*c.confidence,c.dy*c.confidence,c.confidence};
                    }
                    cv::resize(grid,grid,rgba.size(),0,0,cv::INTER_LINEAR);
                    cv::Mat map(rgba.size(),CV_32FC2);
                    confidence.create(rgba.size(),CV_32FC1);
                    for(int y=0;y<rgba.rows;++y)for(int x=0;x<rgba.cols;++x){
                        auto c=grid.at<cv::Vec3f>(y,x);
                        float px=x+c[0]*rgba.cols/std::max(c[2],.001f);
                        float py=y+c[1]*rgba.rows/std::max(c[2],.001f);
                        map.at<cv::Vec2f>(y,x)={px,py};
                        confidence.at<float>(y,x)=px>=0&&py>=0&&px<=rgba.cols-1&&py<=rgba.rows-1?c[2]:0.f;
                    }
                    cv::remap(previous_,warped_ref,map,cv::Mat(),cv::INTER_LINEAR,cv::BORDER_REPLICATE);
                    cv::remap(old_delta,warped_delta,map,cv::Mat(),cv::INTER_LINEAR,cv::BORDER_REPLICATE);
                }
            }
#endif
            // Compute all weights before applying anything. Reject globally
            // unmatched frames rather than allowing a few coincidental matches.
            std::vector<float> weights(rgba.total());
            std::vector<uint8_t> use_warp(rgba.total());
            size_t accepted=0;double total=0.;
            auto match=[](const cv::Vec4b&a,const cv::Vec4b&b){
                int error=0;for(int c=0;c<4;++c)error=std::max(error,std::abs(int(a[c])-int(b[c])));
                float t=std::clamp((error/255.f-.035f)/.125f,0.f,1.f);
                return 1.f-t*t*(3.f-2.f*t);
            };
            for(int y=0;y<rgba.rows;++y)for(int x=0;x<rgba.cols;++x){
                size_t i=size_t(y)*rgba.cols+x;
                const auto current=rgba.at<cv::Vec4b>(y,x);
                float c0=match(current,previous_.at<cv::Vec4b>(y,x));
                float c1=warped_ref.empty()?0.f:match(current,warped_ref.at<cv::Vec4b>(y,x))*confidence.at<float>(y,x);
                // No historical color contribution to a newly transparent pixel.
                float c=current[3]?std::max(c0,c1):0.f;
                use_warp[i]=c1>c0;
                weights[i]=strength*c;
                if(c>.5f)++accepted;
                total+=weights[i];
            }
            if(accepted>=rgba.total()*.2){
                stats.coverage=float(accepted)/rgba.total();
                stats.mean_weight=float(total/rgba.total());
                // Keep raw current result for the next keyframe, not its
                // blended result. Old appearance cannot persist indefinitely.
                auto raw_next=delta;
                for(int y=0;y<rgba.rows;++y)for(int x=0;x<rgba.cols;++x){
                    size_t i=size_t(y)*rgba.cols+x;
                    const auto old=use_warp[i]?warped_delta.at<cv::Vec4f>(y,x):old_delta.at<cv::Vec4f>(y,x);
                    for(int c=0;c<3;++c)delta[i*4+c]+=weights[i]*(old[c]-delta[i*4+c]);
                }
                raw_=std::move(raw_next);
            }else raw_=delta;
        }else raw_=delta;
        previous_=rgba.clone();generation_=generation;at_=captured;
        return stats;
    }
private:
    const int max_edge_;
    cv::Mat previous_;
    std::vector<float> raw_;
    uint64_t generation_=0;
    Clock::time_point at_{};
#ifdef SIGHT_HAVE_DENSE_FLOW
    sight::DenseMotion engine_;
#endif
};
}
