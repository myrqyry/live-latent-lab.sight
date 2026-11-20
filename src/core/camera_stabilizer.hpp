#pragma once
#include "sight_flow_v1.h"
#include <algorithm>
#include <array>
#include <cmath>
namespace sight {
struct CameraCorrection {float x=0,y=0,consensus=0;bool valid=false;};
class CameraStabilizer {
    float x_=0,y_=0;
public:
    void reset() {x_=y_=0;}
    CameraCorrection step(const sight_flow_v1& flow,unsigned w,unsigned h,float tau,float margin) {
        const float dt=flow.interval_seconds;
        if(w<32||h<32||w>640||h>640||!std::isfinite(dt)||dt<=0||dt>.75f||
           !std::isfinite(tau)||tau<=0||!std::isfinite(margin)||margin<0||margin>.2f) {
            reset();return {};
        }
        std::array<float,SIGHT_FLOW_CELLS> dx{},dy{};
        unsigned n=0;
        auto usable=[](const sight_flow_cell_v1& v) {
            return std::isfinite(v.confidence)&&v.confidence>=.5f&&v.confidence<=1.f&&
                std::isfinite(v.dx)&&std::isfinite(v.dy)&&std::abs(v.dx)<=.25f&&std::abs(v.dy)<=.25f;
        };
        for(auto v:flow.cells)if(usable(v)){dx[n]=-v.dx;dy[n++]=-v.dy;}
        if(n<SIGHT_FLOW_CELLS*.35f){reset();return {};}
        std::nth_element(dx.begin(),dx.begin()+n/2,dx.begin()+n);
        std::nth_element(dy.begin(),dy.begin()+n/2,dy.begin()+n);
        const float mx=dx[n/2],my=dy[n/2];
        float sx=0,sy=0,weight=0;
        unsigned inliers=0;
        std::array<unsigned,4> quadrants{};
        for(unsigned y=0;y<SIGHT_FLOW_HEIGHT;++y)for(unsigned x=0;x<SIGHT_FLOW_WIDTH;++x) {
            const auto v=flow.cells[y*SIGHT_FLOW_WIDTH+x];
            if(!usable(v)||std::hypot((-v.dx-mx)*w,(-v.dy-my)*h)>1.5f)continue;
            ++inliers;++quadrants[(y>=SIGHT_FLOW_HEIGHT/2)*2+(x>=SIGHT_FLOW_WIDTH/2)];
            sx-=v.dx*v.confidence;sy-=v.dy*v.confidence;weight+=v.confidence;
        }
        const float consensus=float(inliers)/n;
        if(consensus<.7f||inliers<SIGHT_FLOW_CELLS*.35f||
           *std::min_element(quadrants.begin(),quadrants.end())<SIGHT_FLOW_CELLS/40) {
            reset();return {};
        }
        float fx=sx/weight,fy=sy/weight;
        if(std::hypot(fx*w,fy*h)<.05f)fx=fy=0;
        // Residual of causal exponential trajectory smoothing. No unbounded
        // integrated position and no replay of a capture's displacement.
        const float limit=margin*.95f;
        const float pressure=limit>0?std::clamp(std::max(std::abs(x_+fx),std::abs(y_+fy))/limit,0.f,1.f):1.f;
        const float effective_tau=tau*(1.f-.75f*std::max(0.f,(pressure-.5f)*2.f));
        const float retain=std::exp(-dt/effective_tau);
        x_=std::clamp((x_+fx)*retain,-limit,limit);
        y_=std::clamp((y_+fy)*retain,-limit,limit);
        return {x_,y_,consensus,true};
    }
};
}
