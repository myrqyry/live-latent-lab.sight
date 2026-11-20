#pragma once
#include "tasks_tracker.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

namespace sight {
enum class FaceQualityMode { Legacy=0, Adaptive=1, Precision=2 };
struct FaceImage {const uint8_t* rgba=nullptr; unsigned width=0,height=0; size_t bytes=0;};

namespace face_detail {
struct XY {
    float x=0,y=0;
    XY operator+(XY b)const{return {x+b.x,y+b.y};}
    XY operator-(XY b)const{return {x-b.x,y-b.y};}
    XY operator*(float n)const{return {x*n,y*n};}
};
inline float dot(XY a,XY b){return a.x*b.x+a.y*b.y;}
inline float length(XY a){return std::hypot(a.x,a.y);}
inline XY pixels(const TrackedPoint& p,FaceImage im){return {p.x*im.width,p.y*im.height};}
inline bool finite(const TrackedPoint& p){
    return std::isfinite(p.x)&&std::isfinite(p.y)&&std::isfinite(p.z)&&std::isfinite(p.confidence);
}
inline bool valid_image(FaceImage im){
    return im.rgba&&im.width>=16&&im.height>=16&&im.width<=8192&&im.height<=8192&&
           im.bytes==size_t(im.width)*im.height*4;
}
inline float luma(FaceImage im, XY at){
    if(!std::isfinite(at.x)||!std::isfinite(at.y)||at.x<0||at.y<0||
       at.x>=float(im.width-1)||at.y>=float(im.height-1))return -1;
    unsigned x=unsigned(at.x),y=unsigned(at.y);
    const float fx=at.x-x,fy=at.y-y;
    auto value=[&](unsigned a,unsigned b){
        const auto* p=im.rgba+(size_t(b)*im.width+a)*4;
        return p[3]>=200?(.2126f*p[0]+.7152f*p[1]+.0722f*p[2]):-1.f;
    };
    float a=value(x,y),b=value(x+1,y),c=value(x,y+1),d=value(x+1,y+1);
    if(std::min({a,b,c,d})<0)return -1;
    return (a+(b-a)*fx)*(1-fy)+(c+(d-c)*fx)*fy;
}
// Bounded local correction only; never move eye corners, iris, lower lids,
// expressions or the head matrix. Weak contrast/occlusion leaves points raw.
inline void refine_lids(std::vector<TrackedPoint>& p,FaceImage im){
    if(!valid_image(im))return;
    constexpr int lids[2][9]={{133,173,157,158,159,160,161,246,33},
                              {362,398,384,385,386,387,388,466,263}};
    constexpr int lower[2]={145,374};
    const XY up=pixels(p[10],im)-pixels(p[152],im);
    float widths[2]{};
    for(int eye=0;eye<2;++eye)
        widths[eye]=length(pixels(p[lids[eye][8]],im)-pixels(p[lids[eye][0]],im));
    for(int eye=0;eye<2;++eye){
        const float width=widths[eye];
        if(width<14||width<.68f*widths[1-eye])continue;
        const XY along=(pixels(p[lids[eye][8]],im)-pixels(p[lids[eye][0]],im))*(1.f/width);
        XY normal{along.y,-along.x};
        if(dot(normal,up)<0)normal=normal*(-1);
        const float aperture=dot(pixels(p[lids[eye][4]],im)-pixels(p[lower[eye]],im),normal);
        if(aperture<.14f*width)continue;
        const float radius=std::min({5.f,width*.045f,aperture*.20f});
        if(radius<.75f)continue;
        std::array<float,7> offsets{};
        std::array<bool,7> usable{};
        for(size_t i=0;i<offsets.size();++i){
            const XY center=pixels(p[lids[eye][i+1]],im);
            float best=1e9f,low=256,high=-1;
            for(int step=-8;step<=8;++step){
                const float offset=radius*step/8.f;
                float mean=0;
                bool valid=true;
                for(int k=-1;k<=1;++k){
                    float v=luma(im,center+normal*offset+along*(width*.025f*k));
                    if(v<0){valid=false;break;}
                    mean+=v/3;
                }
                if(!valid)continue;
                low=std::min(low,mean);high=std::max(high,mean);
                float penalty=40.f*offset*offset/(radius*radius);
                if(mean+penalty<best){best=mean+penalty;offsets[i]=offset;}
            }
            usable[i]=high-low>=22.f;
            if(!usable[i])offsets[i]=0;
        }
        if(std::count(usable.begin(),usable.end(),true)<4)continue;
        auto settled=offsets;
        for(size_t i=1;i+1<offsets.size();++i){
            std::array<float,3> near{offsets[i-1],offsets[i],offsets[i+1]};
            std::sort(near.begin(),near.end());settled[i]=near[1];
        }
        for(size_t i=0;i<settled.size();++i){
            if(!usable[i])continue;
            auto& pt=p[lids[eye][i+1]];
            const float taper=(i==0||i+1==settled.size())?.65f:1.f;
            XY delta=normal*(settled[i]*taper);
            pt.x+=delta.x/im.width;pt.y+=delta.y/im.height;
        }
    }
}
inline float alpha(float cutoff,double dt){
    const double omega=6.283185307179586*cutoff*dt;
    return float(omega/(1+omega));
}
}
class FaceQuality {
    struct Axis {float raw=0,value=0,velocity=0;};
    std::vector<std::array<Axis,3>> state_;
    double last_=0;
    unsigned w_=0,h_=0;
    FaceQualityMode mode_=FaceQualityMode::Legacy;
    float strength_=0,scale_=0;
    face_detail::XY center_{};
public:
    void reset(){state_.clear();last_=0;w_=h_=0;scale_=0;}
    bool apply(TrackingData& d,FaceImage im,double time,FaceQualityMode mode,float strength){
        using namespace face_detail;
        auto& points=d.landmarks;
        if((points.size()!=468&&points.size()!=478)||
            !std::all_of(points.begin(),points.end(),finite)||
            !std::isfinite(time)||!im.width||!im.height){
            d={};reset();return false;
        }
        strength=std::isfinite(strength)?std::clamp(strength,0.f,.95f):0;
        if(mode==FaceQualityMode::Legacy){reset();return true;}
        if(mode==FaceQualityMode::Precision)refine_lids(points,im);
        const float scale=length(pixels(points[10],im)-pixels(points[152],im));
        const XY center=(pixels(points[10],im)+pixels(points[152],im))*.5f;
        if(!std::isfinite(scale)||scale<4){reset();return true;}
        const double dt=time-last_;
        if(state_.size()!=points.size()||w_!=im.width||h_!=im.height||
            mode_!=mode||strength_!=strength||dt<=0||dt>.5||
            scale<scale_*.5f||scale>scale_*2.f||length(center-center_)>scale*.65f||strength==0){
            state_.resize(points.size());
            for(size_t i=0;i<points.size();++i){
                const auto& p=points[i];state_[i]={Axis{p.x,p.x,0},Axis{p.y,p.y,0},Axis{p.z,p.z,0}};
            }
        }else{
            // One Euro filtering, normalized to apparent face size and frame aspect ratio.
            const float derivative=alpha(4.f,dt);
            const float minimum=3.5f-3.0f*strength;
            const float beta=30.f;
            for(size_t i=0;i<points.size();++i){
                auto& p=points[i];
                float* axes[]={&p.x,&p.y,&p.z};
                const float factors[]={float(im.width)/scale,float(im.height)/scale,float(im.width)/scale};
                float speed2=0;
                for(int j=0;j<3;++j){
                    auto& a=state_[i][j];
                    const float speed=std::clamp(float((*axes[j]-a.raw)*factors[j]/dt),-100.f,100.f);
                    a.velocity+=derivative*(speed-a.velocity);
                    a.raw=*axes[j];speed2+=a.velocity*a.velocity;
                }
                const float blend=alpha(minimum+beta*std::sqrt(speed2),dt);
                for(int j=0;j<3;++j){
                    auto& a=state_[i][j];a.value+=blend*(a.raw-a.value);*axes[j]=a.value;
                }
            }
        }
        w_=im.width;h_=im.height;last_=time;mode_=mode;strength_=strength;scale_=scale;center_=center;
        return true;
    }
};
}
