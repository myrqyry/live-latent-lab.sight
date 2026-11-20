#pragma once
// Reidentifiable-by-motion boxes for a single local Sight object-detection
// filter. Identity is only meaningful within this filter instance, not across
// cameras or occlusion longer than the bounded retention window.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>
namespace sight {
struct ObjectDetection {
    float x=0,y=0,w=0,h=0,score=0;
    std::string label;
    int category=-1;
    int id=0;
};
inline bool valid_object(const ObjectDetection& p) {
    return std::isfinite(p.x)&&std::isfinite(p.y)&&
           std::isfinite(p.w)&&std::isfinite(p.h)&&
           std::isfinite(p.score) && p.score>=.25f && p.score<=1.f &&
           p.w>.001f&&p.h>.001f&&p.w<=1.f&&p.h<=1.f &&
           p.x>=0.f&&p.y>=0.f&&p.x+p.w<=1.001f&&p.y+p.h<=1.001f &&
           !p.label.empty()&&p.label.size()<=96;
}
class ObjectTracks {
    struct Track {
        ObjectDetection value;
        float vx=0,vy=0;
        unsigned seen=0,missed=0;
        bool matched=false;
    };
    std::vector<Track> tracks_;
    int next_id_=1;
    static float distance(float ax,float ay,float bx,float by) {
        return std::hypot(ax-bx,ay-by);
    }
    static float overlap(const ObjectDetection& a,const ObjectDetection& b) {
        float x=std::max(0.f,std::min(a.x+a.w,b.x+b.w)-std::max(a.x,b.x));
        float y=std::max(0.f,std::min(a.y+a.h,b.y+b.h)-std::max(a.y,b.y));
        float intersection=x*y;
        float total=a.w*a.h+b.w*b.h-intersection;
        return total>0?intersection/total:0.f;
    }
public:
    void reset(){tracks_.clear();next_id_=1;}
    std::vector<ObjectDetection> step(std::vector<ObjectDetection> detections) {
        detections.erase(std::remove_if(detections.begin(),detections.end(),
                                        [](const auto& d){return !valid_object(d);}),
                         detections.end());
        if(detections.size()>16) {
            std::stable_sort(detections.begin(),detections.end(),
                [](const auto& a,const auto& b){return a.score>b.score;});
            detections.resize(16);
        }
        for(auto& t:tracks_)t.matched=false;
        struct Pair {size_t t,d;float cost;};
        std::vector<Pair> possible;
        for(size_t ti=0;ti<tracks_.size();++ti) {
            const auto& track=tracks_[ti];
            const auto& a=track.value;
            for(size_t di=0;di<detections.size();++di) {
                const auto& b=detections[di];
                if(a.label!=b.label)continue;
                // Predict the box center with bounded velocity so crossing
                // same-class objects do not swap merely due to array order.
                const float speed=std::min(2.f,float(track.missed+1));
                const float ax=a.x+a.w*.5f+track.vx*speed;
                const float ay=a.y+a.h*.5f+track.vy*speed;
                const float bx=b.x+b.w*.5f,by=b.y+b.h*.5f;
                const float dist=distance(ax,ay,bx,by);
                if(dist>.26f)continue;
                const float iou=overlap(a,b);
                possible.push_back({ti,di,dist-.06f*iou});
            }
        }
        std::sort(possible.begin(),possible.end(),[](const Pair& a,const Pair& b){
            if(a.cost!=b.cost)return a.cost<b.cost;
            if(a.t!=b.t)return a.t<b.t;
            return a.d<b.d;
        });
        std::vector<bool> used(detections.size(),false);
        for(const auto& p:possible) {
            if(tracks_[p.t].matched||used[p.d])continue;
            auto& track=tracks_[p.t];
            auto d=std::move(detections[p.d]);
            const float old_x=track.value.x+track.value.w*.5f;
            const float old_y=track.value.y+track.value.h*.5f;
            track.vx=std::clamp((d.x+d.w*.5f-old_x),-.18f,.18f);
            track.vy=std::clamp((d.y+d.h*.5f-old_y),-.18f,.18f);
            d.id=track.value.id;
            track.value=std::move(d);
            track.seen=std::min(track.seen+1,10000u);
            track.missed=0;track.matched=true;used[p.d]=true;
        }
        for(auto& track:tracks_)if(!track.matched)++track.missed;
        tracks_.erase(std::remove_if(tracks_.begin(),tracks_.end(),
                     [](const auto& t){return t.missed>4;}),tracks_.end());
        for(size_t i=0;i<detections.size();++i)if(!used[i]){
            auto d=std::move(detections[i]);
            d.id=next_id_++;
            tracks_.push_back(Track{std::move(d),0,0,1,0,true});
        }
        std::vector<ObjectDetection> visible;
        for(const auto& track:tracks_)
            if(track.matched&&track.seen>=2)
                visible.push_back(track.value);
        std::sort(visible.begin(),visible.end(),
                  [](const auto& a,const auto& b){return a.id<b.id;});
        return visible;
    }
};
}
