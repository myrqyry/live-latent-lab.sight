#pragma once
#include "semantic_regions.hpp"
namespace sight::semantic {
inline float overlap(const privacy::Rect& a,const privacy::Rect& b) {
 const float common=std::max(0.f,std::min(a.x1,b.x1)-std::max(a.x0,b.x0))*
                    std::max(0.f,std::min(a.y1,b.y1)-std::max(a.y0,b.y0));
 const float area=(a.x1-a.x0)*(a.y1-a.y0)+(b.x1-b.x0)*(b.y1-b.y0)-common;
 return area>0?common/area:0;
}
inline privacy::Rect unite(const privacy::Rect& a,const privacy::Rect& b) {
 return {std::min(a.x0,b.x0),std::min(a.y0,b.y0),std::max(a.x1,b.x1),std::max(a.y1,b.y1)};
}
class Tracker {
 struct Track {Region observed;privacy::Rect coverage;uint64_t last;float vx=0,vy=0;};
 std::vector<Track> tracks_;
 uint64_t next_id_=1,last_=0;
 static privacy::Rect predicted(const Track& t,uint64_t now) {
  // Use no more than 120 ms extrapolation, and no more than one
  // observed box width/height per axis. Never integrate held predictions.
  const float dt=float(std::min<uint64_t>(now-t.last,120));
  auto b=t.observed.box;
  const float dx=std::clamp(t.vx*dt,b.x0-b.x1,b.x1-b.x0);
  const float dy=std::clamp(t.vy*dt,b.y0-b.y1,b.y1-b.y0);
  return {std::clamp(b.x0+dx,0.f,1.f),std::clamp(b.y0+dy,0.f,1.f),
          std::clamp(b.x1+dx,0.f,1.f),std::clamp(b.y1+dy,0.f,1.f)};
 }
public:
 void reset(){tracks_.clear();last_=0;}
 std::vector<Region> update(std::vector<Region> input,uint64_t now,unsigned hold_ms) {
  if(now<last_)reset();
  last_=now;
  // Union overlapping detections of the same class, never suppress coverage.
  std::vector<Region> merged;
  for(auto r:input) {
   if(!privacy::valid_rect(r.box))continue;
   auto it=std::find_if(merged.begin(),merged.end(),[&](auto& x){
     return x.category==r.category&&overlap(x.box,r.box)>=.5f;});
   if(it==merged.end())merged.push_back(r);
   else {it->box=unite(it->box,r.box);it->score=std::max(it->score,r.score);}
  }
  std::erase_if(tracks_,[&](const Track& t){return now-t.last>hold_ms;});
  std::vector<bool> used(tracks_.size(),false);
  std::vector<Track> next;
  std::vector<Region> out;
  for(auto r:merged) {
   size_t best=tracks_.size();float iou=.15f;
   for(size_t i=0;i<tracks_.size();++i)
    if(!used[i]&&tracks_[i].observed.category==r.category) {
     float score=std::max(overlap(tracks_[i].observed.box,r.box),overlap(predicted(tracks_[i],now),r.box));
     if(score>iou){iou=score;best=i;}
    }
   auto coverage=r.box;float vx=0,vy=0;
   if(best<tracks_.size()) {
    used[best]=true;r.id=tracks_[best].observed.id;
    // Keep the current box and recent motion envelope; never shrink below it.
    coverage=unite(r.box,tracks_[best].observed.box);
    const auto& prev=tracks_[best];
    const auto& b=prev.observed.box;
    if(now>prev.last) {
     const float dt=float(now-prev.last);
     vx=(r.box.x0+r.box.x1-b.x0-b.x1)*.5f/dt;
     vy=(r.box.y0+r.box.y1-b.y0-b.y1)*.5f/dt;
    } else {vx=prev.vx;vy=prev.vy;}

   } else r.id=next_id_++;
   next.push_back({r,coverage,now,vx,vy});
   r.box=coverage;out.push_back(r);
  }
  if(hold_ms)for(size_t i=0;i<tracks_.size();++i)if(!used[i]&&now-tracks_[i].last<hold_ms) {
   auto r=tracks_[i].observed;r.held=true;r.box=unite(tracks_[i].coverage,predicted(tracks_[i],now));
   out.push_back(r);next.push_back(tracks_[i]);
  }
  tracks_=std::move(next);return out;
 }
};
}
