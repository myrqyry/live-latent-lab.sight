#include "core/tasks_tracker.hpp"
#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
int main(int argc,char **argv){
 if(argc!=5)return 2;
 std::ifstream in(argv[3],std::ios::binary);
 std::vector<uint8_t> data((std::istreambuf_iterator<char>(in)),{});
 if(data.size()!=512*512*4){std::cerr<<"wrong fixture bytes "<<data.size()<<"\n";return 3;}
 const std::string selected=argv[4];
 sight::TrackingKind kind=selected=="face"?sight::TrackingKind::Face:
     selected.starts_with("hand")?sight::TrackingKind::Hand:sight::TrackingKind::Body;
 sight::TasksTracker tracker;
 if(!tracker.load(argv[1],argv[2],kind)){std::cerr<<"load error: "<<tracker.last_error()<<"\n";return 4;}
 for(int f=0;f<3;f++) {
  sight::TrackingData out;
  auto before=std::chrono::steady_clock::now();
  if(!tracker.detect(data.data(),512,512,33*(f+1),out)) {std::cerr<<"error: "<<tracker.last_error()<<"\n";return 5;}
  auto ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-before).count();
  std::cout<<argv[4]<<" frame="<<f<<" landmark_count="<<out.landmarks.size()<<" blendshape_count="<<out.blendshapes.size()<<" transform="<<out.has_head_transform<<" inference_ms="<<ms<<"\n";
  if(kind==sight::TrackingKind::Hand) {
   if(out.hands.empty()||out.hands.size()>2||out.hands[0].landmarks.size()!=21||
      out.landmarks.size()!=21||out.hands[0].gesture.empty()||
      out.hands[0].pinch<0||out.hands[0].pinch>1) return 6;
   if(selected=="hand" && out.hands[0].gesture!="Thumb_Up")return 7;
   if(selected=="hand_world") {
     if(out.hands[0].world_landmarks.size()!=21)return 9;
     for(const auto& p:out.hands[0].world_landmarks)
       if(!std::isfinite(p.x)||!std::isfinite(p.y)||!std::isfinite(p.z)||
          p.confidence<0||p.confidence>1)return 10;
     std::cout<<"VALIDATED_REAL_HAND_WORLD_METERS=21"
       <<" world_wrist_x="<<out.hands[0].world_landmarks[0].x
       <<" normalized_wrist_x="<<out.hands[0].landmarks[0].x<<"\n";
   }
   std::cout<<"hand_count="<<out.hands.size()
            <<" gesture="<<out.hands[0].gesture
            <<" pinch="<<out.hands[0].pinch<<"\n";
  } else if(out.landmarks.size()!=(kind==sight::TrackingKind::Face?478u:33u))return 6;
  if(selected=="pose_world") {
    if(out.world_landmarks.size()!=33)return 11;
    for(const auto& p:out.world_landmarks)
      if(!std::isfinite(p.x)||!std::isfinite(p.y)||!std::isfinite(p.z)||
         p.confidence<0||p.confidence>1)return 12;
    std::cout<<"VALIDATED_REAL_POSE_WORLD_METERS=33"
      <<" world_hip_x="<<out.world_landmarks[23].x
      <<" normalized_hip_x="<<out.landmarks[23].x<<"\n";
  }
 }
}
