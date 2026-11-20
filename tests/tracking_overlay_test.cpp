#include "core/tracking_overlay.hpp"
#include <iostream>
#include <algorithm>
#include <cmath>
int main(){
 sight::TrackingData no_face;
 auto empty=sight::draw_tracking_overlay(640,360,sight::TrackingKind::Face,no_face);
 if(empty.size()!=640*360*4||std::any_of(empty.begin(),empty.end(),[](auto a){return a!=0;}))return 1;
 sight::TrackingData face;
 face.landmarks.resize(478);
 for(auto &p:face.landmarks){p.x=.5f;p.y=.5f;p.confidence=1;}
 auto generated=sight::draw_tracking_overlay(640,360,sight::TrackingKind::Face,face);
 if(std::count_if(generated.begin(),generated.end(),[](auto a){return a!=0;})<10)return 2;
 sight::TrackingData pose;pose.landmarks.resize(33);
 for(int i=0;i<33;i++){pose.landmarks[i].x=.1f+(i%6)*.1f;pose.landmarks[i].y=.1f+(i/6)*.1f;pose.landmarks[i].confidence=1;}
 auto out=sight::draw_tracking_overlay(640,360,sight::TrackingKind::Body,pose);
 if(std::count_if(out.begin(),out.end(),[](auto a){return a!=0;})<60)return 3;
 sight::TrackingData hands;
 hands.hands.resize(2);
 for(size_t h=0;h<2;h++){
   auto& points=hands.hands[h].landmarks;
   points.resize(21);
   for(unsigned i=0;i<21;i++){
     points[i].x=float(.16+h*.5 + (i%5)*.025);
     points[i].y=float(.12+(i/5)*.065);
     points[i].confidence=1;
   }
 }
 auto hand_image=sight::draw_tracking_overlay(640,360,sight::TrackingKind::Hand,hands);
 if(std::count_if(hand_image.begin(),hand_image.end(),
                  [](auto a){return a!=0;})<150)return 4;
 // Holistic combines the existing three layer renderers without guessing
 // gesture labels or doing additional model inference.
 sight::TrackingData holistic_face,holistic_body,holistic_hands;
 holistic_face.landmarks.resize(478);
 for(auto &p:holistic_face.landmarks) {p.x=.17f;p.y=.2f;p.confidence=1;}
 holistic_body.landmarks.resize(33);
 for(auto &p:holistic_body.landmarks) {p.x=.50f;p.y=.5f;p.confidence=1;}
 holistic_hands.hands.resize(2);
 for(size_t h=0;h<2;++h) {
     holistic_hands.hands[h].landmarks.resize(21);
     for(auto &p:holistic_hands.hands[h].landmarks) {
         p.x=h==0?.8f:.9f;p.y=.8f;p.confidence=1;
     }
 }
 using sight::HolisticOverlayMode;
 const auto layer=[&](HolisticOverlayMode mode) {
     return sight::draw_holistic_overlay(640,360,
         holistic_face,holistic_body,holistic_hands,mode);
 };
 auto all=layer(HolisticOverlayMode::All);
 auto face_only=layer(HolisticOverlayMode::Face);
 auto body_only=layer(HolisticOverlayMode::Body);
 auto hands_only=layer(HolisticOverlayMode::Hands);
 if(all.size()!=640*360*4) return 5;
 auto present=[&](const std::vector<uint8_t>& pixels,float x,float y) {
     const size_t pos=(size_t(std::lround(y*359))*640+
                        size_t(std::lround(x*639)))*4;
     return pixels.size()>pos+3&&pixels[pos+3]!=0;
 };
 for(const auto &image:{face_only,body_only,hands_only,all}) {
     if(image.size()!=all.size())return 6;
 }
 if(!present(face_only,.17f,.2f) ||
    present(face_only,.50f,.5f) || present(face_only,.8f,.8f))return 7;
 if(!present(body_only,.5f,.5f) ||
    present(body_only,.17f,.2f) || present(body_only,.8f,.8f))return 8;
 if(!present(hands_only,.8f,.8f) ||
    !present(hands_only,.9f,.8f) ||
    present(hands_only,.17f,.2f))return 9;
 if(!present(all,.17f,.2f)||!present(all,.5f,.5f)||
    !present(all,.8f,.8f)||!present(all,.9f,.8f))return 10;
 sight::TrackingData no_subjects;
 auto blank=sight::draw_holistic_overlay(640,360,no_subjects,
                   no_subjects,no_subjects,HolisticOverlayMode::All);
 if(blank.size()!=all.size()||
    std::any_of(blank.begin(),blank.end(),[](uint8_t c){return c!=0;}))
     return 11;
 if(!sight::draw_holistic_overlay(0,360,holistic_face,
                   holistic_body,holistic_hands,HolisticOverlayMode::All).empty())
     return 12;
 // Stabilized Holistic frames use point confidence as *opacity* so an
 // occluded wrist/hand fades rather than freezing at full brightness.
 sight::TrackingData fade_pose;
 fade_pose.landmarks.resize(33);
 for(auto& p:fade_pose.landmarks){p.x=-1.f;p.y=-1.f;p.confidence=0.f;}
 fade_pose.landmarks[15]={.5f,.5f,0.f,.3f};
 auto faded_pose=sight::draw_holistic_overlay(640,360,no_face,fade_pose,
                         sight::TrackingData{},HolisticOverlayMode::Body,true);
 auto center_alpha=[&](const std::vector<uint8_t>& pixels,float x,float y){
    auto pos=(size_t(std::lround(y*359))*640+
              size_t(std::lround(x*639)))*4;
    return pixels[pos+3];
 };
 const auto alpha=center_alpha(faded_pose,.5f,.5f);
 if(alpha<1||alpha>=220)return 13;
 fade_pose.landmarks[15].confidence=0.f;
 auto hidden_pose=sight::draw_holistic_overlay(640,360,no_face,fade_pose,
                         sight::TrackingData{},HolisticOverlayMode::Body,true);
 if(center_alpha(hidden_pose,.5f,.5f)!=0)return 14;
 sight::TrackingData fade_hand;
 fade_hand.hands.resize(1);
 fade_hand.hands[0].landmarks.resize(21);
 for(auto& p:fade_hand.hands[0].landmarks)
    p={.30f,.70f,0.f,.25f};
 auto faded_hand=sight::draw_holistic_overlay(640,360,no_face,
                    sight::TrackingData{},fade_hand,HolisticOverlayMode::Hands,true);
 auto solid_hand=sight::draw_holistic_overlay(640,360,no_face,
                    sight::TrackingData{},fade_hand,HolisticOverlayMode::Hands,false);
 if(center_alpha(faded_hand,.30f,.70f)<1 ||
    center_alpha(faded_hand,.30f,.70f)>=
        center_alpha(solid_hand,.30f,.70f))return 15;
 std::cout<<"PASS: per-joint and per-hand Holistic fade opacity with full old-render compatibility\n";
 std::cout<<"PASS: Holistic face, body, both hands and all combined overlays\n";
 std::cout<<"PASS: empty detection transparency, face, body and two hand skeleton overlays\n";
}
