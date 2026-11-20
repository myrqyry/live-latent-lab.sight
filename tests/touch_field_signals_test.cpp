#include "core/touch_field_signals.hpp"
#include <cmath>
#include <iostream>
int main() {
    sight::TrackingSignal sample;
    sample.kind=sight::TrackingKind::Hand;
    sample.source_width=512;sample.source_height=512;
    sample.data.hands.resize(2);
    for(auto& h:sample.data.hands)h.landmarks.resize(21);
    sample.data.hands[0].landmarks[8].x=.25f;
    sample.data.hands[0].landmarks[8].y=.75f;
    sample.data.hands[0].pinch=.8f;
    sample.data.hands[1].landmarks[8].x=.65f;
    sample.data.hands[1].landmarks[8].y=.35f;
    sample.data.hands[1].pinch=.1f;
    const auto first=sight::select_touch_fingertips(sample,512,512);
    if(first.count!=2 || first.fingers[0].x!=.25f ||
       first.fingers[1].y!=.35f || first.fingers[0].pinch!=.8f ||
       first.fingers[1].active!=1.f)return 1;
    if(sight::select_touch_fingertips(sample,640,512).count!=0)return 2;
    // The native capture can be 1920x1080 while the model processes
    // downscaled 512x288. Normalized points still describe the native
    // frame; the signal must carry *native* geometry, not model geometry.
    sample.source_width=1920;sample.source_height=1080;
    if(sight::select_touch_fingertips(sample,1920,1080).count!=2)return 7;
    if(sight::select_touch_fingertips(sample,512,288).count!=0)return 8;
    sample.source_width=512;sample.source_height=512;
    if(sight::select_touch_fingertips(sample,0,0).count!=0)return 3;
    sample.kind=sight::TrackingKind::Face;
    if(sight::select_touch_fingertips(sample,512,512).count!=0)return 4;
    sample.kind=sight::TrackingKind::Hand;
    sample.data.hands[0].landmarks[8].x=NAN;
    sample.data.hands[1].pinch=2.f;
    const auto one=sight::select_touch_fingertips(sample,512,512);
    if(one.count!=1||one.fingers[0].pinch!=1.f)return 5;
    sample.data.hands[1].landmarks.clear();
    if(sight::select_touch_fingertips(sample,512,512).count!=0)return 6;
    std::cout<<"PASS: native camera geometry, two fingertips, invalid inputs, and source mismatch\n";
}
