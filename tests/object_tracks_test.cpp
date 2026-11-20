#include "core/object_tracks.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
using sight::ObjectDetection;
using sight::ObjectTracks;
static ObjectDetection box(float x,float y,float w,float h,
                           const char* label="person",float score=.94f){
    return {x,y,w,h,score,label,0};
}
int main(){
    ObjectTracks tracks;
    assert(tracks.step({}).empty());
    auto first=tracks.step({box(.1f,.2f,.2f,.2f),box(.7f,.2f,.18f,.2f)});
    assert(first.empty()); // new IDs require two observations, no flicker
    auto second=tracks.step({box(.72f,.2f,.18f,.2f),box(.12f,.2f,.2f,.2f)});
    assert(second.size()==2);
    const auto left=second[0].x<second[1].x?second[0]:second[1];
    const auto right=second[0].x<second[1].x?second[1]:second[0];
    assert(left.id!=right.id);
    auto reordered=tracks.step({box(.74f,.2f,.18f,.2f),box(.14f,.2f,.2f,.2f)});
    assert(reordered.size()==2);
    for(const auto& obj:reordered){
        assert(obj.id==(obj.x<.5f?left.id:right.id));
    }
    // Disappearing does not fabricate a visible detection or recycle IDs.
    assert(tracks.step({}).empty());
    auto returned=tracks.step({box(.16f,.2f,.2f,.2f)});
    assert(returned.size()==1 && returned[0].id==left.id);
    for(int i=0;i<6;++i)assert(tracks.step({}).empty());
    assert(tracks.step({box(.17f,.2f,.2f,.2f)}).empty());
    returned=tracks.step({box(.18f,.2f,.2f,.2f)});
    assert(returned.size()==1 && returned[0].id!=left.id);
    // No matching different categories on identical rectangles.
    tracks.reset();
    assert(tracks.step({box(.2f,.2f,.3f,.3f,"cat")}).empty());
    assert(tracks.step({box(.2f,.2f,.3f,.3f,"dog")}).empty());
    auto cats=tracks.step({box(.2f,.2f,.3f,.3f,"cat")});
    assert(cats.size()==1 && cats[0].label=="cat");
    // Ignore invalid geometry, invalid confidence, huge or nonfinite boxes.
    tracks.reset();
    ObjectDetection invalid=box(.1f,.1f,.2f,.2f);
    invalid.x=std::nanf("");
    assert(tracks.step({invalid,box(-100.f,-20.f,1.f,1.f),
                        box(.1f,.1f,0.f,.2f),box(.1f,.1f,.2f,.2f,"person",.1f)}).empty());
    assert(tracks.step({}).empty());
    // Same-name crossing: extrapolate motion to preserve left/right IDs.
    tracks.reset();
    tracks.step({box(.1f,.3f,.1f,.2f),box(.8f,.3f,.1f,.2f)});
    auto visible=tracks.step({box(.2f,.3f,.1f,.2f),box(.7f,.3f,.1f,.2f)});
    assert(visible.size()==2);
    const int moving_right=visible[0].x<visible[1].x?visible[0].id:visible[1].id;
    tracks.step({box(.32f,.3f,.1f,.2f),box(.58f,.3f,.1f,.2f)});
    visible=tracks.step({box(.46f,.3f,.1f,.2f),box(.44f,.3f,.1f,.2f)});
    assert(visible.size()==2);
    auto next=tracks.step({box(.58f,.3f,.1f,.2f),box(.32f,.3f,.1f,.2f)});
    bool found=false;
    for(const auto& obj:next)if(obj.id==moving_right)
        found=obj.x>.47f;
    assert(found);
    std::cout<<"SIGHT_OBJECT_TRACKING_UNIT_PASS=YES\n";
}
