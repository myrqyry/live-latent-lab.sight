#include "core/object_overlay.hpp"
#include <cassert>
#include <iostream>
#include <vector>
int main(){
    std::vector<sight::ObjectDetection> objects{{.1f,.2f,.25f,.3f,.88f,"cat",16,1},
                                                {.6f,.2f,.26f,.35f,.84f,"dog",17,2}};
    const auto img=sight::draw_object_overlay(512,333,objects);
    assert(img.size()==512u*333u*4u);
    auto alpha=[&](int x,int y){return img[(size_t(y)*512+x)*4+3];};
    assert(alpha(52,67)>0);
    assert(alpha(307,67)>0);
    assert(alpha(256,310)==0);
    assert(alpha(200,300)==0);
    assert(sight::draw_object_overlay(0,333,objects).empty());
    assert(sight::draw_object_overlay(512,333,{}).size()==512u*333u*4u);
    const auto ids=objects;
    objects[0].id=12;
    const auto changed=sight::draw_object_overlay(512,333,objects);
    assert(img!=changed); // visible identity badge is actually drawn
    std::cout<<"SIGHT_OBJECT_OVERLAY_UNIT_PASS=YES\n";
}
