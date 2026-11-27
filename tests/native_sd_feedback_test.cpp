#include "core/native_sd_feedback.hpp"
#include <iostream>
#include <algorithm>
int main(){
    using namespace sight::sd;
    Feedback feedback;
    std::vector<uint8_t> rgb(64*64*3);
    for(unsigned y=0;y<64;++y)for(unsigned x=0;x<64;++x){
        size_t i=(y*64+x)*3;rgb[i]=x<32?10:240;rgb[i+1]=y*3;rgb[i+2]=50;
    }
    TransformParams transform;
    bool first=feedback.init_rgb(64,64,transform).empty();
    bool committed=feedback.accept(rgb,64,64,3);
    bool identity=feedback.init_rgb(64,64,transform)==rgb;
    transform.pan_x=4;
    auto moved=feedback.init_rgb(64,64,transform);
    bool motion=moved!=rgb&&moved[(20*64+34)*3]==10&&moved[(20*64+38)*3]==240;
    transform.pan_x=0;transform.rotation_deg=3;
    bool rotated=feedback.init_rgb(64,64,transform)!=rgb;
    transform.zoom_factor=0;
    bool invalid=feedback.init_rgb(64,64,transform).empty()&&!feedback.accept({},64,64,3);
    bool mismatch=feedback.init_rgb(128,64,TransformParams{}).empty();
    bool seed=feedback_seed(42,0,1)==42&&feedback_seed(42,1,1)==43&&feedback_seed(42,3,0)==42&&feedback_seed(INT64_MAX,1,1)>=0;
    std::vector<uint8_t> guide;
    bool edge=make_native_guide(rgb,64,64,"canny",guide)&&guide.size()==rgb.size()&&
        *std::max_element(guide.begin(),guide.end())==255&&*std::min_element(guide.begin(),guide.end())==0;
    bool tile=make_native_guide(rgb,64,64,"tile",guide)&&guide==rgb;
    bool reject=!make_native_guide(rgb,64,64,"depth_anything_v2",guide)&&guide.empty();
    feedback.reset();
    bool reset=feedback.init_rgb(64,64,TransformParams{}).empty();
    std::cout<<"first="<<first<<" accept="<<committed<<" identity="<<identity<<" motion="<<motion<<" rotation="<<rotated<<" invalid="<<invalid<<" mismatch="<<mismatch<<" seed="<<seed<<" canny="<<edge<<" tile="<<tile<<" reject="<<reject<<" reset="<<reset<<"\n";
    return first&&committed&&identity&&motion&&rotated&&invalid&&mismatch&&seed&&edge&&tile&&reject&&reset?0:1;
}
