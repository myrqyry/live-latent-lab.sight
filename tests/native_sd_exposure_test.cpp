#include "core/native_sd_exposure.hpp"
#include <iostream>
#include <cmath>
int main(){
    using namespace sight::sd;
    ExposureAnchor anchor;
    std::vector<uint8_t> first(64*64*3), drift(first.size());
    for(size_t i=0;i<first.size();i+=3){
        int v=40+int(i/3%128);
        first[i]=v+20;first[i+1]=v;first[i+2]=v-20;
        for(int c=0;c<3;++c)drift[i+c]=first[i+c]+12;
    }
    auto orig=drift;
    auto a=anchor.apply(first,3,true,1.f);
    auto off=anchor.apply(drift,3,false,0.f);
    bool neutral=first[0]==60&&drift==orig&&!a.applied&&!off.applied;
    auto r=anchor.apply(drift,3,false,1.f);
    bool correct=r.applied&&drift==first&&r.before_error>11.9f&&r.after_error<.01f;
    drift=orig;
    auto half=anchor.apply(drift,3,false,.5f);
    bool partial=half.applied&&std::abs(half.after_error-6)<.01f;
    std::vector<uint8_t> flat(first.size(),240);
    auto bounded=anchor.apply(flat,3,false,1.f);
    bool cap=bounded.applied&&flat[0]==224;
    std::vector<uint8_t> rgba={255,230,220,13,50,40,30,77};
    auto old=rgba;
    auto reset=anchor.apply(rgba,4,true,1.f);
    bool refresh=!reset.applied&&rgba==old;
    auto shifted=rgba;
    shifted[0]=255;shifted[1]=250;shifted[2]=240;
    shifted[4]=90;shifted[5]=80;shifted[6]=70;
    const auto prior=shifted;
    auto clipped=anchor.apply(shifted,4,false,1.f);
    bool gamut=clipped.applied&&shifted[3]==13&&shifted[7]==77&&
        int(shifted[0])-shifted[1]==int(prior[0])-prior[1]&&
        int(shifted[1])-shifted[2]==int(prior[1])-prior[2];
    auto bad=anchor.apply(rgba,2,false,1.f);
    bool reject=!bad.applied&&rgba==old;
    anchor.reset();
    auto cleared=anchor.apply(drift,3,false,1.f);
    bool cleared_ok=!cleared.applied;
    std::cout<<"neutral="<<neutral<<" correct="<<correct<<" partial="<<partial<<" cap="<<cap<<" refresh="<<refresh<<" reject="<<reject<<" gamut="<<gamut<<" reset="<<cleared_ok<<"\n";
    return neutral&&correct&&partial&&cap&&refresh&&reject&&gamut&&cleared_ok?0:1;
}
