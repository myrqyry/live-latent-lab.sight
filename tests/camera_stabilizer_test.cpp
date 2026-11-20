#include "core/camera_stabilizer.hpp"
#include <cmath>
#include <iostream>
#include <algorithm>
static sight_flow_v1 field(float x,float y) {
    sight_flow_v1 f{};f.interval_seconds=.05f;f.coverage=1;
    for(auto& v:f.cells)v={-x/320,-y/180,1,0};
    return f;
}
int main(){
    sight::CameraStabilizer s;
    double raw=0,corrected=0;float p=0;bool valid=true;
    for(int i=0;i<100;++i) {
        const float next=i%2?4.f:-4.f;
        auto r=s.step(field(next-p,0),320,180,.5f,.06f);
        valid&=r.valid;raw+=next*next;
        corrected+=std::pow(next-r.x*320,2);
        p=next;
    }
    const bool jitter=valid&&corrected/raw<.15;
    s.reset();auto mixed=field(5,-3);
    for(unsigned y=5;y<13;++y)for(unsigned x=9;x<23;++x)mixed.cells[y*32+x]={.06f,.03f,1,0};
    auto r=s.step(mixed,320,180,.5f,.06f);
    const bool subject=r.valid&&r.x>0&&r.y<0&&r.consensus>.75;
    s.reset();auto opposed=field(5,0);
    for(unsigned y=0;y<18;++y)for(unsigned x=16;x<32;++x)opposed.cells[y*32+x].dx=5.f/320;
    const bool ambiguous=!s.step(opposed,320,180,.5f,.06f).valid;
    auto clustered=field(5,0);
    for(unsigned y=0;y<18;++y)for(unsigned x=0;x<32;++x)
        if(y>=9||x>=16)clustered.cells[y*32+x]={};
    clustered.coverage=.25f;
    const bool clustered_rejected=!s.step(clustered,320,180,.5f,.06f).valid;
    s.reset();bool pan=true;float peak=0;
    for(int i=0;i<120;++i){
        auto r=s.step(field(5,3),320,180,.5f,.06f);
        peak=std::max(peak,std::max(std::abs(r.x),std::abs(r.y)));
        pan&=r.valid&&std::abs(r.x)<=.06f&&std::abs(r.y)<=.06f;
    }
    auto cut=field(0,0);cut.interval_seconds=1;
    auto lost=s.step(cut,320,180,.5f,.06f);
    auto stopped=s.step(field(0,0),320,180,.5f,.06f);
    const bool reset=!lost.valid&&lost.x==0&&lost.y==0&&stopped.valid&&stopped.x==0&&stopped.y==0;
    auto zero=s.step(field(5,3),320,180,.5f,0);
    const bool no_crop=zero.valid&&zero.x==0&&zero.y==0;
    std::cout<<"CAMERA_STABILIZER jitter="<<jitter<<" rms_ratio="<<std::sqrt(corrected/raw)
        <<" foreground="<<subject<<" ambiguous="<<ambiguous<<" clustered="<<clustered_rejected
        <<" pan="<<pan<<" peak="<<peak<<" reset="<<reset<<" zero="<<no_crop<<"\n";
    return !(jitter&&subject&&ambiguous&&clustered_rejected&&pan&&reset&&no_crop);
}
