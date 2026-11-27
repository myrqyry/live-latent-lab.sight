#include "core/native_sd_schedule.hpp"
#include <iostream>
#include <cmath>
int main(){
    using namespace sight::sd;
    AnimationSchedule s;
    std::string error;
    bool parsed=s.parse("0: zoom=1, pan_x=0, denoise=.2\n10: zoom=1.02, pan_x=10, denoise=.4\n20: rotation=4\n","0: forest\n10: ocean",error);
    AnimationValues base{};base.transform.pan_y=3;base.transform.rotation_deg=-2;
    base.denoise=.35f;base.control_strength=.65f;
    auto a=s.evaluate(5,base,"base"),b=s.evaluate(10,base,"base"),c=s.evaluate(30,base,"base");
    bool interpolation=std::abs(a.transform.zoom_factor-1.01f)<.0001f&&a.transform.pan_x==5&&std::abs(a.denoise-.3f)<.0001f&&a.transform.pan_y==3;
    bool hold=b.prompt=="ocean"&&c.transform.pan_x==10&&c.transform.rotation_deg==4;
    bool prompt=a.prompt=="forest";
    bool invalid=!s.parse("0: zoom=0", "",error)&&!error.empty();
    bool atomic=s.evaluate(5,base,"base").prompt=="forest";
    bool duplicate=!s.parse("0: pan_x=1\n0: pan_x=2","",error);
    bool unknown=!s.parse("0: mystery=2","",error);
    bool nan=!s.parse("0: denoise=nan","",error);
    bool early=s.parse("10: zoom=1.04","10: sunset",error)&&s.evaluate(0,base,"base").prompt=="base"&&s.evaluate(0,base,"base").transform.zoom_factor==base.transform.zoom_factor;
    std::cout<<"parsed="<<parsed<<" interpolation="<<interpolation<<" hold="<<hold<<" prompt="<<prompt<<" invalid="<<invalid<<" atomic="<<atomic<<" duplicate="<<duplicate<<" unknown="<<unknown<<" nan="<<nan<<" early="<<early<<"\n";
    return parsed&&interpolation&&hold&&prompt&&invalid&&atomic&&duplicate&&unknown&&nan&&early?0:1;
}
