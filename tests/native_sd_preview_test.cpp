#include "core/native_sd_preview.hpp"
#include <iostream>
#include <cmath>
int main(){
    using namespace sight::sd;
    PreviewKey a{1,0,128,128,1000000000,false,{}},b{1,1,128,128,1400000000,true,{}};
    b.transform.pan_x=8;
    bool pair=can_interpolate(a,b);
    auto cut=b;cut.recursive=false;
    auto skip=b;skip.frame=2;
    auto restart=b;restart.epoch=2;
    auto resized=b;resized.width=256;
    bool guards=!can_interpolate(a,cut)&&!can_interpolate(a,skip)&&!can_interpolate(a,restart)&&!can_interpolate(a,resized);
    auto interval=preview_duration_ns(a,b,1000);
    bool timing=interval==400000000&&preview_phase(0,interval)==0&&std::abs(preview_phase(200000000,interval)-.5f)<.001f&&preview_phase(900000000,interval)==1;
    bool cap=preview_duration_ns(a,b,100)==100000000;
    auto middle=preview_mapping(b.transform,.5f);
    bool motion=std::abs(middle.old_pan_x-4.f)<.001f&&std::abs(middle.new_pan_x-4.f)<.001f;
    TransformParams mixed;
    mixed.zoom_factor=1.075f;mixed.pan_x=5;mixed.pan_y=-7;mixed.rotation_deg=12;
    auto m=preview_mapping(mixed,.37f);
    auto forward=[](float x,float y,float scale,float angle,float tx,float ty){
        return std::pair{scale*(std::cos(angle)*x-std::sin(angle)*y)+tx,
                         scale*(std::sin(angle)*x+std::cos(angle)*y)+ty};
    };
    auto partial=forward(31,-17,m.old_scale,m.old_angle,m.old_pan_x,m.old_pan_y);
    auto composed=forward(partial.first,partial.second,m.new_scale,m.new_angle,m.new_pan_x,m.new_pan_y);
    auto full=forward(31,-17,mixed.zoom_factor,mixed.rotation_deg*3.14159265358979323846f/180.f,mixed.pan_x,mixed.pan_y);
    bool composition=std::abs(composed.first-full.first)<.0001f&&std::abs(composed.second-full.second)<.0001f;
    auto end=preview_mapping(b.transform,1);
    bool endpoint=end.new_pan_x==0&&end.new_scale==1&&end.new_angle==0;
    std::cout<<"pair="<<pair<<" guards="<<guards<<" timing="<<timing<<" cap="<<cap<<" motion="<<motion<<" composition="<<composition<<" endpoint="<<endpoint<<"\n";
    return pair&&guards&&timing&&cap&&motion&&composition&&endpoint?0:1;
}
