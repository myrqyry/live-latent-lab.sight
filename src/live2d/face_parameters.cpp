#include "face_parameters.hpp"
#include <algorithm>
#include <cmath>
#include <optional>
#include <set>
#include <string_view>
namespace sight::live2d {
namespace {
using Maybe=std::optional<float>;
Maybe score(const TrackingData& data,std::string_view name){
    if(data.blendshapes.size()!=data.blendshape_names.size()||data.blendshapes.size()>64)return {};
    Maybe value;
    for(size_t i=0;i<data.blendshape_names.size();++i){
        if(data.blendshape_names[i]!=name)continue;
        if(value||!std::isfinite(data.blendshapes[i]))return {};
        value=std::clamp(data.blendshapes[i],0.f,1.f);
    }
    return value;
}
Maybe difference(const TrackingData& data,std::string_view positive,std::string_view negative){
    auto p=score(data,positive),n=score(data,negative);
    if(!p&&!n)return {};
    return p.value_or(0)-n.value_or(0);
}
Maybe average(Maybe a,Maybe b){
    if(a&&b)return (*a+*b)*.5f;
    return a?a:b;
}
}
std::string_view standard_parameter(std::string_view id){
    // Explicit legacy Cubism 2-style IDs used in converted MOC3 rigs.
    static constexpr std::pair<std::string_view,std::string_view> aliases[]={
        {"PARAM_EYE_L_OPEN","ParamEyeLOpen"},{"PARAM_EYE_R_OPEN","ParamEyeROpen"},
        {"PARAM_MOUTH_OPEN_Y","ParamMouthOpenY"},{"PARAM_MOUTH_FORM","ParamMouthForm"},
        {"PARAM_EYE_BALL_X","ParamEyeBallX"},{"PARAM_EYE_BALL_Y","ParamEyeBallY"},
        {"PARAM_ANGLE_X","ParamAngleX"},{"PARAM_ANGLE_Y","ParamAngleY"},
        {"PARAM_ANGLE_Z","ParamAngleZ"},{"PARAM_BREATH","ParamBreath"}
    };
    for(auto [legacy,modern]:aliases)if(id==legacy)return modern;
    return id;
}
Values face_parameters(const TrackingSignal* face,const std::vector<Parameter>& rig,
                       const std::vector<std::string>& lip_ids){
    Values out;
    if(!face||face->kind!=TrackingKind::Face||face->data.landmarks.size()<468)return out;
    const auto& data=face->data;
    const auto left=score(data,"eyeBlinkLeft"),right=score(data,"eyeBlinkRight");
    const auto jaw=score(data,"jawOpen");
    const auto mouth=average(difference(data,"mouthSmileLeft","mouthFrownLeft"),
                             difference(data,"mouthSmileRight","mouthFrownRight"));
    const auto gaze_x=average(difference(data,"eyeLookOutLeft","eyeLookInLeft"),
                              difference(data,"eyeLookInRight","eyeLookOutRight"));
    const auto gaze_y=average(difference(data,"eyeLookUpLeft","eyeLookDownLeft"),
                              difference(data,"eyeLookUpRight","eyeLookDownRight"));
    Maybe yaw,pitch,roll;
    if(data.has_head_transform&&std::all_of(data.head_transform.begin(),data.head_transform.end(),
                                           [](float f){return std::isfinite(f);})){
        const auto& m=data.head_transform;
        const float x=std::hypot(m[0],m[4]),yz=std::hypot(m[9],m[10]);
        if(x>1e-5f&&yz>1e-5f){
            constexpr float degrees=57.2957795131f;
            // Starting convention from the existing Sight head rotation.
            // Live model visual calibration remains a separate gate.
            yaw=degrees*std::atan2(m[8],x);
            pitch=-degrees*std::atan2(m[9],m[10]);
            roll=degrees*std::atan2(m[4],m[0]);
        }
    }
    std::set<std::string> seen,duplicates;
    for(auto& p:rig)if(!seen.insert(p.id).second)duplicates.insert(p.id);
    for(auto& p:rig){
        if(p.id.empty()||duplicates.contains(p.id)||!std::isfinite(p.minimum)||!std::isfinite(p.neutral)||
           !std::isfinite(p.maximum)||p.minimum>p.neutral||p.neutral>p.maximum)continue;
        const auto id=standard_parameter(p.id);
        Maybe value;
        auto signed_value=[&](Maybe s)->Maybe{
            if(!s)return {};
            return p.neutral+*s*(*s<0?p.neutral-p.minimum:p.maximum-p.neutral);
        };
        if(id=="ParamEyeLOpen"&&left)value=p.neutral-*left*(p.neutral-p.minimum);
        else if(id=="ParamEyeROpen"&&right)value=p.neutral-*right*(p.neutral-p.minimum);
        else if((lip_ids.empty()?id=="ParamMouthOpenY":
                 std::find(lip_ids.begin(),lip_ids.end(),p.id)!=lip_ids.end())&&jaw)
            value=p.neutral+*jaw*(p.maximum-p.neutral);
        else if(id=="ParamMouthForm")value=signed_value(mouth);
        else if(id=="ParamEyeBallX")value=signed_value(gaze_x);
        else if(id=="ParamEyeBallY")value=signed_value(gaze_y);
        else if(id=="ParamAngleX"&&yaw)value=p.neutral+*yaw;
        else if(id=="ParamAngleY"&&pitch)value=p.neutral+*pitch;
        else if(id=="ParamAngleZ"&&roll)value=p.neutral+*roll;
        if(value&&std::isfinite(*value))out.emplace(p.id,std::clamp(*value,p.minimum,p.maximum));
    }
    return out;
}
}
