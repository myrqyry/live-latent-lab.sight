#pragma once
#include "common/image-utils.hpp"
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
#include <algorithm>

namespace sight::sd {
struct AnimationValues {
    TransformParams transform{};
    float denoise=.35f,control_strength=.65f;
    std::string prompt;
};
class AnimationSchedule {
    struct Key {uint64_t frame;float value;};
    enum Field {Zoom,PanX,PanY,Rotation,Denoise,Control,Count};
    std::array<std::vector<Key>,Count> tracks_;
    std::vector<std::pair<uint64_t,std::string>> prompts_;
    static std::string_view trim(std::string_view x){
        auto a=x.find_first_not_of(" \t\r");
        if(a==x.npos)return {};
        return x.substr(a,x.find_last_not_of(" \t\r")-a+1);
    }
    template<class T>static bool number(std::string_view x,T&v){
        x=trim(x);if(x.empty())return false;
        auto r=std::from_chars(x.data(),x.data()+x.size(),v);
        return r.ec==std::errc{}&&r.ptr==x.data()+x.size();
    }
    template<class F>static bool lines(std::string_view input,F fn,std::string&error){
        if(input.size()>65536){error="Schedule exceeds 64 KiB";return false;}
        size_t line=0;
        while(!input.empty()){
            ++line;
            auto end=input.find('\n');auto row=trim(input.substr(0,end));
            if(end==input.npos)input={};else input.remove_prefix(end+1);
            if(row.empty()||row.front()=='#')continue;
            auto colon=row.find(':');uint64_t frame=0;
            if(colon==row.npos||!number(row.substr(0,colon),frame)||frame>1000000||
               !fn(frame,trim(row.substr(colon+1)))){
                error="Invalid schedule line "+std::to_string(line)+": "+std::string(row.substr(0,120));return false;
            }
        }
        return true;
    }
    static float sample(const std::vector<Key>&keys,uint64_t frame,float base){
        if(keys.empty())return base;
        auto next=std::upper_bound(keys.begin(),keys.end(),frame,[](uint64_t f,const Key&k){return f<k.frame;});
        if(next==keys.begin()){
            if(frame==0)return base;
            return float(base+(next->value-base)*(double(frame)/next->frame));
        }
        const auto&prev=*(next-1);
        if(next==keys.end())return prev.value;
        return float(prev.value+(next->value-prev.value)*(double(frame-prev.frame)/double(next->frame-prev.frame)));
    }
public:
    bool parse(std::string_view motion,std::string_view prompts,std::string&error){
        AnimationSchedule candidate;error.clear();
        constexpr std::array<std::string_view,Count> names={"zoom","pan_x","pan_y","rotation","denoise","control"};
        constexpr std::array<float,Count> lo={.9f,-100,-100,-15,0,0},hi={1.1f,100,100,15,1,1};
        if(!lines(motion,[&](uint64_t frame,std::string_view row){
            if(row.empty())return false;
            while(!row.empty()){
                auto comma=row.find(',');auto entry=trim(row.substr(0,comma));auto eq=entry.find('=');
                if(eq==entry.npos)return false;
                auto it=std::find(names.begin(),names.end(),trim(entry.substr(0,eq)));
                float val=0;if(it==names.end()||!number(entry.substr(eq+1),val)||!std::isfinite(val))return false;
                size_t index=it-names.begin();
                auto&track=candidate.tracks_[index];
                if(val<lo[index]||val>hi[index]||(!track.empty()&&track.back().frame>=frame))return false;
                track.push_back({frame,val});
                if(comma==row.npos)break;
                row.remove_prefix(comma+1);if(trim(row).empty())return false;
            }
            return true;
        },error))return false;
        if(!lines(prompts,[&](uint64_t frame,std::string_view row){
            if(row.empty()||(!candidate.prompts_.empty()&&candidate.prompts_.back().first>=frame))return false;
            candidate.prompts_.emplace_back(frame,std::string(row));return true;
        },error))return false;
        *this=std::move(candidate);return true;
    }
    AnimationValues evaluate(uint64_t frame,AnimationValues base,std::string_view prompt)const{
        auto&t=base.transform;
        t.zoom_factor=sample(tracks_[Zoom],frame,t.zoom_factor);
        t.pan_x=sample(tracks_[PanX],frame,t.pan_x);t.pan_y=sample(tracks_[PanY],frame,t.pan_y);
        t.rotation_deg=sample(tracks_[Rotation],frame,t.rotation_deg);
        base.denoise=sample(tracks_[Denoise],frame,base.denoise);
        base.control_strength=sample(tracks_[Control],frame,base.control_strength);
        base.prompt=prompt;
        auto next=std::upper_bound(prompts_.begin(),prompts_.end(),frame,[](uint64_t f,const auto&k){return f<k.first;});
        if(next!=prompts_.begin())base.prompt=(next-1)->second;
        return base;
    }
};
}
