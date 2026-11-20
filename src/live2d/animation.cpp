#include "animation.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <set>
namespace sight::live2d {
namespace {
using JSON=nlohmann::json;
JSON read(const std::filesystem::path& path){
    std::ifstream f(path,std::ios::binary|std::ios::ate);auto size=f.tellg();
    if(size<=0||size>8*1024*1024)throw std::runtime_error("Invalid animation resource size");
    std::string data(size_t(size),'\0');f.seekg(0);
    if(!f.read(data.data(),size))throw std::runtime_error("Unreadable animation resource");
    return JSON::parse(data);
}
float number(const JSON& v){
    if(!v.is_number())throw std::runtime_error("Non-numeric curve point");
    float f=v.get<float>();if(!std::isfinite(f)||std::abs(f)>100000.f)throw std::runtime_error("Invalid curve number");
    return f;
}
float cubic(float a,float b,float c,float d,float t){
    const float s=1-t;return s*s*s*a+3*s*s*t*b+3*s*t*t*c+t*t*t*d;
}
}
float Curve::sample(float time) const{
    if(time<=first.time||segments.empty())return first.value;
    for(const auto& s:segments){
        const auto& p=s.points;
        if(time>=p[3].time)continue;
        if(s.type==2)return p[0].value;
        if(s.type==3)return p[3].value;
        if(s.type==0)return std::lerp(p[0].value,p[3].value,(time-p[0].time)/(p[3].time-p[0].time));
        float lo=0,hi=1;
        for(int i=0;i<18;++i){
            const float mid=(lo+hi)*.5f;
            if(cubic(p[0].time,p[1].time,p[2].time,p[3].time,mid)<time)lo=mid;
            else hi=mid;
        }
        return cubic(p[0].value,p[1].value,p[2].value,p[3].value,(lo+hi)*.5f);
    }
    return segments.back().points[3].value;
}
void Motion::sample(double time,std::map<std::string,float>& parameters,std::map<std::string,float>& parts) const{
    parameters.clear();parts.clear();
    if(duration<=0||!std::isfinite(time))return;
    float t=float(std::fmod(std::max(time,0.0),double(duration)));
    for(const auto& curve:curves)(curve.part?parts:parameters)[curve.id]=curve.sample(t);
}
bool read_motion(const std::filesystem::path& path,Motion& out,std::string& error){
    out={};error.clear();
    try{
        auto d=read(path);
        if(d.at("Version")!=3)throw std::runtime_error("Unsupported motion version");
        Motion m;m.duration=number(d.at("Meta").at("Duration"));
        if(m.duration<=0||m.duration>3600)throw std::runtime_error("Invalid motion duration");
        const auto& curves=d.at("Curves");
        if(!curves.is_array()||curves.size()>2048)throw std::runtime_error("Invalid motion curves");
        size_t total=0;std::set<std::pair<bool,std::string>> seen;
        for(const auto& raw:curves){
            const auto target=raw.at("Target").get<std::string>();
            if(target!="Parameter"&&target!="PartOpacity")continue;
            Curve c;c.id=raw.at("Id").get<std::string>();c.part=target=="PartOpacity";
            if(c.id.empty()||c.id.size()>512||!seen.emplace(c.part,c.id).second)throw std::runtime_error("Invalid curve ID");
            const auto& arr=raw.at("Segments");
            if(!arr.is_array()||arr.size()<2||arr.size()>100000)throw std::runtime_error("Invalid curve segments");
            size_t i=0;
            auto key=[&](){
                if(i+2>arr.size())throw std::runtime_error("Truncated motion key");
                Key p{number(arr.at(i)),number(arr.at(i+1))};i+=2;
                if(p.time<0||p.time>m.duration+1)throw std::runtime_error("Invalid motion time");
                return p;
            };
            c.first=key();Key last=c.first;
            while(i<arr.size()){
                if(++total>100000)throw std::runtime_error("Motion segment budget exceeded");
                const auto& type=arr.at(i++);if(!type.is_number_integer())throw std::runtime_error("Invalid segment type");
                Segment seg;seg.type=type.get<int>();seg.points[0]=last;
                if(seg.type==1){seg.points[1]=key();seg.points[2]=key();}
                else if(seg.type<0||seg.type>3)throw std::runtime_error("Unsupported motion segment");
                seg.points[3]=key();
                if(seg.points[3].time<=last.time||
                   (seg.type==1&&(seg.points[1].time<last.time||seg.points[2].time<seg.points[1].time||seg.points[3].time<seg.points[2].time)))
                    throw std::runtime_error("Non-monotonic motion keys");
                last=seg.points[3];c.segments.push_back(seg);
            }
            m.curves.push_back(std::move(c));
        }
        if(m.curves.empty())throw std::runtime_error("No supported parameter or part curves");
        out=std::move(m);return true;
    }catch(const std::exception& e){error=e.what();return false;}
}
bool read_expression(const std::filesystem::path& path,std::vector<ExpressionValue>& out,std::string& error){
    out.clear();error.clear();
    try{
        auto d=read(path);const auto& entries=d.at("Parameters");
        if(!entries.is_array()||entries.size()>2048)throw std::runtime_error("Invalid expression parameters");
        std::vector<ExpressionValue> value;
        for(const auto& entry:entries){
            ExpressionValue e{entry.at("Id").get<std::string>(),number(entry.at("Value")),ExpressionBlend::Add};
            if(e.id.empty()||e.id.size()>512)throw std::runtime_error("Invalid expression ID");
            auto blend=entry.value("Blend",std::string("Add"));
            if(blend=="Multiply")e.blend=ExpressionBlend::Multiply;
            else if(blend=="Overwrite")e.blend=ExpressionBlend::Overwrite;
            else if(blend!="Add")throw std::runtime_error("Unsupported expression blend");
            value.push_back(std::move(e));
        }
        out=std::move(value);return true;
    }catch(const std::exception& e){error=e.what();return false;}
}
}
