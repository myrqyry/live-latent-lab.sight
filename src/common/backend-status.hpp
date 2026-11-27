#pragma once
#include <obs-module.h>
#include "inference-health.hpp"
#include <cstdio>
#include <mutex>
#include <string>
#include <string_view>
#include <filesystem>

namespace sight_backend {
// Owned by a filter and protected by its existing input mutex. Publish only
// after the worker's generation check. No render-side instrumentation.
struct Status {
    enum class Phase { Unselected, Off, Loading, Loaded, Failed };
    Phase phase=Phase::Unselected;
    std::string model,backend,error,detail;
    InferenceHealth health;
    void request(const std::string&path,bool off=false){
        phase=off?Phase::Off:path.empty()?Phase::Unselected:Phase::Loading;
        model=path;backend.clear();error.clear();detail.clear();health={};
    }
    void complete(const std::string&path,bool ok,const char*provider,std::string_view why,bool off=false){
        request(path,off);
        if(off||path.empty())return;
        phase=ok?Phase::Loaded:Phase::Failed;
        if(ok)backend=provider?provider:"Backend not reported";
        else error=why.substr(0,768);
    }
};
inline std::string plain(std::string value){
    // OBS info labels use auto-rich-text; do not interpret a model filename
    // or backend exception as markup or an external link.
    for(char&c:value){
        if(c=='<')c='[';else if(c=='>')c=']';
        else if(c=='\r'||c=='\n'||c=='\t')c=' ';
        else if(static_cast<unsigned char>(c)<32)c=' ';
    }
    return value;
}
inline void show(obs_property_t*label,const Status&s){
    using P=Status::Phase;
    std::string message;
    auto type=OBS_TEXT_INFO_NORMAL;
    switch(s.phase){
    case P::Off:message="Tracking off.";break;
    case P::Unselected:message="No model selected.";type=OBS_TEXT_INFO_WARNING;break;
    case P::Loading:message="Loading model…";break;
    case P::Loaded:message="Loaded — "+s.backend+". Model session initialized.";break;
    case P::Failed:message="Load failed. "+(s.error.empty()?std::string("See OBS log for backend details."):s.error);type=OBS_TEXT_INFO_ERROR;break;
    }
    if(!s.model.empty())message+=" Applied model: "+std::filesystem::path(s.model).filename().string().substr(0,256);
    if(!s.detail.empty())message+=". "+s.detail;
    message="Backend snapshot: "+message;
    obs_property_set_description(label,plain(std::move(message)).c_str());
    obs_property_text_set_info_type(label,type);
}
inline void show_health(obs_property_t*label,const Status&s){
    if(!label)return;
    const auto now=InferenceHealth::Clock::now();
    const auto&h=s.health;
    std::string message;
    auto type=OBS_TEXT_INFO_NORMAL;
    if(s.phase!=Status::Phase::Loaded){
        message="Inference health unavailable until a model loads.";
    }else{
        using State=InferenceHealth::State;
        switch(h.state(now)){
        case State::Waiting:message="Waiting for first inference.";break;
        case State::Recent:message="Recent inference.";break;
        case State::Quiet:message="No recent inference (source may be idle).";type=OBS_TEXT_INFO_WARNING;break;
        case State::Error:message="Last inference failed. See OBS log.";type=OBS_TEXT_INFO_ERROR;break;
        }
        char buf[160];
        if(h.successes||h.failures){
            std::snprintf(buf,sizeof(buf)," Last inference: %.1f ms. Successful: %llu. Failed: %llu.",
                h.last_ms,(unsigned long long)h.successes,(unsigned long long)h.failures);
            message+=buf;
        }
        if(h.successes){
            std::snprintf(buf,sizeof(buf)," Last successful completion: %.1f s ago.",h.age(now)/1000.);
            message+=buf;
        }
        if(h.rate(now)>0){
            std::snprintf(buf,sizeof(buf)," Recent result rate: %.1f/s.",h.rate(now));
            message+=buf;
        }
    }
    obs_property_set_description(label,message.c_str());
    obs_property_text_set_info_type(label,type);
}
template<class Filter>
bool refresh(obs_properties_t*props,obs_property_t*,void*data){
    auto*f=static_cast<Filter*>(data);
    auto*label=obs_properties_get(props,"sight_backend_status");
    if(!f||!label)return false;
    Status snapshot;
    {std::lock_guard lock(f->input_mutex);snapshot=f->backend_status;}
    show(label,snapshot);
    show_health(obs_properties_get(props,"sight_inference_health"),snapshot);
    return true;
}
template<class Filter>
void add(obs_properties_t*props,Filter*f){
    auto*label=obs_properties_add_text(props,"sight_backend_status","Backend status available on an applied source.",OBS_TEXT_INFO);
    obs_property_text_set_info_word_wrap(label,true);
    auto*health=obs_properties_add_text(props,"sight_inference_health","Inference health available on an applied source.",OBS_TEXT_INFO);
    obs_property_text_set_info_word_wrap(health,true);
    if(f){
        refresh<Filter>(props,nullptr,f);
        obs_properties_add_button2(props,"sight_backend_refresh","Refresh backend status",refresh<Filter>,nullptr);
    }
}
}
