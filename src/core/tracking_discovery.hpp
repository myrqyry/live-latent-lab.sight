#pragma once
// UI/control-thread inventory. Configured endpoints are discoverable without
// starting inference. Never called per-frame from the graphics/audio paths.
#include <obs-module.h>
#include "tracking_signal_bus.hpp"
#include "tracking_mode.hpp"
#include "sight_snapshot_v1.h"
#include <map>
#include <optional>
#include <string>
#include <algorithm>
namespace sight::discovery {
struct ChannelChoice {
    sight::TrackingKind kind;
    std::string owner;
    bool enabled;
    bool gestures;
    bool ambiguous=false;
};
using ChannelChoices=std::map<std::string,ChannelChoice>;
inline void discover_producer(obs_source_t* parent,obs_source_t* producer,void* opaque) {
    const std::string id=obs_source_get_unversioned_id(producer);
    const bool unified=id=="sight_unified_tracking_filter";
    const bool holistic=id=="sight_holistic_motion_filter";
    if(!unified&&!holistic&&id!="sight_face_motion_filter"&&id!="sight_body_motion_filter"&&
       id!="sight_hand_motion_filter"&&id!="sight_rtmw_tracking_filter")return;
    auto& choices=*static_cast<ChannelChoices*>(opaque);
    auto* settings=obs_source_get_settings(producer);
    const std::string owner=std::string(obs_source_get_name(parent))+" / "+obs_source_get_name(producer);
    const auto add=[&](const char* key,sight::TrackingKind kind,bool gestures){
        const std::string channel=obs_data_get_string(settings,key);
        if(channel.empty()||channel.size()>96)return;
        const auto [it,inserted]=choices.try_emplace(channel,ChannelChoice{
            kind,owner,obs_source_enabled(producer),gestures});
        if(!inserted&&(it->second.owner!=owner||it->second.kind!=kind))it->second.ambiguous=true;
    };
    if(unified){
        const auto mode=parse_tracking_mode(obs_data_get_string(settings,"tracking_mode"));
        if(mode==TrackingMode::Holistic){
            add(tracking_mode_key(mode,"signal_channel_face").c_str(),TrackingKind::Face,false);
            add(tracking_mode_key(mode,"signal_channel_body").c_str(),TrackingKind::Body,false);
            add(tracking_mode_key(mode,"signal_channel_hand").c_str(),TrackingKind::Hand,false);
        }else if(mode!=TrackingMode::None&&mode!=TrackingMode::Object){
            const auto kind=mode==TrackingMode::Face?TrackingKind::Face:
                mode==TrackingMode::Body?TrackingKind::Body:
                mode==TrackingMode::Hand?TrackingKind::Hand:TrackingKind::WholeBody;
            add(tracking_mode_key(mode,"signal_channel").c_str(),kind,mode==TrackingMode::Hand);
        }
    }else if(holistic){
        add("signal_channel_face",sight::TrackingKind::Face,false);
        add("signal_channel_body",sight::TrackingKind::Body,false);
        add("signal_channel_hand",sight::TrackingKind::Hand,false);
    }else{
        const auto kind=id=="sight_face_motion_filter"?sight::TrackingKind::Face:
            id=="sight_body_motion_filter"?sight::TrackingKind::Body:
            id=="sight_hand_motion_filter"?sight::TrackingKind::Hand:sight::TrackingKind::WholeBody;
        add("signal_channel",kind,kind==sight::TrackingKind::Hand);
    }
    obs_data_release(settings);
}
inline bool discover_source(void* opaque,obs_source_t* source) {
    obs_source_enum_filters(source,discover_producer,opaque);
    return true;
}
inline ChannelChoices discover_channels(obs_source_t* context) {
    ChannelChoices choices;
    obs_enum_sources(discover_source,&choices);
    obs_enum_scenes(discover_source,&choices);
    if(auto* parent=context?obs_filter_get_parent(context):nullptr)
        discover_source(&choices,parent);
    return choices;
}
inline bool list_contains(obs_property_t* p,const std::string& value) {
    for(size_t i=0;i<obs_property_list_item_count(p);++i){
        const char* item=obs_property_list_item_string(p,i);
        if(item&&value==item)return true;
    }
    return false;
}
inline void retain_missing(obs_property_t* p,const std::string& value) {
    if(!value.empty()&&!list_contains(p,value)){
        const auto item=obs_property_list_add_string(p,(value+" (saved; unavailable)").c_str(),value.c_str());
        obs_property_list_item_disable(p,item,true);
    }
}
inline const char* kind_name(sight::TrackingKind kind) {
    switch(kind){
        case sight::TrackingKind::Face:return "Face";
        case sight::TrackingKind::Body:return "Body";
        case sight::TrackingKind::Hand:return "Hand";
        case sight::TrackingKind::WholeBody:return "WholeBody";
    }
    return "Unknown";
}

inline const char* status(const ChannelChoice& c,const std::string& key,
                          std::chrono::milliseconds max_age){
    TrackingSignal snapshot;
    const bool fresh=TrackingSignalBus::read(key,max_age,snapshot)&&snapshot.kind==c.kind;
    return c.ambiguous?"duplicate channel":!c.enabled?"disabled":
        !fresh?"waiting / idle":snapshot.data.landmarks.empty()?"no subject":"live";
}
inline unsigned wire_kind(TrackingKind kind){
    switch(kind){
        case TrackingKind::Face:return SIGHT_SNAPSHOT_FACE;
        case TrackingKind::Body:return SIGHT_SNAPSHOT_BODY;
        case TrackingKind::Hand:return SIGHT_SNAPSHOT_HAND;
        case TrackingKind::WholeBody:return SIGHT_SNAPSHOT_WHOLEBODY;
    }
    return 0;
}
inline std::string label(const std::string& key,const ChannelChoice& c,
                         std::chrono::milliseconds age){
    return key+" — "+c.owner+" ["+kind_name(c.kind)+"; "+status(c,key,age)+"]";
}
inline obs_property_t* add_picker(obs_properties_t* props,obs_source_t* context,
                                  const char* key,const char* title,
                                  const char* fallback,
                                  std::optional<TrackingKind> required={}){
    auto* list=obs_properties_add_list(props,key,title,OBS_COMBO_TYPE_LIST,OBS_COMBO_FORMAT_STRING);
    obs_property_list_add_string(list,"None","");
    auto* settings=context?obs_source_get_settings(context):nullptr;
    const std::string saved=settings?obs_data_get_string(settings,key):fallback;
    if(settings)obs_data_release(settings);
    for(const auto& [name,c]:discover_channels(context)){
        if(required&&*required!=c.kind)continue;
        const auto item=obs_property_list_add_string(list,label(name,c,std::chrono::milliseconds(750)).c_str(),name.c_str());
        if(c.ambiguous)obs_property_list_item_disable(list,item,true);
    }
    retain_missing(list,saved);
    return list;
}
// Caller owns the OBS data object. Metadata only; no landmark/texture
// copies are retained, no model launch, no external C++ ABI linkage.
inline void export_inventory(obs_source_t* context,obs_data_t* out){
    auto* rows=obs_data_array_create();
    for(const auto& [name,c]:discover_channels(context)){
        auto* row=obs_data_create();
        obs_data_set_string(row,"channel",name.c_str());
        obs_data_set_string(row,"label",label(name,c,std::chrono::milliseconds(750)).c_str());
        obs_data_set_int(row,"kind",wire_kind(c.kind));
        obs_data_set_bool(row,"enabled",c.enabled);
        obs_data_set_bool(row,"ambiguous",c.ambiguous);
        obs_data_set_bool(row,"gestures",c.gestures);
        obs_data_array_push_back(rows,row);
        obs_data_release(row);
    }
    obs_data_set_int(out,"version",1);
    obs_data_set_array(out,"channels",rows);
    obs_data_array_release(rows);
}
} // namespace sight::discovery
