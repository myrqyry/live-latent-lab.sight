#include <obs.h>
#include <obs-module.h>
#include <graphics/graphics.h>
#include <chrono>
#include <thread>
#include <cstdio>
#include <iostream>
#include <string>
#include <filesystem>
#include <vector>
#include "core/sight_snapshot_v1.h"
using namespace std::chrono_literals;
static bool read(obs_source_t* receiver,const std::string& channel,unsigned expected,unsigned count=0,int face_extras=-1,bool empty=false){
    sight_snapshot_v1 snap{};snap.abi_version=SIGHT_SNAPSHOT_ABI_V1;snap.struct_size=sizeof(snap);
    calldata_t d{};calldata_set_string(&d,"channel",channel.c_str());
    calldata_set_int(&d,"max_age_ms",5000);calldata_set_ptr(&d,"snapshot",&snap);
    bool ok=proc_handler_call(obs_source_get_proc_handler(receiver),"sight_tracking_read_v1",&d);
    bool available=calldata_bool(&d,"available");calldata_free(&d);
    return ok&&(expected?available&&snap.kind==expected&&snap.point_count>=count&&
        (!empty||snap.point_count==0)&&(face_extras<0||
        (face_extras? (snap.blendshape_count==52&&snap.has_head_transform):(!snap.blendshape_count&&!snap.has_head_transform))):!available);
}
static bool backend(obs_source_t*source,const char*expected){
    auto*props=obs_source_properties(source);
    if(!props)return false;
    auto*status=obs_properties_get(props,"sight_backend_status");
    auto*button=obs_properties_get(props,"sight_backend_refresh");
    bool ok=status&&button&&obs_property_button_clicked(button,source);
    const char*desc=status?obs_property_description(status):nullptr;
    ok=ok&&desc&&std::string(desc).find(expected)!=std::string::npos;
    auto*health=obs_properties_get(props,"sight_inference_health");
    const std::string health_text=health?obs_property_description(health):"missing";
    if(std::string(expected).find("Loaded")!=std::string::npos)
        ok=ok&&health_text.find("Recent inference")!=std::string::npos&&health_text.find("Successful:")!=std::string::npos;
    else if(std::string(expected)=="Tracking off")
        ok=ok&&health_text.find("unavailable")!=std::string::npos;
    std::cout<<"SIGHT_TRACKING_HEALTH "<<health_text<<"\n";
    if(!ok)fprintf(stderr,"backend expected %s, got %s\n",expected,desc?desc:"missing");
    obs_properties_destroy(props);return ok;
}
static bool inventory(obs_source_t* receiver,const std::string& channel,unsigned kind){
    auto* out=obs_data_create();calldata_t d{};calldata_set_ptr(&d,"inventory",out);
    bool ok=proc_handler_call(obs_source_get_proc_handler(receiver),"sight_tracking_list_v1",&d);
    auto* rows=obs_data_get_array(out,"channels");bool found=false;
    for(size_t i=0;rows&&i<obs_data_array_count(rows);++i){
        auto* r=obs_data_array_item(rows,i);
        if(channel==obs_data_get_string(r,"channel")&&obs_data_get_int(r,"kind")==kind)found=true;
        obs_data_release(r);
    }
    if(rows)obs_data_array_release(rows);obs_data_release(out);calldata_free(&d);return ok&&found;
}
static bool objects(obs_source_t* f,bool expect,int max_count=16){
    calldata_t d{};bool ok=proc_handler_call(obs_source_get_proc_handler(f),"sight_object_read_v1",&d);
    bool a=calldata_bool(&d,"available");
    auto n=calldata_int(&d,"count");calldata_free(&d);
    return ok&&(expect?a&&n>0&&n<=max_count:!a&&n==0);
}
static void render(obs_source_t* s){
    obs_enter_graphics();
    auto* tr=gs_texrender_create(GS_RGBA,GS_ZS_NONE);
    if(gs_texrender_begin(tr,512,512)){
        gs_ortho(0,512,0,512,-100,100);obs_source_video_render(s);gs_texrender_end(tr);
    }
    gs_texrender_destroy(tr);obs_leave_graphics();
}
template<class P>static bool wait(obs_source_t* s,P predicate,int seconds=12){
    auto end=std::chrono::steady_clock::now()+std::chrono::seconds(seconds);
    do{
        render(s);
        if(predicate())return true;
        std::this_thread::sleep_for(33ms);
    }while(std::chrono::steady_clock::now()<end);
    return false;
}
static void set_mode(obs_source_t* f,const char* mode){
    auto* d=obs_data_create();obs_data_set_string(d,"tracking_mode",mode);obs_source_update(f,d);obs_data_release(d);
}
int main(int argc,char** argv){
    if(argc!=14||!obs_startup("en-US",nullptr,nullptr))return 2;
    obs_video_info v{};v.graphics_module="libobs-opengl";v.fps_num=30;v.fps_den=1;
    v.base_width=v.output_width=512;v.base_height=v.output_height=512;v.output_format=VIDEO_FORMAT_RGBA;
    v.colorspace=VIDEO_CS_DEFAULT;v.range=VIDEO_RANGE_DEFAULT;v.scale_type=OBS_SCALE_DISABLE;
    if(obs_reset_video(&v)!=OBS_VIDEO_SUCCESS)return 3;
    obs_module_t* m=nullptr;
    if(obs_open_module(&m,argv[1],argv[2])!=MODULE_SUCCESS||!obs_init_module(m))return 4;
    const auto* name=obs_source_get_display_name("sight_unified_tracking_filter");
    if(!name||std::string(name)!="lll.sight Tracking"){
        puts("SIGHT_UNIFIED_REGISTRY=0");obs_shutdown();return 1;
    }
    for(const auto* id:{"sight_face_motion_filter","sight_body_motion_filter",
        "sight_hand_motion_filter","sight_holistic_motion_filter",
        "sight_object_tracking_filter","sight_rtmw_tracking_filter","sight_touch_field_filter"}){
        if(!obs_source_get_display_name(id)||!(obs_get_source_output_flags(id)&OBS_SOURCE_CAP_OBSOLETE)){
            printf("FAIL: missing or visible legacy compatibility type: %s\n",id);
            obs_shutdown();return 1;
        }
    }
    puts("SIGHT_UNIFIED_REGISTRY=1");
    obs_module_t* image=nullptr;
    if(obs_open_module(&image,argv[13],"/usr/local/share/obs/obs-plugins/image-source")!=MODULE_SUCCESS||!obs_init_module(image))return 5;
    auto* d=obs_data_create();obs_data_set_string(d,"file",argv[4]);
    auto* source=obs_source_create("image_source","Unified fixture",d,nullptr);obs_data_release(d);
    d=obs_data_create();obs_data_set_string(d,"tracking_mode","face");
    const char* modes[]={"face","body","hand","holistic","object","rtmw"};
    for(int i=0;i<6;++i){
        auto p=std::string(modes[i])+"_";
        obs_data_set_string(d,(p+"model_path").c_str(),argv[6+i]);
        obs_data_set_string(d,(p+"mediapipe_runtime").c_str(),argv[3]);
        obs_data_set_string(d,(p+"signal_channel").c_str(),("unified/"+std::string(modes[i])).c_str());
        obs_data_set_bool(d,(p+"draw_overlay").c_str(),false);
        obs_data_set_int(d,(p+"inference_cadence").c_str(),1);
    }
    obs_data_set_int(d,"face_temporal_blend",17);
    obs_data_set_int(d,"body_temporal_blend",61);
    obs_data_set_int(d,"body_detection_confidence",45);
    obs_data_set_int(d,"body_presence_confidence",45);
    obs_data_set_int(d,"body_tracking_confidence",55);
    obs_data_set_int(d,"hand_max_hands",1);
    obs_data_set_int(d,"holistic_face_detection_confidence",45);
    obs_data_set_int(d,"holistic_pose_detection_confidence",45);
    obs_data_set_int(d,"holistic_pose_suppression_threshold",35);
    obs_data_set_int(d,"object_max_objects",1);
    obs_data_set_string(d,"holistic_signal_channel_face","unified/holistic/face");
    obs_data_set_string(d,"holistic_signal_channel_body","unified/holistic/body");
    obs_data_set_string(d,"holistic_signal_channel_hand","unified/holistic/hand");
    obs_data_set_string(d,"rtmw_rtmw_person_detector",argv[12]);
    auto* f=obs_source_create("sight_unified_tracking_filter","Unified tracking",d,nullptr);
    obs_data_release(d);
    auto* receiver=obs_source_create_private("sight_tracking_receiver_filter","Receiver",nullptr);
    if(!source||!f||!receiver)return 6;
    obs_source_filter_add(source,f);obs_source_inc_showing(source);
    auto* p=obs_source_properties(f);
    d=obs_source_get_settings(f);
    auto* selector=obs_properties_get(p,"tracking_mode");
    bool props=selector&&obs_property_list_item_count(selector)>=6;
    for(int i=0;i<6&&props;++i){
        obs_data_set_string(d,"tracking_mode",modes[i]);obs_property_modified(selector,d);
        for(int j=0;j<6;++j){
            auto* field=obs_properties_get(p,(std::string(modes[j])+"_model_path").c_str());
            props=props&&field&&obs_property_visible(field)==(i==j);
        }
    }
    obs_data_set_string(d,"tracking_mode","face");obs_property_modified(selector,d);
    const char* native_fields[]={"face_detection_confidence","face_presence_confidence","face_tracking_confidence",
        "body_detection_confidence","hand_max_hands","holistic_face_detection_confidence","holistic_pose_suppression_threshold","rtmw_person_confidence"};
    bool native=true;
    for(const auto* key:native_fields)native=native&&obs_properties_get(p,key);
    obs_data_release(d);obs_properties_destroy(p);
    bool face=wait(source,[&]{return read(receiver,"unified/face",SIGHT_SNAPSHOT_FACE,478);});
    bool cpu_backend=backend(f,"Loaded — MediaPipe CPU");
    bool face_list=inventory(receiver,"unified/face",SIGHT_SNAPSHOT_FACE);
    const bool face_extras=read(receiver,"unified/face",SIGHT_SNAPSHOT_FACE,478,1);
    d=obs_data_create();obs_data_set_bool(d,"face_publish_blendshapes",false);obs_data_set_bool(d,"face_publish_head_transform",false);
    obs_source_update(f,d);obs_data_release(d);
    bool no_extras=wait(source,[&]{return read(receiver,"unified/face",SIGHT_SNAPSHOT_FACE,478,0);});
    d=obs_data_create();obs_data_set_bool(d,"face_publish_blendshapes",true);obs_data_set_bool(d,"face_publish_head_transform",true);
    obs_source_update(f,d);obs_data_release(d);
    bool extras_restored=wait(source,[&]{return read(receiver,"unified/face",SIGHT_SNAPSHOT_FACE,478,1);});
    set_mode(f,"body");
    bool body=wait(source,[&]{return read(receiver,"unified/body",SIGHT_SNAPSHOT_BODY,33)&&read(receiver,"unified/face",0);});
    bool body_list=inventory(receiver,"unified/body",SIGHT_SNAPSHOT_BODY)&&!inventory(receiver,"unified/face",SIGHT_SNAPSHOT_FACE);
    d=obs_data_create();obs_data_set_string(d,"file",argv[5]);obs_source_update(source,d);obs_data_release(d);
    set_mode(f,"hand");
    bool hand=wait(source,[&]{return read(receiver,"unified/hand",SIGHT_SNAPSHOT_HAND,21)&&read(receiver,"unified/body",0);});
    d=obs_data_create();obs_data_set_string(d,"file",argv[4]);obs_source_update(source,d);obs_data_release(d);
    set_mode(f,"holistic");
    bool holistic=wait(source,[&]{return read(receiver,"unified/holistic/face",SIGHT_SNAPSHOT_FACE,468)&&read(receiver,"unified/holistic/body",SIGHT_SNAPSHOT_BODY,33)&&read(receiver,"unified/hand",0);});
    bool triple=inventory(receiver,"unified/holistic/face",SIGHT_SNAPSHOT_FACE)&&inventory(receiver,"unified/holistic/body",SIGHT_SNAPSHOT_BODY)&&inventory(receiver,"unified/holistic/hand",SIGHT_SNAPSHOT_HAND);
    set_mode(f,"object");
    bool obj=wait(source,[&]{return objects(f,true,1)&&read(receiver,"unified/holistic/face",0)&&read(receiver,"unified/holistic/body",0)&&read(receiver,"unified/holistic/hand",0);});
    set_mode(f,"rtmw");
    bool rtmw=wait(source,[&]{return read(receiver,"unified/rtmw",SIGHT_SNAPSHOT_WHOLEBODY,133)&&objects(f,false);});
    bool gpu_backend=backend(f,"Loaded — LiteRT GPU");
    d=obs_data_create();obs_data_set_int(d,"rtmw_person_confidence",95);obs_source_update(f,d);obs_data_release(d);
    bool rejects=wait(source,[&]{return read(receiver,"unified/rtmw",SIGHT_SNAPSHOT_WHOLEBODY,0,-1,true);});
    d=obs_data_create();obs_data_set_int(d,"rtmw_person_confidence",35);obs_source_update(f,d);obs_data_release(d);
    bool reaccept=wait(source,[&]{return read(receiver,"unified/rtmw",SIGHT_SNAPSHOT_WHOLEBODY,133);});
    set_mode(f,"face");
    bool restore=wait(source,[&]{return read(receiver,"unified/face",SIGHT_SNAPSHOT_FACE,478)&&read(receiver,"unified/rtmw",0);});
    d=obs_source_get_settings(f);restore=restore&&obs_data_get_int(d,"face_temporal_blend")==17&&obs_data_get_int(d,"body_temporal_blend")==61;obs_data_release(d);
    d=obs_data_create();obs_data_set_string(d,"tracking_mode","body");
    obs_data_set_string(d,"body_model_path",argv[7]);
    obs_data_set_string(d,"body_mediapipe_runtime",argv[3]);
    obs_data_set_string(d,"body_signal_channel","unified/second");obs_data_set_bool(d,"body_draw_overlay",false);
    auto* second=obs_source_create("sight_unified_tracking_filter","Independent tracking",d,nullptr);obs_data_release(d);
    if(!second)return 7;
    obs_source_filter_add(source,second);
    bool multiple=wait(source,[&]{return read(receiver,"unified/face",SIGHT_SNAPSHOT_FACE,478)&&
        read(receiver,"unified/second",SIGHT_SNAPSHOT_BODY,33);});
    set_mode(f,"none");
    bool off=wait(source,[&]{return read(receiver,"unified/face",0)&&!inventory(receiver,"unified/face",SIGHT_SNAPSHOT_FACE)&&read(receiver,"unified/second",SIGHT_SNAPSHOT_BODY,33);});
    bool off_backend=backend(f,"Tracking off");
    set_mode(second,"none");
    bool second_off=wait(source,[&]{return read(receiver,"unified/second",0);});
    obs_source_filter_remove(source,second);obs_source_release(second);
    printf("SIGHT_UNIFIED_NATIVE_RUNTIME extras=%d disable=%d restore=%d rtmw_reject=%d rtmw_recover=%d multiple=%d second_off=%d\n",face_extras,no_extras,extras_restored,rejects,reaccept,multiple,second_off);
    printf("SIGHT_UNIFIED_NATIVE_FIELDS=%d\n",native);
    printf("SIGHT_UNIFIED_REAL props=%d face=%d list=%d body=%d body_list=%d hand=%d holistic=%d triple=%d object=%d rtmw=%d restore=%d off=%d\n",props,face,face_list,body,body_list,hand,holistic,triple,obj,rtmw,restore,off);
    obs_source_dec_showing(source);obs_source_filter_remove(source,f);obs_source_release(f);obs_source_release(receiver);obs_source_release(source);
    obs_wait_for_destroy_queue();obs_shutdown();
    return !(cpu_backend&&gpu_backend&&off_backend&&face_extras&&no_extras&&extras_restored&&rejects&&reaccept&&multiple&&second_off&&native&&props&&face&&face_list&&body&&body_list&&hand&&holistic&&triple&&obj&&rtmw&&restore&&off);
}
