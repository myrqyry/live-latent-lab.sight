#include "common/preferences.hpp"
#include <obs-module.h>
#include "core/analysis_capture.hpp"
#include <graphics/graphics.h>
#include <util/platform.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include "core/text_ocr.hpp"
#include "core/ocr_output.hpp"
#include "core/ocr_processing.hpp"
#include "core/sight_text_v1.h"
#ifdef SIGHT_HAVE_OCR_EDITOR
#include "ui/ocr-region-editor.hpp"
#include <QApplication>
#include <QThread>
#include <QMessageBox>
#endif

namespace {
using Clock=std::chrono::steady_clock;
std::atomic<uint64_t> text_sequence{0};
int64_t millis(Clock::time_point t) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(t.time_since_epoch()).count();
}
constexpr size_t RegionCount=4;
struct RegionConfig {
    std::string region_name,target_uuid,format;
    int rx=0,ry=0,rw=100,rh=100,stable=2;
    bool enabled=false,export_text=false,change_only=false;
    int preprocessing=0,page_mode=11,confidence=0,padding=8;
    bool operator==(const RegionConfig&) const=default;
};
struct Config {
    std::string directory,language;
    std::array<RegionConfig,RegionCount> regions{};
    bool preview=false;
    int edge=1280,interval=250;
    bool operator==(const Config&) const=default;
};
std::string region_key(size_t i,const char* field){
    return "ocr_r"+std::to_string(i+1)+"_"+field;
}
struct Frame {
    std::vector<uint8_t> rgba;
    unsigned w=0,h=0,sw=0,sh=0;
    uint64_t revision=0;
    Clock::time_point captured{};
};
struct Mask {
    std::vector<uint8_t> pixels;
    unsigned w=0,h=0,sw=0,sh=0;
    uint64_t revision=0,sequence=0;
    Clock::time_point captured{};
    bool valid=false,preview=false;
};
struct TextOutput {
    uint64_t revision=0,sequence=0;
    unsigned sw=0,sh=0;
    Clock::time_point captured{};
    std::string text;
};
struct Filter {
    obs_source_t* context=nullptr;
    std::mutex mutex;
    std::condition_variable cv;
    std::thread worker;
    Config config;
    Frame frame;
    Mask result;
    sight_text_v1 published{};
    std::array<TextOutput,RegionCount> outputs{};
    std::atomic<bool> route_queued{false};
    std::array<uint64_t,RegionCount> route_sequence{};
    std::atomic<uint64_t> recognition_runs{0},cache_hits{0};
#ifdef SIGHT_HAVE_OCR_EDITOR
    // Owned only while the editor requests a snapshot, never kept on disk.
    std::mutex editor_mutex;
    std::atomic<uint64_t> editor_request{0};
    uint64_t editor_ticket=0;
    Frame editor_frame;
    bool editor_open=false; // UI thread only
#endif
    Clock::time_point published_time{};
    uint64_t published_revision=0;
    std::atomic<int64_t> model_retry_at{0};
    bool pending=false,stopping=false;
    std::atomic<uint64_t> revision{1};
    std::atomic<bool> wants{true};
    std::atomic<int> view{0},block{24},age{2000},edge{1280},interval{250};
    gs_texrender_t* capture=nullptr;
    gs_stagesurf_t* stage=nullptr;
    gs_texture_t* mask=nullptr;
    gs_effect_t* effect=nullptr;
    unsigned cw=0,ch=0,sw=0,sh=0,uploaded_w=0,uploaded_h=0;
    uint64_t stage_revision=0,uploaded_sequence=0,uploaded_revision=0;
    Clock::time_point captured{},next_capture{},uploaded_time{};
    bool staged=false,uploaded_valid=false,announced=false,uploaded_preview=false;
};
std::string default_directory() {
    char* local=obs_module_config_path("models/tessdata");
    std::string directory=local?local:"";
    bfree(local);
    std::error_code ec;
    if(!directory.empty()&&std::filesystem::is_regular_file(
            std::filesystem::path(directory)/"eng.traineddata",ec))
        return directory;
    return "/usr/share/tesseract-ocr/5/tessdata";
}
Config read_config(obs_data_t* s) {
    Config c;
    c.directory=obs_data_get_string(s,"tessdata_path");
    c.language=obs_data_get_string(s,"ocr_language");
    c.edge=std::clamp(int(obs_data_get_int(s,"ocr_edge")),640,2048);
    c.interval=std::clamp(int(obs_data_get_int(s,"ocr_interval")),100,2000);
    c.preview=obs_data_get_int(s,"ocr_view")==4;
    const bool analysis=obs_data_get_int(s,"ocr_view")==3;
    for(size_t i=0;i<RegionCount;++i){
        auto& r=c.regions[i];
        auto key=[&](const char* field){return i?region_key(i,field):std::string("ocr_")+field;};
        auto integer=[&](const char* field){return obs_data_get_int(s,key(field).c_str());};
        auto flag=[&](const char* field){return obs_data_get_bool(s,key(field).c_str());};
        auto str=[&](const char* field){return std::string(obs_data_get_string(s,key(field).c_str()));};
        r.enabled=!i||flag("enabled");
        if(!r.enabled)continue;
        r.page_mode=int(integer("layout"));
        if(r.page_mode!=3&&r.page_mode!=6&&r.page_mode!=7&&r.page_mode!=11)r.page_mode=11;
        r.confidence=int(std::clamp<int64_t>(integer("confidence"),0,100));
        r.padding=int(std::clamp<int64_t>(integer("padding"),0,40));
        r.rx=int(std::clamp<int64_t>(integer("x"),0,99));
        r.ry=int(std::clamp<int64_t>(integer("y"),0,99));
        r.rw=int(std::clamp<int64_t>(integer("w"),1,100));
        r.rh=int(std::clamp<int64_t>(integer("h"),1,100));
        r.stable=int(std::clamp<int64_t>(integer("stable"),1,5));
        r.preprocessing=int(std::clamp<int64_t>(integer("preprocessing"),0,4));
        r.export_text=analysis&&flag("export");
        if(r.export_text){
            r.region_name=str(i?"name":"region_name");r.target_uuid=str("target_uuid");r.format=str("format");
            if(r.region_name.size()>128||r.target_uuid.size()>128||r.format.size()>512)r.export_text=false;
        }
        r.change_only=r.export_text&&flag("change_only");
    }
    return c;
}
bool route_unique(const Config& c,size_t i){
    const auto& r=c.regions[i];
    if(!r.enabled||!r.export_text||r.target_uuid.empty())return false;
    for(size_t j=0;j<RegionCount;++j)
        if(i!=j&&c.regions[j].enabled&&c.regions[j].export_text&&
           c.regions[j].target_uuid==r.target_uuid)return false;
    return true;
}
void worker(Filter* f) {
    sight::TextOcr ocr;
    std::string loaded_path,loaded_language;
    bool attempted=false,ready=false;
    struct RegionState {
        sight::OcrChangeGate change_gate;
        sight::TextResult cached;
        sight::OcrConsensus consensus;
        TextOutput output;
        void reset(){change_gate.reset();cached={};consensus.reset();output={};}
    };
    std::array<RegionState,RegionCount> states{};
    uint64_t state_revision=0;unsigned last_sw=0,last_sh=0,last_w=0,last_h=0;
    for(;;) {
        Frame frame;
        Config cfg;
        {
            std::unique_lock lock(f->mutex);
            f->cv.wait(lock,[&]{return f->stopping||f->pending;});
            if(f->stopping)return;
            frame=std::move(f->frame);f->pending=false;cfg=f->config;
        }
#ifdef SIGHT_HAVE_OCR_EDITOR
        if(const auto ticket=f->editor_request.load();ticket&&
            frame.rgba.size()==size_t(frame.w)*frame.h*4){
            // Copy on worker, not graphics thread. Cancel/refresh invalidates
            // an in-flight copy. One snapshot per explicit request.
            Frame preview=frame;
            std::lock_guard lock(f->editor_mutex);
            if(f->editor_request==ticket&&frame.revision==f->revision.load()){
                f->editor_frame=std::move(preview);f->editor_request=0;
            }
        }
#endif
        if(!attempted||!ready||loaded_path!=cfg.directory||loaded_language!=cfg.language) {
            loaded_path=cfg.directory;loaded_language=cfg.language;attempted=true;
            ready=ocr.load(loaded_path,loaded_language);
            blog(ready?LOG_INFO:LOG_WARNING,"[Sight/OCR] SIGHT_TEXT_OCR_20261117: local model %s",ready?"ready":"unavailable");
        }
        if(state_revision!=frame.revision||last_sw!=frame.sw||last_sh!=frame.sh||last_w!=frame.w||last_h!=frame.h){
            for(auto& state:states)state.reset();
            state_revision=frame.revision;last_sw=frame.sw;last_sh=frame.sh;last_w=frame.w;last_h=frame.h;
        }
        Mask out;
        out.w=frame.w;out.h=frame.h;out.sw=frame.sw;out.sh=frame.sh;
        out.revision=frame.revision;out.captured=frame.captured;out.sequence=++text_sequence;out.preview=cfg.preview;
        sight_text_v1 snapshot{};
        snapshot.abi_version=SIGHT_TEXT_ABI_V1;snapshot.struct_size=sizeof(snapshot);
        snapshot.source_width=frame.sw;snapshot.source_height=frame.sh;snapshot.sequence=out.sequence;
        if(ready&&frame.rgba.size()==size_t(frame.w)*frame.h*4) {
            std::vector<uint8_t> gray(size_t(frame.w)*frame.h);
            for(size_t i=0;i<gray.size();++i) {
                const auto* p=frame.rgba.data()+i*4;
                gray[i]=uint8_t((77u*p[0]+150u*p[1]+29u*p[2]+128u)>>8);
            }
            out.valid=true;out.pixels.assign(gray.size(),0);
            for(size_t i=0;i<RegionCount;++i){
                const auto& r=cfg.regions[i];auto& state=states[i];
                if(!r.enabled){state.reset();continue;}
                // The capture is shared, while cache and consensus belong to each region.
                const auto roi=sight::ocr_region(frame.w,frame.h,r.rx,r.ry,r.rw,r.rh);
                auto cropped=sight::ocr_crop(gray,frame.w,frame.h,roi);
                const bool reused=r.change_only&&state.change_gate.reuse(cropped,millis(frame.captured));
                auto processed=reused?std::vector<uint8_t>{}:sight::ocr_preprocess(cropped,roi.w,roi.h,r.preprocessing);
                sight::TextResult text;
                if(reused){text=state.cached;++f->cache_hits;}
                else {text=ocr.recognize(processed,roi.w,roi.h,r.page_mode);++f->recognition_runs;}
                if(!text.valid){out.valid=false;state.reset();continue;}
                if(r.change_only&&!reused){state.cached=text;state.change_gate.reset();}
                std::string reading;
                for(const auto& line:text.lines){
                    if(line.confidence<r.confidence)continue;
                    if(r.export_text){
                        auto clean=sight::ocr_clean(line.text);
                        if(!clean.empty()){if(!reading.empty())reading+='\n';reading+=clean;}
                    }
                    const int x0=std::max(int(roi.x),int(roi.x)+line.x0-r.padding);
                    const int y0=std::max(int(roi.y),int(roi.y)+line.y0-r.padding);
                    const int x1=std::min(int(roi.x+roi.w),int(roi.x)+line.x1+r.padding);
                    const int y1=std::min(int(roi.y+roi.h),int(roi.y)+line.y1+r.padding);
                    if(snapshot.count<SIGHT_TEXT_MAX_LINES){
                        auto& box=snapshot.lines[snapshot.count++];
                        box={float(x0)/frame.w,float(y0)/frame.h,float(x1)/frame.w,float(y1)/frame.h,
                             std::clamp(line.confidence/100.f,0.f,1.f)};
                    }else out.valid=false;
                    if(!cfg.preview)for(int y=y0;y<y1;++y)
                        std::fill(out.pixels.begin()+size_t(y)*frame.w+x0,
                                  out.pixels.begin()+size_t(y)*frame.w+x1,uint8_t(255));
                }
                if(cfg.preview)for(unsigned y=0;y<roi.h;++y)
                    std::copy_n(processed.begin()+size_t(y)*roi.w,roi.w,
                        out.pixels.begin()+size_t(roi.y+y)*frame.w+roi.x);
                if(r.export_text&&reading.size()<=8192){
                    if(reused||state.consensus.observe(reading,unsigned(r.stable))){
                        state.output={frame.revision,out.sequence,frame.sw,frame.sh,frame.captured,state.consensus.value()};
                        if(r.change_only&&!reused)state.change_gate.confirm(cropped,millis(frame.captured));
                    }
                }else state.reset();
            }
        }else for(auto& state:states)state.reset();
        {
            std::lock_guard lock(f->mutex);
            if(f->revision==frame.revision) {
                for(size_t i=0;i<RegionCount;++i)f->outputs[i]=states[i].output;
                f->published=out.valid?snapshot:sight_text_v1{};
                f->published_time=frame.captured;f->published_revision=frame.revision;
                f->result=std::move(out);
                f->model_retry_at=ready?0:millis(Clock::now()+std::chrono::seconds(5));
            }
            f->wants=true;
        }
    }
}
void defaults(obs_data_t* s) {
    obs_data_set_default_string(s,"tessdata_path",default_directory().c_str());
    obs_data_set_default_string(s,"ocr_language","eng");
    obs_data_set_default_int(s,"ocr_layout",11);
    obs_data_set_default_int(s,"ocr_edge",1280);
    obs_data_set_default_int(s,"ocr_padding",8);
    obs_data_set_default_int(s,"ocr_confidence",0);
    obs_data_set_default_int(s,"ocr_interval",250);
    obs_data_set_default_int(s,"ocr_age",2000);
    obs_data_set_default_int(s,"ocr_view",0);
    obs_data_set_default_int(s,"ocr_block",24);
    obs_data_set_default_int(s,"ocr_preprocessing",0);
    obs_data_set_default_bool(s,"ocr_change_only",false);
    obs_data_set_default_int(s,"ocr_x",0);obs_data_set_default_int(s,"ocr_y",0);
    obs_data_set_default_int(s,"ocr_w",100);obs_data_set_default_int(s,"ocr_h",100);
    obs_data_set_default_bool(s,"ocr_export",false);
    obs_data_set_default_int(s,"ocr_stable",2);
    obs_data_set_default_string(s,"ocr_region_name","Text");
    obs_data_set_default_string(s,"ocr_format","{text}");
    for(size_t i=1;i<RegionCount;++i){
        auto key=[&](const char* field){return region_key(i,field);};
        obs_data_set_default_bool(s,key("enabled").c_str(),false);
        obs_data_set_default_string(s,key("name").c_str(),("Region "+std::to_string(i+1)).c_str());
        obs_data_set_default_int(s,key("x").c_str(),0);obs_data_set_default_int(s,key("y").c_str(),0);
        obs_data_set_default_int(s,key("w").c_str(),100);obs_data_set_default_int(s,key("h").c_str(),100);
        obs_data_set_default_int(s,key("layout").c_str(),11);
        obs_data_set_default_int(s,key("confidence").c_str(),0);obs_data_set_default_int(s,key("padding").c_str(),8);
        obs_data_set_default_int(s,key("preprocessing").c_str(),0);
        obs_data_set_default_int(s,key("stable").c_str(),2);
        obs_data_set_default_string(s,key("format").c_str(),"{text}");
    }
}
void update(void* data,obs_data_t* s) {
    auto* f=static_cast<Filter*>(data);
    auto cfg=read_config(s);
    f->view=std::clamp(int(obs_data_get_int(s,"ocr_view")),0,4);
    f->block=std::clamp(int(obs_data_get_int(s,"ocr_block")),4,80);
    f->age=std::clamp(int(obs_data_get_int(s,"ocr_age")),500,5000);
    f->edge=cfg.edge;f->interval=cfg.interval;
    std::lock_guard lock(f->mutex);
    if(!(cfg==f->config)) {
        f->config=std::move(cfg);f->outputs={};
        ++f->revision;
        f->model_retry_at=0;
    }
}
void read_text(void* data,calldata_t* args) {
    auto* f=static_cast<Filter*>(data);
    calldata_set_bool(args,"available",false);
    void* buffer=nullptr;
    long long age=0;
    if(!calldata_get_ptr(args,"snapshot",&buffer)||!buffer||
       !calldata_get_int(args,"max_age_ms",&age)||age<1||age>5000)return;
    auto* out=static_cast<sight_text_v1*>(buffer);
    if(out->abi_version!=SIGHT_TEXT_ABI_V1||out->struct_size!=sizeof(*out))return;
    out->sequence=0;out->count=0;
    std::unique_lock lock(f->mutex,std::try_to_lock);
    if(!lock||!obs_source_enabled(f->context)||!f->published.sequence||
       f->published_revision!=f->revision.load()||
       Clock::now()-f->published_time>std::chrono::milliseconds(age))return;
    *out=f->published;
    calldata_set_bool(args,"available",true);
}
// Caller holds the worker mutex. Never export stale or detached text.
bool output_ready(Filter* f,size_t i,int age,unsigned w,unsigned h){
    const auto& r=f->config.regions[i];const auto& out=f->outputs[i];
    return r.enabled&&r.export_text&&f->view==3&&out.sequence&&
        out.revision==f->revision.load()&&out.sw==w&&out.sh==h&&
        Clock::now()-out.captured<=std::chrono::milliseconds(age);
}
void read_output_slot(void* data,calldata_t* args,bool indexed){
    auto* f=static_cast<Filter*>(data);
    calldata_set_bool(args,"available",false);calldata_set_string(args,"text","");
    calldata_set_string(args,"region","");calldata_set_int(args,"sequence",0);
    long long age=0,slot=1;
    if(indexed&&(!calldata_get_int(args,"slot",&slot)||slot<1||slot>static_cast<long long>(RegionCount)))return;
    if(!calldata_get_int(args,"max_age_ms",&age)||age<1||age>5000||!obs_source_enabled(f->context))return;
    auto* target=obs_filter_get_target(f->context);
    if(!target)return;
    auto* live=obs_source_get_settings(f->context);
    if(!live)return;
    const auto cfg=read_config(live);obs_data_release(live);
    const size_t i=size_t(slot-1);
    if(!cfg.regions[i].export_text)return;
    const auto w=obs_source_get_base_width(target),h=obs_source_get_base_height(target);
    std::unique_lock lock(f->mutex,std::try_to_lock);
    if(!lock||cfg!=f->config||!output_ready(f,i,int(age),w,h))return;
    calldata_set_string(args,"text",f->outputs[i].text.c_str());
    calldata_set_string(args,"region",f->config.regions[i].region_name.c_str());
    calldata_set_int(args,"sequence",static_cast<long long>(f->outputs[i].sequence));
    calldata_set_bool(args,"available",true);
}
void read_output(void* data,calldata_t* args){read_output_slot(data,args,false);}
void read_region(void* data,calldata_t* args){read_output_slot(data,args,true);}
bool is_text_source(obs_source_t* s){
    if(!s)return false;
    std::string_view id=obs_source_get_unversioned_id(s);
    return id=="text_ft2_source"||id=="text_gdiplus";
}
struct RouteTask {Filter* filter;obs_source_t* retained;};
void route_on_ui(void* data){
    std::unique_ptr<RouteTask> task(static_cast<RouteTask*>(data));
    auto* f=task->filter;
    auto* source=obs_filter_get_target(f->context);
    struct Write {std::string uuid,text;};
    std::vector<Write> writes;
    uint64_t revision=0;
    auto* live=obs_source_get_settings(f->context);
    if(live&&source&&obs_filter_get_parent(f->context)&&obs_source_enabled(f->context)){
        const auto cfg=read_config(live);
        auto w=obs_source_get_base_width(source),h=obs_source_get_base_height(source);
        std::lock_guard lock(f->mutex);
        if(cfg==f->config){
            revision=f->revision.load();
            for(size_t i=0;i<RegionCount;++i){
                if(!route_unique(cfg,i)||!output_ready(f,i,f->age.load(),w,h))continue;
                const auto& r=cfg.regions[i];
                auto text=sight::ocr_format(r.format,r.region_name,f->outputs[i].text);
                if(text)writes.push_back({r.target_uuid,std::move(*text)});
            }
        }
    }
    if(live)obs_data_release(live);
    for(const auto& write:writes){
        if(revision!=f->revision.load()||f->view!=3||!obs_source_enabled(f->context))break;
        auto* dest=obs_get_source_by_uuid(write.uuid.c_str());
        if(is_text_source(dest)&&dest!=obs_filter_get_parent(f->context)&&!obs_source_removed(dest)){
            auto* cfg=obs_source_get_settings(dest);
            if(cfg&&!obs_data_get_bool(cfg,"from_file")&&!obs_data_get_bool(cfg,"read_from_file")&&
               write.text!=obs_data_get_string(cfg,"text")){
                auto* next=obs_data_create();obs_data_set_string(next,"text",write.text.c_str());
                obs_source_update(dest,next);obs_data_release(next);
            }
            if(cfg)obs_data_release(cfg);
        }
        if(dest)obs_source_release(dest);
    }
    f->route_queued=false;
    obs_source_release(task->retained);
}
void route_if_ready(Filter* f,unsigned w,unsigned h){
    if(f->route_queued.load()||f->view!=3)return;
    std::array<uint64_t,RegionCount> sequences{};
    bool changed=false;
    {
        std::unique_lock lock(f->mutex,std::try_to_lock);
        if(!lock)return;
        for(size_t i=0;i<RegionCount;++i)
            if(route_unique(f->config,i)&&output_ready(f,i,f->age.load(),w,h)&&
               f->route_sequence[i]!=f->outputs[i].sequence){
                sequences[i]=f->outputs[i].sequence;changed=true;
            }
    }
    if(!changed)return;
    if(auto* ref=obs_source_get_ref(f->context)){
        for(size_t i=0;i<RegionCount;++i)if(sequences[i])f->route_sequence[i]=sequences[i];
        f->route_queued=true;
        obs_queue_task(OBS_TASK_UI,route_on_ui,new RouteTask{f,ref},false);
    }
}
void read_stats(void* data,calldata_t* args){
    auto* f=static_cast<Filter*>(data);
    calldata_set_int(args,"recognitions",static_cast<long long>(f->recognition_runs.load()));
    calldata_set_int(args,"cache_hits",static_cast<long long>(f->cache_hits.load()));
}
void* create(obs_data_t* s,obs_source_t* context) {
    auto* f=new Filter;f->context=context;update(f,s);
    obs_enter_graphics();
    char* path=obs_module_file("sight-text-obfuscation.effect");
    if(path) {
        char* error=nullptr;
        f->effect=gs_effect_create_from_file(path,&error);
        if(error){blog(LOG_ERROR,"[Sight/OCR] effect: %s",error);bfree(error);}
        bfree(path);
    }
    obs_leave_graphics();
    if(auto* procs=obs_source_get_proc_handler(context)){
        proc_handler_add(procs,"void sight_text_read_v1(in int max_age_ms, in ptr snapshot, out bool available)",read_text,f);
        proc_handler_add(procs,"void sight_ocr_output_v1(in int max_age_ms, out string text, out string region, out int sequence, out bool available)",read_output,f);
        proc_handler_add(procs,"void sight_ocr_region_v1(in int max_age_ms, in int slot, out string text, out string region, out int sequence, out bool available)",read_region,f);
        proc_handler_add(procs,"void sight_ocr_stats_v1(out int recognitions, out int cache_hits)",read_stats,f);
    }
    f->worker=std::thread(worker,f);
    return f;
}
void destroy(void* data) {
    auto* f=static_cast<Filter*>(data);
    {std::lock_guard lock(f->mutex);f->stopping=true;}
    f->cv.notify_one();if(f->worker.joinable())f->worker.join();
    obs_enter_graphics();
    if(f->stage)gs_stagesurface_destroy(f->stage);
    if(f->capture)gs_texrender_destroy(f->capture);
    if(f->mask)gs_texture_destroy(f->mask);
    if(f->effect)gs_effect_destroy(f->effect);
    obs_leave_graphics();delete f;
}
#ifdef SIGHT_HAVE_OCR_EDITOR
sight::EditorRegions editor_regions(obs_data_t* settings){
    sight::EditorRegions regions;
    for(size_t i=0;i<RegionCount;++i){
        auto key=[&](const char* field){return i?region_key(i,field):std::string("ocr_")+field;};
        auto integer=[&](const char* field){return int(std::clamp<int64_t>(obs_data_get_int(settings,key(field).c_str()),0,100));};
        auto& r=regions[i];
        r.rect=sight::ocr_editor_clamp({integer("x"),integer("y"),integer("w"),integer("h")});
        r.enabled=!i||obs_data_get_bool(settings,key("enabled").c_str());
        r.name=QString::fromUtf8(obs_data_get_string(settings,key(i?"name":"region_name").c_str()));
    }
    return regions;
}
bool open_editor(obs_properties_t*,obs_property_t*,void* data){
    auto* f=static_cast<Filter*>(data);
    auto* app=qobject_cast<QApplication*>(QCoreApplication::instance());
    if(!f||!app||app->thread()!=QThread::currentThread()||f->editor_open)return false;
    auto* retained=obs_source_get_ref(f->context);
    if(!retained)return false;
    auto* target=obs_filter_get_target(retained);
    auto* target_ref=target?obs_source_get_ref(target):nullptr;
    auto* settings=obs_source_get_settings(retained);
    if(!target_ref||!settings){
        if(settings)obs_data_release(settings);if(target_ref)obs_source_release(target_ref);
        obs_source_release(retained);return false;
    }
    const auto sw=obs_source_get_base_width(target_ref),sh=obs_source_get_base_height(target_ref);
    const auto initial=editor_regions(settings);obs_data_release(settings);
    if(!sw||!sh){obs_source_release(target_ref);obs_source_release(retained);return false;}
    f->editor_open=true;
    blog(LOG_INFO,"[Sight/OCR] SIGHT_OCR_EDITOR_20261125_V2 opened");
    auto selected=sight::edit_ocr_regions(QApplication::activeWindow(),initial,sw,sh,
        [f](){
            std::lock_guard lock(f->editor_mutex);
            f->editor_frame={};f->editor_request=++f->editor_ticket;
        },
        [f]()->std::optional<sight::EditorPreview>{
            Frame frame;
            {
                std::unique_lock lock(f->editor_mutex,std::try_to_lock);
                if(!lock||f->editor_frame.rgba.empty())return {};
                frame=std::move(f->editor_frame);f->editor_frame={};
            }
            // QImage takes an owned copy on UI thread; no borrowed worker buffer.
            QImage image(frame.rgba.data(),int(frame.w),int(frame.h),int(frame.w*4),QImage::Format_RGBA8888);
            return sight::EditorPreview{image.copy(),frame.sw,frame.sh};
        });
    {
        std::lock_guard lock(f->editor_mutex);
        f->editor_request=0;f->editor_frame={};
    }
    f->editor_open=false;
    bool changed=false;
    if(selected&&*selected!=initial){
        auto* live=obs_source_get_settings(retained);
        // Refuse to apply a rectangle to a replaced/resized source or
        // overwrite region edits that happened while the modal was open.
        if(live&&obs_filter_get_target(retained)==target_ref&&obs_filter_get_parent(retained)&&
           obs_source_get_base_width(target_ref)==sw&&obs_source_get_base_height(target_ref)==sh&&
           editor_regions(live)==initial){
            auto* patch=obs_data_create();
            for(size_t i=0;i<RegionCount;++i){
                const auto& r=(*selected)[i];
                if(r==initial[i])continue;
                auto key=[&](const char* field){return i?region_key(i,field):std::string("ocr_")+field;};
                if(i&&r.enabled!=initial[i].enabled)obs_data_set_bool(patch,key("enabled").c_str(),r.enabled);
                if(r.name!=initial[i].name)obs_data_set_string(patch,key(i?"name":"region_name").c_str(),r.name.toUtf8().constData());
                if(r.rect!=initial[i].rect){
                    obs_data_set_int(patch,key("x").c_str(),r.rect.x);obs_data_set_int(patch,key("y").c_str(),r.rect.y);
                    obs_data_set_int(patch,key("w").c_str(),r.rect.w);obs_data_set_int(patch,key("h").c_str(),r.rect.h);
                }
            }
            obs_source_update(retained,patch);obs_data_release(patch);changed=true;
        }else{
            QMessageBox::warning(QApplication::activeWindow(),QObject::tr("lll.sight · OCR regions"),
                QObject::tr("The source or region settings changed while editing. Reopen the editor to try again."));
        }
        if(live)obs_data_release(live);
    }
    obs_source_release(target_ref);obs_source_release(retained);
    // OBS handles the property refresh when this callback returns.
    // Never call obs_source_update_properties inside the callback.
    return changed;
}
#endif
obs_properties_t* properties(void* data) {
    auto* p=obs_properties_create();
    auto* path=obs_properties_add_path(p,"tessdata_path","Tesseract model folder",OBS_PATH_DIRECTORY,nullptr,sight_preferences::browse(default_directory()).c_str());
    obs_property_set_long_description(path,"Folder containing language .traineddata files. OCR runs locally.");
    obs_properties_add_text(p,"ocr_language","OCR language (e.g. eng, eng+spa)",OBS_TEXT_DEFAULT);
    auto* layout=obs_properties_add_list(p,"ocr_layout","Text layout",OBS_COMBO_TYPE_LIST,OBS_COMBO_FORMAT_INT);
    obs_property_list_add_int(layout,"Scattered screen text",11);
    obs_property_list_add_int(layout,"Automatic page",3);
    obs_property_list_add_int(layout,"Single block",6);
    obs_property_list_add_int(layout,"Single line",7);
    auto* view=obs_properties_add_list(p,"ocr_view","Treatment",OBS_COMBO_TYPE_LIST,OBS_COMBO_FORMAT_INT);
    obs_property_list_add_int(view,"Opaque black",0);
    obs_property_list_add_int(view,"Mosaic (cosmetic)",1);
    obs_property_list_add_int(view,"Mask preview",2);
    obs_property_list_add_int(view,"Original source",3);
    obs_property_list_add_int(view,"OCR input preview",4);
    obs_property_set_long_description(view,"Best-effort text obscuring. Missed, changed or stale text may remain visible; missing OCR passes through live video. Not a privacy guarantee.");
    obs_properties_add_int_slider(p,"ocr_edge","OCR input longest edge",640,2048,128);
    obs_properties_add_int_slider(p,"ocr_padding","Line padding (OCR pixels)",0,40,1);
    obs_properties_add_int_slider(p,"ocr_confidence","Minimum line confidence",0,100,1);
    obs_properties_add_int_slider(p,"ocr_interval","Minimum capture interval (ms)",100,2000,50);
    obs_properties_add_int_slider(p,"ocr_age","Maximum mask age (ms)",500,5000,100);
    obs_properties_add_int_slider(p,"ocr_block","Mosaic cell size (pixels)",4,80,2);
    auto* preprocess=obs_properties_add_list(p,"ocr_preprocessing","OCR preprocessing",OBS_COMBO_TYPE_LIST,OBS_COMBO_FORMAT_INT);
    obs_property_list_add_int(preprocess,"Grayscale (default)",0);
    obs_property_list_add_int(preprocess,"Invert",1);
    obs_property_list_add_int(preprocess,"Contrast stretch",2);
    obs_property_list_add_int(preprocess,"Automatic threshold (Otsu)",3);
    obs_property_list_add_int(preprocess,"Adaptive local threshold",4);
#ifdef SIGHT_HAVE_OCR_EDITOR
    if(data&&qobject_cast<QApplication*>(QCoreApplication::instance())){
        auto* editor=obs_properties_add_button2(p,"ocr_visual_editor","Edit regions visually…",open_editor,data);
        obs_property_set_long_description(editor,"Select all four regions using a frozen snapshot from the existing OCR capture. No additional capture loop or text export.");
    }
#else
    (void)data;
#endif
    auto* region=obs_properties_create();
    obs_properties_add_int_slider(region,"ocr_x","Left (%)",0,99,1);
    obs_properties_add_int_slider(region,"ocr_y","Top (%)",0,99,1);
    obs_properties_add_int_slider(region,"ocr_w","Width (%)",1,100,1);
    obs_properties_add_int_slider(region,"ocr_h","Height (%)",1,100,1);
    obs_properties_add_group(p,"ocr_region_group","Recognition region",OBS_GROUP_NORMAL,region);
    auto* output=obs_properties_create();
    auto* opt=obs_properties_add_bool(output,"ocr_export","Enable recognized text output (original source only)");
    obs_property_set_long_description(opt,"Opt-in: recognized text can be read by local plugins or copied into an OBS text source, which can be saved with the scene. Never enable for sensitive content.");
    obs_properties_add_text(output,"ocr_region_name","Region name",OBS_TEXT_DEFAULT);
    obs_properties_add_int_slider(output,"ocr_stable","Matching readings required",1,5,1);
    auto* change=obs_properties_add_bool(output,"ocr_change_only","Reuse identical confirmed input");
    obs_property_set_long_description(change,"Analysis-only opt-in. Exact crop comparison, with a fresh OCR pass at least every two seconds. Cached readings never count toward initial consensus.");
    auto* targets=obs_properties_add_list(output,"ocr_target_uuid","Output text source",OBS_COMBO_TYPE_LIST,OBS_COMBO_FORMAT_STRING);
    obs_property_list_add_string(targets,"None (local snapshot only)","");
    obs_enum_sources([](void* param,obs_source_t* src){
        if(is_text_source(src))obs_property_list_add_string(static_cast<obs_property_t*>(param),obs_source_get_name(src),obs_source_get_uuid(src));
        return true;
    },targets);
    auto* format=obs_properties_add_text(output,"ocr_format","Output format",OBS_TEXT_DEFAULT);
    obs_property_set_long_description(format,"Use {text} and {region}. Confirmed empty recognition clears the destination. Disabling output leaves its last text unchanged.");
    obs_properties_add_group(p,"ocr_output_group","Text output",OBS_GROUP_NORMAL,output);
    for(size_t i=1;i<RegionCount;++i){
        auto* group=obs_properties_create();
        auto key=[&](const char* field){return region_key(i,field);};
        obs_properties_add_text(group,key("name").c_str(),"Region name",OBS_TEXT_DEFAULT);
        obs_properties_add_int_slider(group,key("x").c_str(),"Left (%)",0,99,1);
        obs_properties_add_int_slider(group,key("y").c_str(),"Top (%)",0,99,1);
        obs_properties_add_int_slider(group,key("w").c_str(),"Width (%)",1,100,1);
        obs_properties_add_int_slider(group,key("h").c_str(),"Height (%)",1,100,1);
        auto* layout=obs_properties_add_list(group,key("layout").c_str(),"Text layout",OBS_COMBO_TYPE_LIST,OBS_COMBO_FORMAT_INT);
        for(auto [label,id]:std::initializer_list<std::pair<const char*,int>>{{"Scattered screen text",11},{"Automatic page",3},{"Single block",6},{"Single line",7}})
            obs_property_list_add_int(layout,label,id);
        auto* prep=obs_properties_add_list(group,key("preprocessing").c_str(),"Preprocessing",OBS_COMBO_TYPE_LIST,OBS_COMBO_FORMAT_INT);
        for(auto [label,id]:std::initializer_list<std::pair<const char*,int>>{{"Grayscale",0},{"Invert",1},{"Contrast stretch",2},{"Otsu",3},{"Adaptive local threshold",4}})
            obs_property_list_add_int(prep,label,id);
        obs_properties_add_int_slider(group,key("confidence").c_str(),"Minimum confidence",0,100,1);
        obs_properties_add_int_slider(group,key("padding").c_str(),"Line padding",0,40,1);
        auto* export_prop=obs_properties_add_bool(group,key("export").c_str(),"Enable recognized text output (original source only)");
        obs_property_set_long_description(export_prop,"Opt-in for this region. Text routed to OBS can be saved with the scene.");
        obs_properties_add_int_slider(group,key("stable").c_str(),"Matching readings required",1,5,1);
        obs_properties_add_bool(group,key("change_only").c_str(),"Reuse identical confirmed input");
        auto* dest=obs_properties_add_list(group,key("target_uuid").c_str(),"Output text source",OBS_COMBO_TYPE_LIST,OBS_COMBO_FORMAT_STRING);
        obs_property_list_add_string(dest,"None (local snapshot only)","");
        obs_enum_sources([](void* param,obs_source_t* src){
            if(is_text_source(src))obs_property_list_add_string(static_cast<obs_property_t*>(param),obs_source_get_name(src),obs_source_get_uuid(src));
            return true;
        },dest);
        obs_properties_add_text(group,key("format").c_str(),"Output format",OBS_TEXT_DEFAULT);
        obs_properties_add_group(p,key("enabled").c_str(),("Additional region "+std::to_string(i+1)).c_str(),OBS_GROUP_CHECKABLE,group);
    }
    return p;
}
void render(void* data,gs_effect_t*) {
    auto* f=static_cast<Filter*>(data);
    auto* target=obs_filter_get_target(f->context);
    unsigned w=target?obs_source_get_base_width(target):0,h=target?obs_source_get_base_height(target):0;
    if(!w||!h||!f->effect){obs_source_skip_video_filter(f->context);return;}
    const auto now=Clock::now();
    const auto rev=f->revision.load();
    if(f->staged) {
        bool submitted=false;
        if(f->stage_revision==rev&&f->sw==w&&f->sh==h) {
            uint8_t* bytes=nullptr;uint32_t pitch=0;
            if(gs_stagesurface_map(f->stage,&bytes,&pitch)) {
                if(pitch>=f->cw*4) {
                    Frame frame;
                    frame.w=f->cw;frame.h=f->ch;frame.sw=w;frame.sh=h;
                    frame.revision=rev;frame.captured=f->captured;
                    frame.rgba.resize(size_t(f->cw)*f->ch*4);
                    for(unsigned y=0;y<f->ch;++y)
                        std::memcpy(frame.rgba.data()+size_t(y)*f->cw*4,bytes+size_t(y)*pitch,size_t(f->cw)*4);
                    std::unique_lock lock(f->mutex,std::try_to_lock);
                    if(lock&&f->revision==rev&&!f->pending) {
                        f->frame=std::move(frame);f->pending=true;submitted=true;f->cv.notify_one();
                    }
                }
                gs_stagesurface_unmap(f->stage);
            }
        }
        f->staged=false;
        if(!submitted)f->wants=true;
    }
    if(now>=f->next_capture&&millis(now)>=f->model_retry_at.load()&&f->wants.exchange(false)) {
        const unsigned edge=std::min(unsigned(f->edge.load()),std::max(w,h));
        unsigned cw=std::max(1u,unsigned(uint64_t(w)*edge/std::max(w,h)));
        unsigned ch=std::max(1u,unsigned(uint64_t(h)*edge/std::max(w,h)));
        if(!f->capture)f->capture=gs_texrender_create(GS_RGBA,GS_ZS_NONE);
        if(!f->stage||f->cw!=cw||f->ch!=ch) {
            if(f->stage)gs_stagesurface_destroy(f->stage);
            f->stage=gs_stagesurface_create(cw,ch,GS_RGBA);f->cw=cw;f->ch=ch;
        }
        auto* base=obs_get_base_effect(OBS_EFFECT_DEFAULT);
        if(f->capture&&f->stage&&base) {
            gs_texrender_reset(f->capture);
            if(gs_texrender_begin(f->capture,cw,ch)) {
                gs_projection_push();gs_matrix_push();gs_matrix_identity();
                gs_ortho(0,float(w),0,float(h),-100,100);
                struct vec4 clear;vec4_zero(&clear);gs_clear(GS_CLEAR_COLOR,&clear,0,0);
                gs_blend_state_push();gs_blend_function(GS_BLEND_ONE,GS_BLEND_ZERO);
                sight_draw_analysis_source(target);
                gs_blend_state_pop();
                gs_matrix_pop();gs_projection_pop();gs_texrender_end(f->capture);
                if(auto* tex=gs_texrender_get_texture(f->capture)) {
                    gs_stage_texture(f->stage,tex);
                    f->sw=w;f->sh=h;f->staged=true;f->stage_revision=rev;f->captured=now;
                }
            }
        }
        f->next_capture=now+std::chrono::milliseconds(f->interval.load());
        if(!f->staged)f->wants=true;
    }
    Mask result;
    {
        std::unique_lock lock(f->mutex,std::try_to_lock);
        if(lock&&f->result.sequence&&f->result.sequence!=f->uploaded_sequence) {
            result=std::move(f->result);
            f->result.sequence=0;
        }
    }
    if(result.sequence) {
        f->uploaded_sequence=result.sequence;f->uploaded_valid=false;
        if(result.valid&&result.revision==rev&&result.sw==w&&result.sh==h&&
           result.pixels.size()==size_t(result.w)*result.h) {
            if(!f->mask||gs_texture_get_width(f->mask)!=result.w||gs_texture_get_height(f->mask)!=result.h) {
                if(f->mask)gs_texture_destroy(f->mask);
                f->mask=gs_texture_create(result.w,result.h,GS_R8,1,nullptr,GS_DYNAMIC);
            }
            if(f->mask) {
                gs_texture_set_image(f->mask,result.pixels.data(),result.w,false);
                f->uploaded_valid=true;f->uploaded_revision=rev;f->uploaded_time=result.captured;
                f->uploaded_w=w;f->uploaded_h=h;f->uploaded_preview=result.preview;
                // Store source geometry independently of the next queued capture.
                if(!f->announced){blog(LOG_INFO,"[Sight/OCR] SIGHT_TEXT_OBFUSCATION_GPU_20261117");f->announced=true;}
            }
        }
    }
    route_if_ready(f,w,h);
    // The last uploaded mask must not survive a source resize or changed settings.
    if(f->uploaded_w!=w||f->uploaded_h!=h)f->uploaded_valid=false;
    if(!f->uploaded_valid||f->uploaded_revision!=f->revision.load()||
       now-f->uploaded_time>std::chrono::milliseconds(f->age.load())||f->view==3||
       f->uploaded_preview!=(f->view==4)) {
        obs_source_skip_video_filter(f->context);return;
    }
    if(obs_source_process_filter_begin(f->context,GS_RGBA,OBS_NO_DIRECT_RENDERING)) {
        gs_effect_set_texture(gs_effect_get_param_by_name(f->effect,"text_mask"),f->mask);
        gs_effect_set_int(gs_effect_get_param_by_name(f->effect,"view_mode"),f->view.load());
        gs_effect_set_float(gs_effect_get_param_by_name(f->effect,"block_px"),float(f->block.load()));
        struct vec2 size;vec2_set(&size,float(w),float(h));
        gs_effect_set_vec2(gs_effect_get_param_by_name(f->effect,"image_size"),&size);
        obs_source_process_filter_end(f->context,f->effect,w,h);
    }
}
const char* name(void*){return "lll.sight Text Obfuscation (OCR)";}
}
struct obs_source_info sight_text_obfuscation_filter_info={
    .id="sight_text_obfuscation_filter",.type=OBS_SOURCE_TYPE_FILTER,
    .output_flags=OBS_SOURCE_VIDEO,.get_name=name,.create=create,.destroy=destroy,
    .get_defaults=defaults,.get_properties=properties,.update=update,.video_render=render,
};
