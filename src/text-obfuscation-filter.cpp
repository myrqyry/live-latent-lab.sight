#include <obs-module.h>
#include "core/analysis_capture.hpp"
#include <graphics/graphics.h>
#include <util/platform.h>
#include <algorithm>
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
#include "core/sight_text_v1.h"

namespace {
using Clock=std::chrono::steady_clock;
std::atomic<uint64_t> text_sequence{0};
int64_t millis(Clock::time_point t) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(t.time_since_epoch()).count();
}
struct Config {
    std::string directory, language;
    int page_mode=11,edge=1280,confidence=0,padding=8,interval=250;
    bool operator==(const Config&) const=default;
};
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
    bool valid=false;
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
    bool staged=false,uploaded_valid=false,announced=false;
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
    c.page_mode=int(obs_data_get_int(s,"ocr_layout"));
    if(c.page_mode!=3&&c.page_mode!=6&&c.page_mode!=7&&c.page_mode!=11)c.page_mode=11;
    c.edge=std::clamp(int(obs_data_get_int(s,"ocr_edge")),640,2048);
    c.confidence=std::clamp(int(obs_data_get_int(s,"ocr_confidence")),0,100);
    c.padding=std::clamp(int(obs_data_get_int(s,"ocr_padding")),0,40);
    c.interval=std::clamp(int(obs_data_get_int(s,"ocr_interval")),100,2000);
    return c;
}
void worker(Filter* f) {
    sight::TextOcr ocr;
    std::string loaded_path,loaded_language;
    bool attempted=false,ready=false;
    for(;;) {
        Frame frame;
        Config cfg;
        {
            std::unique_lock lock(f->mutex);
            f->cv.wait(lock,[&]{return f->stopping||f->pending;});
            if(f->stopping)return;
            frame=std::move(f->frame);f->pending=false;cfg=f->config;
        }
        if(!attempted||!ready||loaded_path!=cfg.directory||loaded_language!=cfg.language) {
            loaded_path=cfg.directory;loaded_language=cfg.language;attempted=true;
            ready=ocr.load(loaded_path,loaded_language);
            blog(ready?LOG_INFO:LOG_WARNING,"[Sight/OCR] SIGHT_TEXT_OCR_20261117: local model %s",ready?"ready":"unavailable");
        }
        Mask out;
        out.w=frame.w;out.h=frame.h;out.sw=frame.sw;out.sh=frame.sh;
        out.revision=frame.revision;out.captured=frame.captured;out.sequence=++text_sequence;
        sight_text_v1 snapshot{};
        snapshot.abi_version=SIGHT_TEXT_ABI_V1;snapshot.struct_size=sizeof(snapshot);
        snapshot.source_width=frame.sw;snapshot.source_height=frame.sh;
        snapshot.sequence=out.sequence;
        if(ready&&frame.rgba.size()==size_t(frame.w)*frame.h*4) {
            std::vector<uint8_t> gray(size_t(frame.w)*frame.h);
            for(size_t i=0;i<gray.size();++i) {
                const auto* p=frame.rgba.data()+i*4;
                gray[i]=uint8_t((77u*p[0]+150u*p[1]+29u*p[2]+128u)>>8);
            }
            const auto text=ocr.recognize(gray,frame.w,frame.h,cfg.page_mode);
            out.valid=text.valid;
            if(out.valid) {
                out.pixels.assign(gray.size(),0);
                for(const auto& line:text.lines) {
                    if(line.confidence<cfg.confidence)continue;
                    const int x0=std::max(0,line.x0-cfg.padding),y0=std::max(0,line.y0-cfg.padding);
                    const int x1=std::min(int(frame.w),line.x1+cfg.padding),y1=std::min(int(frame.h),line.y1+cfg.padding);
                    auto& box=snapshot.lines[snapshot.count++];
                    box={float(x0)/frame.w,float(y0)/frame.h,float(x1)/frame.w,float(y1)/frame.h,
                         std::clamp(line.confidence/100.f,0.f,1.f)};
                    for(int y=y0;y<y1;++y)
                        std::fill(out.pixels.begin()+size_t(y)*frame.w+x0,
                                  out.pixels.begin()+size_t(y)*frame.w+x1,uint8_t(255));
                }
            }
        }
        {
            std::lock_guard lock(f->mutex);
            if(f->revision==frame.revision) {
                f->published=out.valid?snapshot:sight_text_v1{};
                f->published_time=frame.captured;
                f->published_revision=frame.revision;
                f->result=std::move(out);
                // Avoid repeatedly copying frames if the model is missing.
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
}
void update(void* data,obs_data_t* s) {
    auto* f=static_cast<Filter*>(data);
    auto cfg=read_config(s);
    f->view=std::clamp(int(obs_data_get_int(s,"ocr_view")),0,3);
    f->block=std::clamp(int(obs_data_get_int(s,"ocr_block")),4,80);
    f->age=std::clamp(int(obs_data_get_int(s,"ocr_age")),500,5000);
    f->edge=cfg.edge;f->interval=cfg.interval;
    std::lock_guard lock(f->mutex);
    if(!(cfg==f->config)) {
        f->config=std::move(cfg);
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
    if(auto* procs=obs_source_get_proc_handler(context))
        proc_handler_add(procs,"void sight_text_read_v1(in int max_age_ms, in ptr snapshot, out bool available)",
                         read_text,f);
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
obs_properties_t* properties(void*) {
    auto* p=obs_properties_create();
    auto* path=obs_properties_add_path(p,"tessdata_path","Tesseract model folder",OBS_PATH_DIRECTORY,nullptr,default_directory().c_str());
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
    obs_property_set_long_description(view,"Best-effort text obscuring. Missed, changed or stale text may remain visible; missing OCR passes through live video. Not a privacy guarantee.");
    obs_properties_add_int_slider(p,"ocr_edge","OCR input longest edge",640,2048,128);
    obs_properties_add_int_slider(p,"ocr_padding","Line padding (OCR pixels)",0,40,1);
    obs_properties_add_int_slider(p,"ocr_confidence","Minimum line confidence",0,100,1);
    obs_properties_add_int_slider(p,"ocr_interval","Minimum capture interval (ms)",100,2000,50);
    obs_properties_add_int_slider(p,"ocr_age","Maximum mask age (ms)",500,5000,100);
    obs_properties_add_int_slider(p,"ocr_block","Mosaic cell size (pixels)",4,80,2);
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
                f->uploaded_w=w;f->uploaded_h=h;
                // Store source geometry independently of the next queued capture.
                if(!f->announced){blog(LOG_INFO,"[Sight/OCR] SIGHT_TEXT_OBFUSCATION_GPU_20261117");f->announced=true;}
            }
        }
    }
    // The last uploaded mask must not survive a source resize or changed settings.
    if(f->uploaded_w!=w||f->uploaded_h!=h)f->uploaded_valid=false;
    if(!f->uploaded_valid||f->uploaded_revision!=f->revision.load()||
       now-f->uploaded_time>std::chrono::milliseconds(f->age.load())||f->view==3) {
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
const char* name(void*){return "Sight: Text Obfuscation (OCR)";}
}
struct obs_source_info sight_text_obfuscation_filter_info={
    .id="sight_text_obfuscation_filter",.type=OBS_SOURCE_TYPE_FILTER,
    .output_flags=OBS_SOURCE_VIDEO,.get_name=name,.create=create,.destroy=destroy,
    .get_defaults=defaults,.get_properties=properties,.update=update,.video_render=render,
};
