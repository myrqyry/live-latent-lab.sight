#include <obs.h>
#include <obs-module.h>
#include <graphics/graphics.h>
#include <callback/proc.h>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <utility>

namespace {
constexpr uint32_t W = 256, H = 144;

std::mutex ui_mutex;
std::vector<std::pair<obs_task_t, void *>> ui_tasks;

void queue_ui(obs_task_t task, void *param, bool wait)
{
    if (wait) {
        std::cerr << "FAIL: synchronous UI task in test\n";
        return;
    }
    std::lock_guard lock(ui_mutex);
    ui_tasks.emplace_back(task, param);
}

void drain_ui()
{
    std::vector<std::pair<obs_task_t, void *>> ready;
    {
        std::lock_guard lock(ui_mutex);
        ready.swap(ui_tasks);
    }
    for (const auto &item : ready)
        item.first(item.second);
}

std::atomic<bool> alternate{false};
std::atomic<int> motion_phase{0};
constexpr int DX=8,DY=-4;
struct Pattern { gs_texture_t *tex = nullptr,*alt=nullptr,*detail=nullptr,*moved=nullptr,*cut=nullptr; };
gs_texture_t* pick(Pattern*p){
    switch(motion_phase.load()){case 1:return p->detail;case 2:return p->moved;case 3:return p->cut;default:return alternate?p->alt:p->tex;}
}
const char *pattern_name(void *) { return "Native diffusion test pattern"; }
uint32_t pattern_width(void *) { return W; }
uint32_t pattern_height(void *) { return H; }

void *pattern_create(obs_data_t *, obs_source_t *)
{
    auto *p = new Pattern;
    std::vector<uint8_t> rgba(W * H * 4);
    for (uint32_t y = 0; y < H; ++y)
        for (uint32_t x = 0; x < W; ++x) {
            const auto i = (y * W + x) * 4;
            const bool block = x > 60 && x < 260 && y > 45 && y < 152;
            rgba[i] = block ? 245 : static_cast<uint8_t>(20 + x / 6);
            rgba[i + 1] = static_cast<uint8_t>(80 + (x % 2) * 70);
            rgba[i + 2] = block ? 60 : static_cast<uint8_t>(30 + y / 7);
            rgba[i + 3] = 255;
        }
    const uint8_t *bytes = rgba.data();
    obs_enter_graphics();
    p->tex = gs_texture_create(W, H, GS_RGBA, 1, &bytes, 0);
    for(size_t i=0;i<rgba.size();i+=4){
        rgba[i]=uint8_t(255-rgba[i]);
        rgba[i+1]=uint8_t(255-rgba[i+1]);
        rgba[i+2]=uint8_t(255-rgba[i+2]);
    }
    p->alt=gs_texture_create(W,H,GS_RGBA,1,&bytes,0);
    // Bounded, nonperiodic texture plus a smooth global gradient makes
    // forward/backward flow and the signed correction observable.
    auto hash=[](int x,int y){
        uint32_t h=uint32_t(x)*374761393u+uint32_t(y)*668265263u+17u;
        h=(h^(h>>13))*1274126177u;return float((h^(h>>16))&255u);
    };
    for(unsigned y=0;y<H;++y)for(unsigned x=0;x<W;++x){
        float tx=float(x%8)/8.f,ty=float(y%8)/8.f;
        int gx=x/8,gy=y/8;
        float v=(1-ty)*((1-tx)*hash(gx,gy)+tx*hash(gx+1,gy))+ty*((1-tx)*hash(gx,gy+1)+tx*hash(gx+1,gy+1));
        auto i=(y*W+x)*4;
        rgba[i]=uint8_t(25+v*.55f+x*.07f);
        rgba[i+1]=uint8_t(30+v*.48f+y*.10f);
        rgba[i+2]=uint8_t(20+v*.43f+x*.08f);
        rgba[i+3]=255;
    }
    bytes=rgba.data();p->detail=gs_texture_create(W,H,GS_RGBA,1,&bytes,0);
    auto shifted=rgba;
    for(int y=0;y<int(H);++y)for(int x=0;x<int(W);++x){
        int sx=std::clamp(x-DX,0,int(W)-1),sy=std::clamp(y-DY,0,int(H)-1);
        std::memcpy(shifted.data()+(y*W+x)*4,rgba.data()+(sy*W+sx)*4,4);
    }
    bytes=shifted.data();p->moved=gs_texture_create(W,H,GS_RGBA,1,&bytes,0);
    std::fill(shifted.begin(),shifted.end(),255);
    bytes=shifted.data();p->cut=gs_texture_create(W,H,GS_RGBA,1,&bytes,0);
    obs_leave_graphics();
    return p;
}
void pattern_destroy(void *data)
{
    auto *p = static_cast<Pattern *>(data);
    obs_enter_graphics();
    if (p->tex) gs_texture_destroy(p->tex);
    if(p->alt)gs_texture_destroy(p->alt);
    if(p->detail)gs_texture_destroy(p->detail);
    if(p->moved)gs_texture_destroy(p->moved);
    if(p->cut)gs_texture_destroy(p->cut);
    obs_leave_graphics();
    delete p;
}
void pattern_render(void *data, gs_effect_t *)
{
    auto *p = static_cast<Pattern *>(data);
    auto *tex=pick(p);
    if(tex)obs_source_draw(tex,0,0,W,H,false);
}

// BrowserSource::Render owns its own effect loop. Exercise this contract
// without CEF, a network page, or the user's scene collection.
std::atomic<unsigned> nested_draws{0};
void custom_pattern_render(void *data, gs_effect_t *) {
    auto *p=static_cast<Pattern*>(data);
    if(gs_get_effect()){
        ++nested_draws;
        return;
    }
    auto *tex=pick(p);
    if(!tex)return;
    auto *effect=obs_get_base_effect(OBS_EFFECT_DEFAULT);
    gs_effect_set_texture(gs_effect_get_param_by_name(effect,"image"),tex);
    while(gs_effect_loop(effect,"Draw"))gs_draw_sprite(tex,0,W,H);
}

obs_source_info pattern_info = {
    .id = "native_sd_live_smoke_pattern",
    .type = OBS_SOURCE_TYPE_INPUT,
    .output_flags = OBS_SOURCE_VIDEO,
    .get_name = pattern_name,
    .create = pattern_create,
    .destroy = pattern_destroy,
    .get_width = pattern_width,
    .get_height = pattern_height,
    .video_render = pattern_render,
};

bool render(obs_source_t *source, std::vector<uint8_t> &pixels)
{
    obs_enter_graphics();
    auto *surface = gs_stagesurface_create(W, H, GS_RGBA);
    auto *target = gs_texrender_create(GS_RGBA, GS_ZS_NONE);
    bool success = false;
    if (surface && target && gs_texrender_begin(target, W, H)) {
        gs_projection_push();
        gs_matrix_push();
        gs_matrix_identity();
        gs_ortho(0.f, float(W), 0.f, float(H), -100.f, 100.f);
        obs_source_video_render(source);
        gs_matrix_pop();
        gs_projection_pop();
        gs_texrender_end(target);
        auto *tex = gs_texrender_get_texture(target);
        if (tex) {
            gs_stage_texture(surface, tex);
            uint8_t *bytes = nullptr;
            uint32_t stride = 0;
            if (gs_stagesurface_map(surface, &bytes, &stride)) {
                if (stride >= W * 4) {
                    pixels.resize(size_t(W) * H * 4);
                    for (uint32_t y = 0; y < H; ++y)
                        std::memcpy(pixels.data() + size_t(y) * W * 4,
                                    bytes + size_t(y) * stride, W * 4);
                    success = true;
                }
                gs_stagesurface_unmap(surface);
            }
        }
    }
    if (surface) gs_stagesurface_destroy(surface);
    if (target) gs_texrender_destroy(target);
    obs_leave_graphics();
    drain_ui();
    return success;
}


bool motion_stats(obs_source_t* filter,float&coverage){
    calldata_t args;calldata_init(&args);
    const bool ok=proc_handler_call(obs_source_get_proc_handler(filter),"sight_native_sd_live_motion_stats_v1",&args);
    const bool drawn=ok&&calldata_bool(&args,"motion_drawn");
    coverage=ok?float(calldata_float(&args,"flow_coverage")):0.f;
    calldata_free(&args);return drawn;
}
float history_weight(obs_source_t*filter){
    calldata_t args;calldata_init(&args);
    const bool ok=proc_handler_call(obs_source_get_proc_handler(filter),"sight_native_sd_live_temporal_stats_v1",&args);
    float value=ok?float(calldata_float(&args,"history_weight")):0.f;
    calldata_free(&args);return value;
}
uint64_t displayed(obs_source_t* filter){
    calldata_t args;calldata_init(&args);
    const bool ok=proc_handler_call(obs_source_get_proc_handler(filter),"sight_native_sd_live_stats_v1",&args);
    auto value=ok?uint64_t(calldata_int(&args,"sequence")):0;
    calldata_free(&args);
    return value;
}
}
int main(int argc,char**argv){
    if(argc!=3||!obs_startup("en-US",nullptr,nullptr))return 2;
    obs_set_ui_task_handler(queue_ui);
    obs_video_info video{};
    video.graphics_module="libobs-opengl";video.fps_num=30;video.fps_den=1;
    video.base_width=W;video.base_height=H;video.output_width=W;video.output_height=H;
    video.output_format=VIDEO_FORMAT_RGBA;video.scale_type=OBS_SCALE_DISABLE;
    if(obs_reset_video(&video)!=OBS_VIDEO_SUCCESS){obs_shutdown();return 3;}
    obs_module_t*module=nullptr;
    if(obs_open_module(&module,argv[1],argv[2])!=MODULE_SUCCESS||!obs_init_module(module)){
        obs_shutdown();return 4;
    }
    auto*props=obs_get_source_properties("sight_native_sd_live_filter");
    if(!props){std::cerr<<"FAIL: native live diffusion filter not registered\n";obs_shutdown();return 5;}
    for(auto* key:{"model_path","taesd_path","backend","prompt","steps","denoise","continuous_denoise","sync_mode","temporal_stability","long_edge","seed","enabled","max_generation_fps","weight_precision"}){
        if(!obs_properties_get(props,key)){std::cerr<<"FAIL: missing "<<key<<"\n";obs_properties_destroy(props);obs_shutdown();return 6;}
    }
    auto*precision=obs_properties_get(props,"weight_precision");
    bool has_default=false,has_q8=false;
    for(size_t i=0;i<obs_property_list_item_count(precision);++i){
        auto*value=obs_property_list_item_string(precision,i);
        if(value){has_default|=std::strcmp(value,"default")==0;has_q8|=std::strcmp(value,"q8_0")==0;}
    }
    obs_properties_destroy(props);
    if(!has_default||!has_q8){std::cerr<<"FAIL: expected weight precision choices\n";obs_shutdown();return 6;}
    pattern_info.output_flags|=OBS_SOURCE_CUSTOM_DRAW;
    pattern_info.video_render=custom_pattern_render;
    obs_register_source(&pattern_info);
    auto*source=obs_source_create_private(pattern_info.id,"diffusion fixture",nullptr);
    std::vector<uint8_t> a,b,out,first;
    if(!source||!render(source,a)){if(source)obs_source_release(source);obs_shutdown();return 7;}
    alternate=true;
    if(!render(source,b)){obs_source_release(source);obs_shutdown();return 7;}
    alternate=false;
    auto*s=obs_data_create();
    obs_data_set_bool(s,"enabled",true);obs_data_set_string(s,"model_path","/nonexistent/sight-model.safetensors");
    obs_data_set_string(s,"backend","cpu");obs_data_set_int(s,"long_edge",256);
    auto*f=obs_source_create_private("sight_native_sd_live_filter","native diffusion",s);
    if(!f){obs_data_release(s);obs_source_release(source);obs_shutdown();return 8;}
    obs_source_filter_add(source,f);
    bool pass=true;
    for(int i=0;i<10;++i){
        pass=pass&&render(source,out)&&out==a&&displayed(f)==0;
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
    }
    bool generated=true,changed=true,disabled=true,motion_pass=true;
    if(!std::getenv("SIGHT_SD_LIVE_TEST_MODEL"))std::cout<<"model_backed=not_requested\n";
    if(const char*model=std::getenv("SIGHT_SD_LIVE_TEST_MODEL")){
        obs_data_set_string(s,"model_path",model);
        if(const char*tae=std::getenv("SIGHT_SD_LIVE_TEST_TAE"))obs_data_set_string(s,"taesd_path",tae);
        if(const char*device=std::getenv("SIGHT_SD_LIVE_TEST_DEVICE"))obs_data_set_string(s,"backend",device);
        obs_data_set_string(s,"prompt","cinematic illustration, abstract blocks, warm colors");
        obs_data_set_int(s,"steps",2);obs_data_set_double(s,"denoise",.35);
        obs_data_set_bool(s,"continuous_denoise",true);
        obs_source_update(f,s);
        auto wait_output=[&](uint64_t after,std::vector<uint8_t>&pixels){
            auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(100);
            do{
                if(!render(source,pixels))return false;
                if(displayed(f)>after)return true;
                std::this_thread::sleep_for(std::chrono::milliseconds(30));
            }while(std::chrono::steady_clock::now()<deadline);
            return false;
        };
        generated=wait_output(0,first)&&first!=a;
        const uint64_t seq=displayed(f);
        alternate=true;
        // A request might already be in flight for A; require actual B-dependent pixels.
        changed=false;
        for(int i=0;i<3&&generated;++i){
            auto previous=displayed(f);
            if(!wait_output(previous,out))break;
            if(out!=first&&out!=b){changed=true;break;}
        }
        obs_data_set_bool(s,"enabled",false);obs_source_update(f,s);
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
        disabled=render(source,out)&&out==b&&displayed(f)==0;
        std::cout<<"model_backed="<<generated<<" new_source_output="<<changed<<" disabled_passthrough="<<disabled<<" first_sequence="<<seq<<"\n";
        if(std::getenv("SIGHT_SD_LIVE_TEST_WEIGHT_SWITCH")){
            alternate=false;
            obs_data_set_bool(s,"enabled",true);
            obs_data_set_string(s,"weight_precision","q8_0");
            obs_source_update(f,s);
            std::vector<uint8_t> quantized,restored;
            const bool q8=wait_output(0,quantized)&&quantized!=a&&quantized!=first;
            const auto qseq=displayed(f);
            obs_data_set_string(s,"weight_precision","default");
            obs_source_update(f,s);
            const bool restore=wait_output(qseq,restored)&&restored==first;
            obs_data_set_bool(s,"enabled",false);obs_source_update(f,s);
            pass &= q8&&restore;
            std::cout<<"weight_switch_q8="<<q8<<" default_restored="<<restore<<"\n";
        }
#ifdef SIGHT_SD_TEST_DENSE_FLOW
        if(std::getenv("SIGHT_SD_LIVE_TEST_MOTION")){
            motion_phase=1;
            std::vector<uint8_t> raw,corrected,shifted_raw;
            bool captured_raw=render(source,raw);
            motion_phase=2;captured_raw=captured_raw&&render(source,shifted_raw);
            motion_phase=1;
            obs_data_set_bool(s,"enabled",true);
            obs_data_set_int(s,"sync_mode",1);
            // Exercise asynchronous motion reprojection on the same keyframe
            // without racing unrestricted diffusion keyframe production.
            obs_data_set_int(s,"max_generation_fps",3);
            obs_data_set_int(s,"steps",1);
            obs_data_set_double(s,"denoise",.35);
            obs_source_update(f,s);
            float coverage=0.f;
            bool started=false;
            auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(100);
            while(std::chrono::steady_clock::now()<deadline){
                if(!render(source,corrected))break;
                if(motion_stats(f,coverage)&&corrected!=raw&&displayed(f)>0){started=true;break;}
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            const auto old_seq=displayed(f);
            motion_phase=2;
            bool warped=false;double warp_error=0,unwarped_error=0,change=0;
            // Only accept pixels from the same generated keyframe, before
            // another model result can replace it.
            for(int i=0;i<35&&started;++i){
                if(!render(source,out)||displayed(f)!=old_seq)break;
                if(motion_stats(f,coverage)&&coverage>.2f){
                    warp_error=unwarped_error=change=0;size_t n=0;
                    for(int y=16;y<int(H)-16;++y)for(int x=16;x<int(W)-16;++x){
                        auto j=(y*W+x)*4;auto k=((y-DY)*W+x-DX)*4;
                        for(unsigned c=0;c<3;++c){
                            int actual=int(out[j+c])-shifted_raw[j+c];
                            int expected=int(corrected[k+c])-raw[k+c];
                            int unwarped=int(corrected[j+c])-raw[j+c];
                            warp_error+=std::abs(actual-expected);unwarped_error+=std::abs(actual-unwarped);
                            change+=std::abs(actual);++n;
                        }
                    }
                    warp_error/=n;unwarped_error/=n;change/=n;
                    if(change>.5&&warp_error<unwarped_error*.95){warped=true;break;}
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(8));
            }
            motion_phase=3;
            bool cut=render(source,out);
            if(cut){
                // White cut is far outside the old reference's RGB range.
                cut=std::all_of(out.begin(),out.end(),[](uint8_t v){return v>=254;});
            }
            obs_data_set_bool(s,"enabled",false);obs_source_update(f,s);
            std::this_thread::sleep_for(std::chrono::milliseconds(120));
            const bool bypass=render(source,out)&&displayed(f)==0&&
                std::all_of(out.begin(),out.end(),[](uint8_t v){return v>=254;});
            motion_pass=captured_raw&&started&&warped&&cut&&bypass;
            // Exercise actual historical blending on a second real
            // keyframe and reset the setting before testing passthrough.
            if(std::getenv("SIGHT_SD_LIVE_TEST_TEMPORAL")){
                motion_phase=1;
                obs_data_set_double(s,"temporal_stability",.5);
                obs_data_set_bool(s,"enabled",true);
                obs_source_update(f,s);
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                bool first_key=wait_output(0,out);
                const auto prior=displayed(f);
                const bool first_clean=first_key&&history_weight(f)==0.f;
                motion_phase=2;
                float weight=0.f;
                bool blended=false;
                for(int j=0;j<5&&first_key;++j){
                    if(!wait_output(j==0?prior:displayed(f),out))break;
                    weight=history_weight(f);
                    if(weight>.15f){blended=true;break;}
                }
                obs_data_set_double(s,"temporal_stability",0.);
                obs_source_update(f,s);
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                bool zero=wait_output(0,out)&&history_weight(f)==0.f;
                obs_data_set_bool(s,"enabled",false);obs_source_update(f,s);
                motion_pass=motion_pass&&first_clean&&blended&&zero;
                std::cout<<"temporal_first_clean="<<first_clean<<" temporal_blended="<<blended
                    <<" history_weight="<<weight<<" temporal_neutral="<<zero<<"\n";
            }
            std::cout<<"motion_started="<<started<<" same_keyframe_warp="<<warped<<" coverage="<<coverage
                <<" warp_error="<<warp_error<<" unwarped_error="<<unwarped_error<<" correction="<<change
                <<" cut_passthrough="<<cut<<" motion_disabled="<<bypass<<"\n";
        }
#endif
    }
    obs_source_filter_remove(source,f);obs_source_release(f);obs_data_release(s);obs_source_release(source);
    obs_shutdown();
    std::cout<<"missing_model_passthrough="<<pass<<" nested_draws="<<nested_draws.load()<<"\n";
    return pass&&generated&&changed&&disabled&&motion_pass&&!nested_draws?0:1;
}
