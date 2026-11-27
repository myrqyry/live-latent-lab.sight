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
constexpr uint32_t W = 640, H = 360;

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
struct Pattern { gs_texture_t *tex = nullptr,*alt=nullptr; };
const char *pattern_name(void *) { return "Neural rendering test pattern"; }
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
    obs_leave_graphics();
    return p;
}
void pattern_destroy(void *data)
{
    auto *p = static_cast<Pattern *>(data);
    obs_enter_graphics();
    if (p->tex) gs_texture_destroy(p->tex);
    if(p->alt)gs_texture_destroy(p->alt);
    obs_leave_graphics();
    delete p;
}
void pattern_render(void *data, gs_effect_t *)
{
    auto *p = static_cast<Pattern *>(data);
    auto *tex=alternate?p->alt:p->tex;
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
    auto *tex=alternate?p->alt:p->tex;
    if(!tex)return;
    auto *effect=obs_get_base_effect(OBS_EFFECT_DEFAULT);
    gs_effect_set_texture(gs_effect_get_param_by_name(effect,"image"),tex);
    while(gs_effect_loop(effect,"Draw"))gs_draw_sprite(tex,0,W,H);
}

obs_source_info pattern_info = {
    .id = "neural_render_smoke_pattern",
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

double mean_rgb_error(const std::vector<uint8_t>&a,const std::vector<uint8_t>&b){
    if(a.empty()||a.size()!=b.size())return 255.;
    uint64_t sum=0;
    for(size_t i=0;i<a.size();i+=4)for(unsigned c=0;c<3;++c)
        sum+=std::abs(int(a[i+c])-int(b[i+c]));
    return double(sum)/(a.size()/4*3);
}
size_t changed(const std::vector<uint8_t> &a, const std::vector<uint8_t> &b)
{
    if (a.size() != b.size()) return 0;
    size_t diff = 0;
    for (size_t i = 0; i < a.size(); i += 4)
        if (a[i] != b[i] || a[i + 1] != b[i + 1] || a[i + 2] != b[i + 2])
            ++diff;
    return diff;
}
}
bool drawn_at(obs_source_t *filter,uint32_t w,uint32_t h){
    calldata_t args;
    calldata_init(&args);
    bool ok=proc_handler_call(obs_source_get_proc_handler(filter),"sight_neural_rendering_stats_v1",&args)
        &&calldata_int(&args,"width")==w&&calldata_int(&args,"height")==h;
    calldata_free(&args);
    return ok;
}
int main(int argc,char **argv)
{
    if(argc!=3) return 2;
    if(!obs_startup("en-US",nullptr,nullptr)) return 1;
    obs_set_ui_task_handler(queue_ui);
    obs_video_info v{};
    v.graphics_module="libobs-opengl";v.fps_num=30;v.fps_den=1;
    v.base_width=v.output_width=W;v.base_height=v.output_height=H;
    v.output_format=VIDEO_FORMAT_RGBA;v.colorspace=VIDEO_CS_DEFAULT;
    v.range=VIDEO_RANGE_DEFAULT;v.scale_type=OBS_SCALE_DISABLE;
    if(obs_reset_video(&v)!=OBS_VIDEO_SUCCESS) {obs_shutdown();return 1;}
    obs_module_t *module=nullptr;
    if(obs_open_module(&module,argv[1],argv[2])!=MODULE_SUCCESS ||
       !obs_init_module(module)) {obs_shutdown();return 1;}
    const char *id="sight_neural_rendering_filter";
    if(!obs_source_get_display_name(id)) {
        std::cerr<<"FAIL: experimental neural rendering filter not registered\n";
        obs_shutdown();return 1;
    }
    auto *props=obs_get_source_properties(id);
    auto *resolution=props?obs_properties_get(props,"resolution"):nullptr;
    bool native_listed=false;
    if(resolution) for(size_t i=0;i<obs_property_list_item_count(resolution);++i)
        native_listed |= obs_property_list_item_int(resolution,i)==0;
    auto *sync=props?obs_properties_get(props,"sync_mode"):nullptr;
    bool locked_listed=false,motion_listed=false;
    if(sync)for(size_t i=0;i<obs_property_list_item_count(sync);++i){
        const auto value=obs_property_list_item_int(sync,i);
        locked_listed|=value==2;
        motion_listed|=value==1;
    }
    bool controls=true;
    for(const char *key:{"strength","relighting","surface_detail","color_strength"}){
        auto *slider=props?obs_properties_get(props,key):nullptr;
        controls=controls&&slider&&obs_property_float_max(slider)>=5.;
    }
    controls=controls&&props&&obs_properties_get(props,"highlight_protection");
    auto *input_group=props?obs_properties_get(props,"neural_input"):nullptr;
    auto *input_props=input_group?obs_property_group_content(input_group):nullptr;
    bool conditioning=input_props!=nullptr;
    for(const char* key:{"structure_preservation","input_clarity","shadow_recovery"}){
        auto *slider=input_props?obs_properties_get(input_props,key):nullptr;
        conditioning=conditioning&&slider&&obs_property_float_max(slider)==1.;
    }
    auto*residual_prop=props?obs_properties_get(props,"residual_format"):nullptr;
    bool formats[3]{};
    if(residual_prop)for(size_t i=0;i<obs_property_list_item_count(residual_prop);++i){
        auto value=obs_property_list_item_int(residual_prop,i);
        if(value>=0&&value<=2)formats[value]=true;
    }
    auto*temporal_prop=props?obs_properties_get(props,"temporal_stability"):nullptr;
    const bool temporal_listed=temporal_prop&&obs_property_float_max(temporal_prop)==.9;
    auto*edge_prop=props?obs_properties_get(props,"edge_aware"):nullptr;
    const bool edge_listed=edge_prop&&obs_property_float_max(edge_prop)==1.;
    auto*depth_prop=props?obs_properties_get(props,"depth_guidance"):nullptr;
    const bool depth_listed=depth_prop&&obs_property_float_max(depth_prop)==1.;
    if(props)obs_properties_destroy(props);
    if(!depth_listed){std::cerr<<"FAIL: depth guidance unavailable\n";obs_shutdown();return 1;}
    if(!edge_listed){std::cerr<<"FAIL: edge-aware upscaling unavailable\n";obs_shutdown();return 1;}
    if(!temporal_listed){std::cerr<<"FAIL: temporal stability unavailable\n";obs_shutdown();return 1;}
    if(!formats[0]||!formats[1]||!formats[2]){
        std::cerr<<"FAIL: neural correction formats unavailable\n";obs_shutdown();return 1;
    }
    if(!conditioning){std::cerr<<"FAIL: neural input conditioning controls unavailable\n";obs_shutdown();return 1;}
    if(!controls){std::cerr<<"FAIL: neural amplification controls unavailable\n";obs_shutdown();return 1;}
    if(!locked_listed||!motion_listed){std::cerr<<"FAIL: synchronization modes unavailable\n";obs_shutdown();return 1;}
    if(!native_listed){std::cerr<<"FAIL: native inference resolution unavailable\n";obs_shutdown();return 1;}
    if(std::getenv("SIGHT_NR_REGISTRATION_ONLY")){
        std::cout<<"PASS neural correction formats registered\n";obs_shutdown();return 0;
    }
    const bool custom=std::getenv("SIGHT_NR_CUSTOM_DRAW")!=nullptr;
    if(custom){
        pattern_info.output_flags|=OBS_SOURCE_CUSTOM_DRAW;
        pattern_info.video_render=custom_pattern_render;
    }
    obs_register_source(&pattern_info);
    auto *source=obs_source_create("neural_render_smoke_pattern","Input",nullptr,nullptr);
    std::vector<uint8_t> baseline,out;
    if(!source || !render(source,baseline)) {obs_shutdown();return 1;}
    alternate=true;
    std::vector<uint8_t> alternate_baseline;
    if(!render(source,alternate_baseline)){obs_source_release(source);obs_shutdown();return 1;}
    alternate=false;
    obs_source_t*depth_producer=nullptr;
    if(const char*model=std::getenv("SIGHT_NR_TEST_DEPTH_MODEL")){
        auto*depth_settings=obs_data_create();
        obs_data_set_string(depth_settings,"model_path",model);
        obs_data_set_int(depth_settings,"view_mode",2);
        obs_data_set_int(depth_settings,"inference_edge",252);
        obs_data_set_int(depth_settings,"inference_cadence",1);
        obs_data_set_int(depth_settings,"temporal_blend",0);
        depth_producer=obs_source_create("sight_depth_filter","Depth upstream",depth_settings,nullptr);
        obs_data_release(depth_settings);
        if(!depth_producer){obs_source_release(source);obs_shutdown();return 1;}
        obs_source_filter_add(source,depth_producer);
    }
    auto *settings=obs_data_create();
    if(depth_producer)obs_data_set_double(settings,"depth_guidance",1.);
    obs_data_set_int(settings,"resolution",320);
    obs_data_set_int(settings,"sync_mode",0);
    obs_data_set_string(settings,"precision","w8a8");
    obs_data_set_double(settings,"strength",1.0);
    const bool combined=depth_producer&&std::getenv("SIGHT_NR_TEST_COMBINED");
    if(combined){
        obs_data_set_int(settings,"residual_format",1);
        obs_data_set_int(settings,"sync_mode",1);
        obs_data_set_double(settings,"temporal_stability",.5);
        obs_data_set_double(settings,"edge_aware",.75);
        obs_data_set_double(settings,"depth_guidance",.75);
    }
    auto *filter=obs_source_create(id,"Neural render",settings,nullptr);
    if(!filter){obs_data_release(settings);obs_source_release(source);obs_shutdown();return 1;}
    obs_source_filter_add(source,filter);
    bool transformed=false,passed=false;
    for(int i=0;i<400;++i) {
        if(!render(source,out))break;
        if(drawn_at(filter,320,180)&&changed(out,baseline)>W*H/10) {transformed=true;break;}
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
    }
    std::cout<<"neural_render_transformed="<<transformed<<" changed="<<changed(out,baseline)<<"\n";
    if(depth_producer){
        auto active=[&](){
            calldata_t args;calldata_init(&args);
            bool result=proc_handler_call(obs_source_get_proc_handler(filter),
                "sight_neural_rendering_depth_stats_v1",&args)&&calldata_bool(&args,"active");
            calldata_free(&args);return result;
        };
        auto combined_ready=[&](){
            if(!combined)return true;
            calldata_t residual,temporal;
            calldata_init(&residual);calldata_init(&temporal);
            auto*proc=obs_source_get_proc_handler(filter);
            bool result=proc_handler_call(proc,"sight_neural_rendering_residual_stats_v1",&residual)&&
                calldata_int(&residual,"format")==1&&
                proc_handler_call(proc,"sight_neural_rendering_temporal_stats_v1",&temporal)&&
                calldata_float(&temporal,"history_weight")>.15;
            calldata_free(&temporal);calldata_free(&residual);return result;
        };
        bool adjacent=obs_filter_get_target(filter)==depth_producer;
        bool guided=false;
        for(int i=0;i<500&&adjacent;++i){
            if(!render(source,out))break;
            if(active()&&combined_ready()&&drawn_at(filter,320,180)&&changed(out,baseline)>W*H/10){guided=true;break;}
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        obs_source_set_enabled(depth_producer,false);
        bool fallback=false;
        for(int i=0;i<100;++i){
            if(!render(source,out))break;
            if(!active()&&drawn_at(filter,320,180)&&changed(out,baseline)>W*H/10){fallback=true;break;}
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        std::cout<<"depth_adjacent="<<adjacent<<" depth_guided="<<guided<<" depth_disabled_fallback="<<fallback<<"\n";
        if(combined)std::cout<<"combined_float_temporal_depth="<<guided<<"\n";
        obs_source_filter_remove(source,filter);obs_source_release(filter);
        obs_source_filter_remove(source,depth_producer);obs_source_release(depth_producer);
        obs_data_release(settings);obs_source_release(source);drain_ui();obs_shutdown();
        return transformed&&adjacent&&guided&&fallback&&!nested_draws?0:1;
    }
    size_t detail_pairs=0,retained=0;
    if(transformed) for(uint32_t y=20;y<H-20;++y)
        for(uint32_t x=20;x<W-20;x+=2){
            const size_t i=(size_t(y)*W+x)*4+1;
            const int a=int(baseline[i+4])-int(baseline[i]);
            const int b=int(out[i+4])-int(out[i]);
            if(a>=60){++detail_pairs;if(b>=50)++retained;}
        }
    const bool detail=detail_pairs>1000&&retained*10>detail_pairs*9;
    std::cout<<"full_resolution_detail="<<detail<<" retained="<<retained<<"/"<<detail_pairs<<"\n";
    if(std::getenv("SIGHT_NR_TEST_FLOAT_RESIDUAL")){
        bool precision_pass=transformed;
        for(int mode:{1,2,0}){
            obs_data_set_int(settings,"residual_format",mode);
            obs_source_update(filter,settings);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            bool actual=false;
            for(int i=0;i<600;++i){
                if(!render(source,out))break;
                calldata_t args;calldata_init(&args);
                bool matches=proc_handler_call(obs_source_get_proc_handler(filter),
                    "sight_neural_rendering_residual_stats_v1",&args)&&calldata_int(&args,"format")==mode;
                calldata_free(&args);
                if(matches&&drawn_at(filter,320,180)&&changed(out,baseline)>W*H/10){actual=true;break;}
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
            std::cout<<"residual_mode="<<mode<<" rendered="<<actual<<"\n";
            precision_pass &= actual;
        }
        obs_source_filter_remove(source,filter);obs_source_release(filter);
        obs_data_release(settings);obs_source_release(source);drain_ui();obs_shutdown();
        return precision_pass&&!nested_draws?0:1;
    }
    if(std::getenv("SIGHT_NR_TEST_TEMPORAL")){
        obs_data_set_int(settings,"residual_format",1);
        obs_data_set_int(settings,"sync_mode",1);
        obs_data_set_double(settings,"temporal_stability",.5);
        obs_source_update(filter,settings);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        auto weight=[&](){
            calldata_t args;calldata_init(&args);
            bool ok=proc_handler_call(obs_source_get_proc_handler(filter),
                "sight_neural_rendering_temporal_stats_v1",&args);
            float v=ok?float(calldata_float(&args,"history_weight")):-1.f;
            calldata_free(&args);return v;
        };
        bool blended=false;
        for(int i=0;i<600;++i){
            if(!render(source,out))break;
            if(drawn_at(filter,320,180)&&weight()>.15f&&changed(out,baseline)>W*H/10){blended=true;break;}
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        const float applied=weight();
        obs_data_set_double(settings,"temporal_stability",0.);
        obs_source_update(filter,settings);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        bool neutral=false;
        for(int i=0;i<600;++i){
            if(!render(source,out))break;
            if(drawn_at(filter,320,180)&&weight()==0.f&&changed(out,baseline)>W*H/10){neutral=true;break;}
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        std::cout<<"neural_temporal_blended="<<blended<<" weight="<<applied<<" neutral="<<neutral<<"\n";
        obs_source_filter_remove(source,filter);obs_source_release(filter);
        obs_data_release(settings);obs_source_release(source);drain_ui();obs_shutdown();
        return transformed&&blended&&neutral&&!nested_draws?0:1;
    }
    const auto normal=out;
    const double normal_error=mean_rgb_error(normal,baseline);
    auto adjustment=[&](const char *key,double value,std::vector<uint8_t>&pixels){
        obs_data_set_double(settings,key,value);
        obs_source_update(filter,settings);
        // libobs defers video-source updates to a video tick. Rendering
        // directly in the test before that tick sees the old settings.
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        for(int i=0;i<80;++i){
            if(render(source,pixels)&&drawn_at(filter,320,180))return true;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        return false;
    };
    std::vector<uint8_t> boosted,protected_out,detail_only,light_only,color_only,restored;
    bool amplification=transformed&&adjustment("strength",5.,boosted);
    const double boost_error=mean_rgb_error(boosted,baseline);
    amplification=amplification&&boost_error>normal_error*2.;
    // 5x alone does not reach the gamut knee on this fixture. Combine
    // gains to deliberately exercise clipping and the protection branch.
    obs_data_set_double(settings,"relighting",5.);
    obs_data_set_double(settings,"surface_detail",5.);
    std::vector<uint8_t> extreme;
    bool protection=adjustment("color_strength",5.,extreme)
        &&adjustment("highlight_protection",1.,protected_out);
    auto clipped=[](const std::vector<uint8_t>&pixels){
        size_t count=0;
        for(size_t i=0;i<pixels.size();i+=4)for(int c=0;c<3;++c)
            count+=pixels[i+c]==0||pixels[i+c]==255;
        return count;
    };
    const auto unprotected_clips=clipped(extreme),protected_clips=clipped(protected_out);
    protection=protection&&unprotected_clips>0&&protected_clips<unprotected_clips;
    std::cout<<"clipped_channels="<<unprotected_clips<<" protected_channels="<<protected_clips<<"\n";
    obs_data_set_double(settings,"relighting",1.);
    obs_data_set_double(settings,"surface_detail",1.);
    obs_data_set_double(settings,"color_strength",1.);
    obs_data_set_double(settings,"highlight_protection",0.);
    adjustment("strength",1.,restored);
    bool neutral=mean_rgb_error(restored,normal)<.01;
    bool components=adjustment("surface_detail",5.,detail_only)
        &&mean_rgb_error(detail_only,normal)>.05;
    adjustment("surface_detail",1.,restored);
    components=components&&adjustment("relighting",5.,light_only)
        &&mean_rgb_error(light_only,normal)>.05;
    adjustment("relighting",1.,restored);
    components=components&&adjustment("color_strength",5.,color_only)
        &&mean_rgb_error(color_only,normal)>.05;
    adjustment("color_strength",1.,restored);
    neutral=neutral&&mean_rgb_error(restored,normal)<.01;
    std::cout<<"amplification="<<amplification<<" normal_rgb_error="<<normal_error
             <<" boost_rgb_error="<<boost_error<<" protection="<<protection
             <<" components="<<components<<" neutral="<<neutral<<"\n";
    obs_data_set_int(settings,"resolution",0);
    obs_source_update(filter,settings);
    bool native_rendered=false;
    for(int i=0;i<600;++i){
        if(!render(source,out))break;
        if(drawn_at(filter,W,H)&&changed(out,baseline)>W*H/10){native_rendered=true;break;}
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
    }
    std::cout<<"native_resolution_transformed="<<native_rendered<<"\n";
    // Input controls change the worker's conditioning, not the compositor.
    // Wait for the newly submitted settings to produce actual drawn pixels.
    obs_data_set_double(settings,"structure_preservation",.5);
    obs_data_set_double(settings,"input_clarity",.65);
    obs_data_set_double(settings,"shadow_recovery",.5);
    obs_source_update(filter,settings);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    bool conditioned=false;
    for(int i=0;i<600;++i){
        if(!render(source,out))break;
        if(drawn_at(filter,W,H)){
            calldata_t args;calldata_init(&args);
            const bool matched=proc_handler_call(obs_source_get_proc_handler(filter),"sight_neural_rendering_input_stats_v1",&args)
                &&calldata_int(&args,"structure")==50&&calldata_int(&args,"clarity")==65
                &&calldata_int(&args,"shadows")==50;
            calldata_free(&args);
            if(matched&&changed(out,baseline)>W*H/10){conditioned=true;break;}
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(15));
    }
    std::cout<<"conditioned_inference="<<conditioned<<"\n";
    obs_data_set_double(settings,"structure_preservation",1.);
    obs_data_set_double(settings,"input_clarity",0.);
    obs_data_set_double(settings,"shadow_recovery",0.);
    // Switch generation, submit old frame once, then change the producer
    // before its CUDA worker returns the result. The locked result must
    // display the old input, never the new frame with the old correction.
    obs_data_set_int(settings,"sync_mode",2);
    obs_data_set_int(settings,"resolution",320);
    obs_source_update(filter,settings);
    for(int i=0;i<40;++i){
        if(!render(source,out))break;
        if(!drawn_at(filter,W,H))break;
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
    }
    // Wait for a request captured under the new generation (stats below).
    bool submitted=false;
    for(int i=0;i<100;++i){
        if(!render(source,out))break;
        calldata_t args;calldata_init(&args);
        if(proc_handler_call(obs_source_get_proc_handler(filter),"sight_neural_rendering_sync_stats_v1",&args))
            submitted=calldata_int(&args,"submitted_mode")==2&&calldata_int(&args,"submitted_sequence")>0;
        calldata_free(&args);
        if(submitted)break;
        std::this_thread::sleep_for(std::chrono::milliseconds(15));
    }
    alternate=true;
    bool locked=false;
    double locked_old=255.,locked_new=0.;
    // A mode/resolution change starts a new worker. The production first
    // request allows 30s startup plus the 3s small-frame exchange timeout.
    // Wait on the same deadline rather than counting render-loop iterations.
    const auto locked_deadline=std::chrono::steady_clock::now()+std::chrono::seconds(35);
    while(submitted&&std::chrono::steady_clock::now()<locked_deadline){
        if(!render(source,out))break;
        if(drawn_at(filter,320,180)){
            locked_old=mean_rgb_error(out,baseline);
            locked_new=mean_rgb_error(out,alternate_baseline);
            locked=locked_old<30.&&locked_new>locked_old+40.;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(15));
    }
    std::cout<<"frame_locked="<<locked<<" old_error="<<locked_old<<" new_error="<<locked_new<<"\n";
    // Switch back to a moving live input. The first correction must be
    // associated with the original scene, then gated on a cut.
    alternate=false;
    obs_data_set_int(settings,"sync_mode",1);
    obs_source_update(filter,settings);
    bool motion_rendered=false;
    for(int i=0;i<600;++i){
        if(!render(source,out))break;
        if(drawn_at(filter,320,180)&&mean_rgb_error(out,baseline)>0.4){motion_rendered=true;break;}
        std::this_thread::sleep_for(std::chrono::milliseconds(15));
    }
    alternate=true;
    // First live frame after cut is allowed to be uncorrected. Old-scene
    // lighting must not remain attached to the new image.
    bool cut_gated=false;
    double cut_error=255.;
    for(int i=0;i<12&&motion_rendered;++i){
        if(!render(source,out))break;
        cut_error=mean_rgb_error(out,alternate_baseline);
        if(cut_error<0.5){cut_gated=true;break;}
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    std::cout<<"motion_rendered="<<motion_rendered<<" scene_cut_gated="<<cut_gated<<" cut_error="<<cut_error<<"\n";
    alternate=false;
    obs_data_set_double(settings,"strength",0.);
    obs_source_update(filter,settings);
    for(int i=0;i<40;++i) {
        if(render(source,out)&&changed(out,baseline)==0) {passed=true;break;}
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
    }
    std::cout<<"zero_strength_passthrough="<<passed<<"\n";
    obs_source_filter_remove(source,filter);obs_source_release(filter);
    // New instance with a missing trusted configuration must pass through
    // repeatedly, not retain a previous instance's texture or spawn a shell.
    setenv("SIGHT_NR_CONFIG","/dev/null/not-a-config.json",1);
    obs_data_set_double(settings,"strength",1.);
    auto *missing=obs_source_create(id,"Missing backend",settings,nullptr);
    bool missing_pass=missing!=nullptr;
    if(missing){
        obs_source_filter_add(source,missing);
        for(int i=0;i<20;++i){
            if(!render(source,out)||changed(out,baseline)!=0){missing_pass=false;break;}
            std::this_thread::sleep_for(std::chrono::milliseconds(25));
        }
        obs_source_filter_remove(source,missing);obs_source_release(missing);
    }
    std::cout<<"missing_backend_passthrough="<<missing_pass<<"\n";
    obs_data_release(settings);obs_source_release(source);drain_ui();obs_shutdown();
    const auto nested=nested_draws.load();
    std::cout<<"custom_draw="<<custom<<" nested_effect_calls="<<nested<<"\n";
    if(nested)std::cerr<<"FAIL: custom source entered under an already active effect\n";
    return transformed&&detail&&amplification&&protection&&components&&neutral&&native_rendered&&conditioned&&locked&&motion_rendered&&cut_gated&&passed&&missing_pass&&!nested?0:1;
}
