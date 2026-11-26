#include "common/preferences.hpp"
#include "live2d/native_model.hpp"
#include "core/tracking_discovery.hpp"
#include <obs-module.h>
#include <graphics/graphics.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <numeric>
using namespace sight::live2d;
namespace {
using Clock=std::chrono::steady_clock;
constexpr char shader[]=R"EFFECT(
uniform float4x4 ViewProj;
uniform texture2d atlas;
uniform texture2d clip_image;
uniform float opacity;
uniform float4 multiply_color;
uniform float4 screen_color;
uniform int clip_mode;
sampler_state linear_clamp {Filter=Linear;AddressU=Clamp;AddressV=Clamp;};
struct Vert {float4 pos:POSITION;float2 uv:TEXCOORD0;};
struct Out {float4 pos:POSITION;float2 uv:TEXCOORD0;float2 screen_uv:TEXCOORD1;};
Out VS(Vert v){Out o;o.pos=mul(float4(v.pos.xyz,1),ViewProj);o.uv=v.uv;o.screen_uv=v.pos.xy;return o;}
float4 Mask(Out v):TARGET {
    float a=atlas.Sample(linear_clamp,v.uv).a;
    return float4(a,a,a,a);
}
float4 Color(Out v):TARGET {
    float4 c=atlas.Sample(linear_clamp,v.uv);
    c.rgb=saturate(c.rgb*multiply_color.rgb);
    c.rgb=saturate(c.rgb+screen_color.rgb-c.rgb*screen_color.rgb);
    float mask=1.0;
    if(clip_mode!=0){
        mask=clip_image.Sample(linear_clamp,v.screen_uv).a;
        if(clip_mode==2)mask=1-mask;
    }
    c.a*=saturate(opacity)*mask;
    return float4(c.rgb*c.a,c.a);
}
technique MaskDraw {pass {vertex_shader=VS(v);pixel_shader=Mask(v);}}
technique Draw {pass {vertex_shader=VS(v);pixel_shader=Color(v);}}
)EFFECT";
struct Config {
    std::string path,channel="face",motion_group="Idle",expression;
    uint32_t width=720,height=1280;
    float zoom=1,x=0,y=0,ease=100;
    bool idle=true;
};
struct GPU {
    std::vector<gs_texture_t*> images;
    std::vector<gs_vertbuffer_t*> streams;
    std::vector<gs_texrender_t*> masks;
    gs_effect_t* effect=nullptr;
    gs_texrender_t* target=nullptr;
    ~GPU(){
        for(auto* p:streams)if(p)gs_vertexbuffer_destroy(p);
        for(auto* p:images)if(p)gs_texture_destroy(p);
        for(auto* p:masks)if(p)gs_texrender_destroy(p);
        if(effect)gs_effect_destroy(effect);if(target)gs_texrender_destroy(target);
    }
};
struct Source {
    obs_source_t* context=nullptr;
    std::mutex mutex;std::condition_variable cv;
    Config config;bool stop=false,staged_ready=false;
    uint64_t generation=0,staged_generation=0;
    std::atomic<uint32_t> width{720},height{1280};
    std::unique_ptr<NativeModel> staged,model;
    std::unique_ptr<GPU> gpu;
    std::thread worker;
    std::string status="Choose a local .model3.json";
    std::vector<std::string> motion_groups,expression_names;
    uint64_t loaded_generation=0;
    Clock::time_point last{};
};
void loader(Source* s){
    uint64_t handled=0;
    for(;;){
        std::string path;uint64_t gen;
        {std::unique_lock lock(s->mutex);s->cv.wait(lock,[&]{return s->stop||s->generation!=handled;});
         if(s->stop)return;gen=s->generation;path=s->config.path;}
        std::string error;
        auto next=path.empty()?nullptr:load_native(path,error);
        {std::lock_guard lock(s->mutex);handled=gen;if(s->stop)return;if(gen!=s->generation)continue;
         s->staged=std::move(next);s->staged_ready=true;s->staged_generation=gen;
         s->status=path.empty()?"Choose a local .model3.json":s->staged?"Model decoded; waiting for graphics":error;
         s->motion_groups.clear();s->expression_names.clear();
         if(s->staged){
             for(const auto& [name,clip]:s->staged->motions)s->motion_groups.push_back(name);
             for(const auto& [name,expr]:s->staged->expressions)s->expression_names.push_back(name);
         }
         if(!error.empty())blog(LOG_WARNING,"[Sight/Live2D] %s",error.c_str());}
    }
}
std::unique_ptr<GPU> make_gpu(NativeModel& model){
    auto g=std::make_unique<GPU>();
    char* error=nullptr;g->effect=gs_effect_create(shader,"sight-native-live2d.effect",&error);
    if(error){blog(LOG_WARNING,"[Sight/Live2D] %s",error);bfree(error);}
    if(!g->effect)return {};
    g->target=gs_texrender_create(GS_RGBA,GS_ZS_NONE);if(!g->target)return {};
    for(auto& image:model.images){
        const uint8_t* pixels=image.rgba.data();
        auto* t=gs_texture_create(image.width,image.height,GS_RGBA,1,&pixels,0);if(!t)return {};
        g->images.push_back(t);
    }
    for(const auto& d:model.drawables){
        if(d.indices.empty()){g->streams.push_back(nullptr);continue;}
        auto* v=gs_vbdata_create();v->num=d.indices.size();v->points=static_cast<vec3*>(bzalloc(v->num*sizeof(vec3)));
        v->num_tex=1;v->tvarray=static_cast<gs_tvertarray*>(bzalloc(sizeof(gs_tvertarray)));
        v->tvarray[0].width=2;v->tvarray[0].array=bzalloc(v->num*sizeof(vec2));
        auto* uv=static_cast<vec2*>(v->tvarray[0].array);
        for(size_t i=0;i<v->num;++i){auto t=d.uv[d.indices[i]];vec2_set(&uv[i],t.X,1.f-t.Y);}
        auto* stream=gs_vertexbuffer_create(v,GS_DYNAMIC);if(!stream)return {};
        g->streams.push_back(stream);
    }
    for(size_t i=0;i<model.masks.size();++i){
        auto* t=gs_texrender_create(GS_RGBA,GS_ZS_NONE);if(!t)return {};g->masks.push_back(t);
    }
    return g;
}
void defaults(obs_data_t* d){
    obs_data_set_default_string(d,"model_path","");obs_data_set_default_string(d,"face_channel","face");
    obs_data_set_default_int(d,"width",720);obs_data_set_default_int(d,"height",1280);
    obs_data_set_default_double(d,"zoom",1);obs_data_set_default_double(d,"offset_x",0);obs_data_set_default_double(d,"offset_y",0);
    obs_data_set_default_double(d,"ease_ms",100);obs_data_set_default_bool(d,"idle",true);obs_data_set_default_string(d,"motion_group","Idle");obs_data_set_default_string(d,"expression","");
}
float setting(obs_data_t* d,const char* key,float min,float max,float fallback){
    float v=obs_data_get_double(d,key);return std::isfinite(v)?std::clamp(v,min,max):fallback;
}
void update(void* data,obs_data_t* d){
    auto* s=static_cast<Source*>(data);Config c;
    c.path=obs_data_get_string(d,"model_path");c.channel=obs_data_get_string(d,"face_channel");
    c.width=std::clamp(obs_data_get_int(d,"width"),128LL,1920LL);
    c.height=std::clamp(obs_data_get_int(d,"height"),128LL,2160LL);
    c.zoom=setting(d,"zoom",.25f,3.f,1);c.x=setting(d,"offset_x",-1,1,0);c.y=setting(d,"offset_y",-1,1,0);
    c.ease=setting(d,"ease_ms",0,500,100);c.idle=obs_data_get_bool(d,"idle");
    c.motion_group=obs_data_get_string(d,"motion_group");c.expression=obs_data_get_string(d,"expression");
    s->width=c.width;s->height=c.height;
    bool changed=false;{std::lock_guard lock(s->mutex);changed=c.path!=s->config.path;
        s->config=std::move(c);if(changed){++s->generation;s->status="Loading local Cubism model";}}
    if(changed)s->cv.notify_one();
}
void* create(obs_data_t* d,obs_source_t* context){
    auto s=std::make_unique<Source>();s->context=context;update(s.get(),d);s->worker=std::thread(loader,s.get());return s.release();
}
void destroy(void* data){
    auto* s=static_cast<Source*>(data);
    {std::lock_guard lock(s->mutex);s->stop=true;}s->cv.notify_one();if(s->worker.joinable())s->worker.join();
    obs_enter_graphics();s->gpu.reset();obs_leave_graphics();delete s;
}
const char* name(void*){return "LLL.sight Live2D Avatar (Native)";}
uint32_t width(void* d){return static_cast<Source*>(d)->width;}
uint32_t height(void* d){return static_cast<Source*>(d)->height;}
obs_properties_t* properties(void* data){
    auto* p=obs_properties_create();
    obs_properties_add_path(p,"model_path","Local Live2D model",OBS_PATH_FILE,"Live2D (*.model3.json)",sight_preferences::directory().c_str());
    auto* source=static_cast<Source*>(data);
    sight::discovery::add_picker(p,source?source->context:nullptr,"face_channel",
         "Sight Face Motion Channel","face",sight::TrackingKind::Face);
    obs_properties_add_int(p,"width","Output Width",128,1920,1);obs_properties_add_int(p,"height","Output Height",128,2160,1);
    obs_properties_add_float_slider(p,"zoom","Zoom",.25,3,.05);
    obs_properties_add_float_slider(p,"offset_x","Horizontal Offset",-1,1,.01);
    obs_properties_add_float_slider(p,"offset_y","Vertical Offset",-1,1,.01);
    obs_properties_add_float_slider(p,"ease_ms","Tracking Ease (ms)",0,500,10);
    obs_properties_add_bool(p,"idle","Idle animation / Blink and Breathing");
    auto* motion=obs_properties_add_list(p,"motion_group","Motion group (first clip, loop)",OBS_COMBO_TYPE_EDITABLE,OBS_COMBO_FORMAT_STRING);
    auto* expression=obs_properties_add_list(p,"expression","Expression",OBS_COMBO_TYPE_EDITABLE,OBS_COMBO_FORMAT_STRING);
    obs_property_list_add_string(motion,"Procedural idle only","");
    obs_property_list_add_string(expression,"None","");
    std::string status="Choose a local .model3.json";
    if(data){
        auto* s=static_cast<Source*>(data);std::lock_guard lock(s->mutex);status=s->status;
        for(const auto& name:s->motion_groups)obs_property_list_add_string(motion,name.c_str(),name.c_str());
        for(const auto& name:s->expression_names)obs_property_list_add_string(expression,name.c_str(),name.c_str());
    }
    obs_properties_add_text(p,"status",status.c_str(),OBS_TEXT_INFO);
    return p;
}
// All model resources are decoded by the loader. No SDK GL context or
// readback; draw Core's deformed art meshes with libobs-owned GPU objects.
void render(void* data,gs_effect_t*){
    auto* s=static_cast<Source*>(data);
    Config c;uint64_t generation;std::unique_ptr<NativeModel> next;bool has=false;
    {std::unique_lock lock(s->mutex,std::try_to_lock);if(!lock)return;
     c=s->config;generation=s->generation;
     if(s->staged_ready&&s->staged_generation==generation){next=std::move(s->staged);s->staged_ready=false;has=true;}}
    if(s->loaded_generation!=generation||has){s->gpu.reset();s->model.reset();s->loaded_generation=generation;}
    if(next){
        s->gpu=make_gpu(*next);
        if(s->gpu){
            blog(LOG_INFO,"[Sight/Live2D] SIGHT_NATIVE_LIVE2D_20261118: Core 6.0.1, %zu meshes, %zu mask groups",next->drawables.size(),next->masks.size());
            {std::unique_lock lock(s->mutex,std::try_to_lock);if(lock&&s->generation==generation)s->status="Ready: "+std::to_string(next->drawables.size())+" meshes, "+std::to_string(next->rig.size())+" parameters";}
            s->model=std::move(next);s->last={};
        }else{std::unique_lock lock(s->mutex,std::try_to_lock);if(lock)s->status="GPU avatar initialization failed";}
    }
    if(!s->model||!s->gpu)return;
    auto& m=*s->model;auto& g=*s->gpu;auto& api=m.core;
    auto now=Clock::now();float dt=s->last==Clock::time_point{}?1.f/30:std::chrono::duration<float>(now-s->last).count();s->last=now;
    sight::TrackingSignal signal;
    const bool fresh=sight::TrackingSignalBus::read(c.channel,std::chrono::milliseconds(750),signal);
    m.animate(fresh?&signal:nullptr,dt,c.idle,c.ease,c.motion_group,c.expression);
    auto positions=api.csmGetDrawableVertexPositions(m.model);
    const float scale=.91f*c.zoom*std::min(float(c.width)/m.canvas.X,float(c.height)/m.canvas.Y);
    for(size_t i=0;i<m.drawables.size();++i){
        auto* vb=g.streams[i];if(!vb)continue;auto* dest=gs_vertexbuffer_get_data(vb)->points;
        const auto& d=m.drawables[i];
        for(size_t j=0;j<d.indices.size();++j){
            auto p=positions[i][d.indices[j]];
            if(!std::isfinite(p.X)||!std::isfinite(p.Y))return;
            vec3_set(dest+j,.5f+c.x+(p.X*m.pixels_per_unit+m.origin.X-m.canvas.X*.5f)*scale/c.width,
                        .5f+c.y+(-p.Y*m.pixels_per_unit+m.origin.Y-m.canvas.Y*.5f)*scale/c.height,0);
        }
        gs_vertexbuffer_flush(vb);
    }
    auto param=[&](const char* n){return gs_effect_get_param_by_name(g.effect,n);};
    auto begin=[&](gs_texrender_t* rt,uint32_t w,uint32_t h){
        gs_texrender_reset(rt);if(!gs_texrender_begin(rt,w,h))return false;
        vec4 clear{};gs_clear(GS_CLEAR_COLOR,&clear,0,0);
        gs_matrix_push();gs_matrix_identity();gs_projection_push();gs_ortho(0,1,0,1,-100,100);
        return true;
    };
    auto end=[&](gs_texrender_t* rt){gs_load_vertexbuffer(nullptr);gs_projection_pop();gs_matrix_pop();gs_texrender_end(rt);};
    gs_blend_state_push();gs_enable_blending(true);gs_blend_function(GS_BLEND_ONE,GS_BLEND_INVSRCALPHA);
    const auto old_cull=gs_get_cull_mode();gs_set_cull_mode(GS_NEITHER);
    auto draw=[&](size_t i,const char* technique){
        if(!g.streams[i])return;
        gs_load_vertexbuffer(g.streams[i]);gs_load_indexbuffer(nullptr);
        gs_effect_set_texture(param("atlas"),g.images[m.drawables[i].texture]);
        while(gs_effect_loop(g.effect,technique))gs_draw(GS_TRIS,0,uint32_t(m.drawables[i].indices.size()));
    };
    bool ok=true;
    const float mask_scale=std::min(1.f,1024.f/std::max(c.width,c.height));
    const uint32_t mw=std::max(1u,uint32_t(c.width*mask_scale)),mh=std::max(1u,uint32_t(c.height*mask_scale));
    for(size_t i=0;i<m.masks.size();++i){
        if(!begin(g.masks[i],mw,mh)){ok=false;break;}
        for(auto j:m.masks[i])draw(j,"MaskDraw");
        end(g.masks[i]);
    }
    if(ok&&begin(g.target,c.width,c.height)){
        auto order=api.csmGetDrawableDrawOrders(m.model);auto flags=api.csmGetDrawableDynamicFlags(m.model);
        auto alpha=api.csmGetDrawableOpacities(m.model);auto mul=api.csmGetDrawableMultiplyColors(m.model);
        auto screen=api.csmGetDrawableScreenColors(m.model);
        std::vector<size_t> sorted(m.drawables.size());std::iota(sorted.begin(),sorted.end(),0);
        std::stable_sort(sorted.begin(),sorted.end(),[&](size_t a,size_t b){return order[a]<order[b];});
        for(size_t i:sorted){
            const auto& d=m.drawables[i];if(!(flags[i]&csmIsVisible)||!std::isfinite(alpha[i])||alpha[i]<=0)continue;
            if(d.blend==1)gs_blend_function_separate(GS_BLEND_ONE,GS_BLEND_ONE,GS_BLEND_ZERO,GS_BLEND_ONE);
            else if(d.blend==2)gs_blend_function_separate(GS_BLEND_DSTCOLOR,GS_BLEND_INVSRCALPHA,GS_BLEND_ZERO,GS_BLEND_ONE);
            else gs_blend_function(GS_BLEND_ONE,GS_BLEND_INVSRCALPHA);
            gs_effect_set_float(param("opacity"),alpha[i]);
            vec4 mc,sc;vec4_set(&mc,mul[i].X,mul[i].Y,mul[i].Z,mul[i].W);vec4_set(&sc,screen[i].X,screen[i].Y,screen[i].Z,screen[i].W);
            gs_effect_set_vec4(param("multiply_color"),&mc);gs_effect_set_vec4(param("screen_color"),&sc);
            gs_effect_set_int(param("clip_mode"),d.mask<0?0:d.inverted?2:1);
            gs_effect_set_texture(param("clip_image"),d.mask<0?g.images[d.texture]:gs_texrender_get_texture(g.masks[d.mask]));
            draw(i,"Draw");
        }
        end(g.target);
        gs_blend_function(GS_BLEND_ONE,GS_BLEND_INVSRCALPHA);
        auto* t=gs_texrender_get_texture(g.target);auto* e=obs_get_base_effect(OBS_EFFECT_DEFAULT);
        gs_effect_set_texture(gs_effect_get_param_by_name(e,"image"),t);
        while(gs_effect_loop(e,"Draw"))gs_draw_sprite(t,0,c.width,c.height);
    }
    gs_set_cull_mode(old_cull);gs_blend_state_pop();
}
}
struct obs_source_info sight_native_live2d_source_info={
    .id="sight_native_live2d_source",.type=OBS_SOURCE_TYPE_INPUT,
    .output_flags=OBS_SOURCE_VIDEO|OBS_SOURCE_CUSTOM_DRAW,
    .get_name=name,.create=create,.destroy=destroy,.get_width=width,.get_height=height,
    .get_defaults=defaults,.get_properties=properties,.update=update,.video_render=render,
};
