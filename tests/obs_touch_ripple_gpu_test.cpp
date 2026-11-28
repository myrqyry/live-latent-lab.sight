#include <obs.h>
#include <graphics/graphics.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

namespace {
constexpr uint32_t Width=320,Height=180;
using Frame=std::vector<uint8_t>;

Frame source_pixels() {
    Frame pixels(size_t(Width)*Height*4);
    for(uint32_t y=0;y<Height;++y)for(uint32_t x=0;x<Width;++x) {
        const size_t i=(size_t(y)*Width+x)*4;
        pixels[i]=uint8_t((x*7+y*3)%255);
        pixels[i+1]=uint8_t((x*3+y*5)%255);
        pixels[i+2]=uint8_t((x*5+y*7)%255);
        pixels[i+3]=255;
    }
    return pixels;
}
void scalar(gs_effect_t* fx,const char* name,float value) {
    auto* p=gs_effect_get_param_by_name(fx,name);
    if(p)gs_effect_set_float(p,value);
}
void vec(gs_effect_t* fx,const char* name,float x,float y,float z,float w) {
    auto* p=gs_effect_get_param_by_name(fx,name);
    if(!p)return;
    vec4 value{};
    vec4_set(&value,x,y,z,w);
    gs_effect_set_vec4(p,&value);
}
Frame render(gs_effect_t* effect,gs_texture_t* source,gs_texrender_t* render_target,
             float strength,float age,float intensity) {
    gs_set_cull_mode(GS_NEITHER);
    gs_texrender_reset(render_target);
    if(!gs_texrender_begin(render_target,Width,Height))return {};
    gs_projection_push();gs_matrix_push();gs_matrix_identity();
    gs_ortho(0,float(Width),0,float(Height),-100,100);
    auto* image=gs_effect_get_param_by_name(effect,"original_frame");
    if(image)gs_effect_set_texture(image,source);
    scalar(effect,"touch_strength",strength);
    scalar(effect,"touch_radius",.38f);
    scalar(effect,"touch_waves",2.5f);
    scalar(effect,"touch_speed",1.2f);
    scalar(effect,"touch_pinch_boost",0.f);
    scalar(effect,"touch_time",.5f);
    scalar(effect,"touch_aspect",float(Width)/Height);
    vec(effect,"fingertip0",.5f,.5f,0.f,0.f);
    vec(effect,"fingertip1",.5f,.5f,0.f,0.f);
    for(int i=0;i<8;++i) {
        char key[32];
        std::snprintf(key,sizeof(key),"ripple%d",i);
        vec(effect,key,.5f,.5f,age,i==0?intensity:0.f);
    }
    while(gs_effect_loop(effect,"Draw"))
        gs_draw_sprite(source,0,Width,Height);
    gs_matrix_pop();gs_projection_pop();
    gs_texrender_end(render_target);
    auto* stage=gs_stagesurface_create(Width,Height,GS_RGBA);
    Frame pixels;
    if(stage) {
        if(auto* result=gs_texrender_get_texture(render_target)) {
            gs_stage_texture(stage,result);
            uint8_t* data=nullptr;uint32_t stride=0;
            if(gs_stagesurface_map(stage,&data,&stride)) {
                pixels.resize(size_t(Width)*Height*4);
                for(uint32_t y=0;y<Height;++y)
                    std::memcpy(pixels.data()+size_t(y)*Width*4,
                                data+size_t(y)*stride,Width*4);
                gs_stagesurface_unmap(stage);
            }
        }
        gs_stagesurface_destroy(stage);
    }
    return pixels;
}
size_t changed(const Frame& a,const Frame& b) {
    if(a.size()!=b.size()||a.empty())return size_t(-1);
    size_t result=0;
    for(size_t i=0;i<a.size();i+=4)
        for(size_t k=0;k<3;++k)
            result+=std::abs(int(a[i+k])-int(b[i+k]))>2;
    return result;
}
}
int main(int argc,char** argv) {
    if(argc!=2)return 2;
    if(!obs_startup("en-US",nullptr,nullptr))return 1;
    obs_video_info video{};
    video.graphics_module="libobs-opengl";
    video.fps_num=30;video.fps_den=1;
    video.base_width=Width;video.base_height=Height;
    video.output_width=Width;video.output_height=Height;
    video.output_format=VIDEO_FORMAT_RGBA;
    video.scale_type=OBS_SCALE_DISABLE;
    if(obs_reset_video(&video)!=OBS_VIDEO_SUCCESS) {
        obs_shutdown();return 1;
    }
    obs_enter_graphics();
    char* errors=nullptr;
    auto* effect=gs_effect_create_from_file(argv[1],&errors);
    if(errors) {
        std::cerr<<"Shader compiler: "<<errors<<"\n";
        bfree(errors);
    }
    bool passed=false;
    if(effect) {
        const auto input=source_pixels();
        const uint8_t* image=input.data();
        auto* source=gs_texture_create(Width,Height,GS_RGBA,1,&image,0);
        auto* target=gs_texrender_create(GS_RGBA,GS_ZS_NONE);
        if(source&&target) {
            const auto clean=render(effect,source,target,1.2f,.3f,0.f);
            const auto ring_early=render(effect,source,target,1.2f,.20f,1.f);
            const auto ring_late=render(effect,source,target,1.2f,.65f,1.f);
            const auto off=render(effect,source,target,0.f,.20f,1.f);
            const auto expired=render(effect,source,target,1.2f,1.7f,1.f);
            const auto delta=changed(clean,ring_early);
            const auto progression=changed(ring_early,ring_late);
            const auto disabled=changed(clean,off);
            const auto aged=changed(clean,expired);
            bool opaque=!ring_early.empty();
            for(size_t i=3;i<ring_early.size();i+=4)
                opaque&=ring_early[i]==255;
            std::cout<<"ripple_gpu_diff="<<delta
                     <<" progression_diff="<<progression
                     <<" disabled_diff="<<disabled
                     <<" expired_diff="<<aged
                     <<" alpha_opaque="<<opaque<<"\n";
            passed=delta>1000 && progression>1000 && disabled==0 &&
                   aged==0 && opaque;
        }
        if(target)gs_texrender_destroy(target);
        if(source)gs_texture_destroy(source);
        gs_effect_destroy(effect);
    }
    obs_leave_graphics();obs_shutdown();
    std::cout<<(passed?"PASS: actual expanding GPU ripple with decay and passthrough":
                       "FAIL: GPU ripple not verified")<<"\n";
    return passed?0:1;
}
