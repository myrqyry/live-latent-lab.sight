#include "core/tracking_signal_bus.hpp"
#include <obs-module.h>
#include <graphics/graphics.h>
#include <chrono>
#include <thread>
#include <iostream>
#include <vector>
#include <cstring>
#include <cmath>
#include <fstream>
#include <cstdlib>
#include <algorithm>
#include <filesystem>
using Frame=std::vector<uint8_t>;
constexpr unsigned W=512,H=768;
static Frame draw(obs_source_t* source){
    Frame out;
    obs_enter_graphics();
    auto* rt=gs_texrender_create(GS_RGBA,GS_ZS_NONE);
    if(rt&&gs_texrender_begin(rt,W,H)){
        vec4 clear{};gs_clear(GS_CLEAR_COLOR,&clear,0,0);
        gs_projection_push();gs_matrix_push();gs_matrix_identity();gs_ortho(0,float(W),0,float(H),-100,100);
        gs_blend_state_push();gs_enable_blending(true);
        gs_blend_function(GS_BLEND_ONE,GS_BLEND_INVSRCALPHA);
        obs_source_video_render(source);
        gs_blend_state_pop();gs_matrix_pop();gs_projection_pop();gs_texrender_end(rt);
        auto* stage=gs_stagesurface_create(W,H,GS_RGBA);
        if(stage){
            gs_stage_texture(stage,gs_texrender_get_texture(rt));
            uint8_t* p=nullptr;uint32_t stride=0;
            if(gs_stagesurface_map(stage,&p,&stride)){
                if(stride>=W*4){out.resize(W*H*4);for(unsigned y=0;y<H;++y)memcpy(out.data()+y*W*4,p+y*stride,W*4);}
                gs_stagesurface_unmap(stage);
            }gs_stagesurface_destroy(stage);
        }
    }
    if(rt)gs_texrender_destroy(rt);
    obs_leave_graphics();return out;
}
static size_t visible(const Frame& f){size_t n=0;for(size_t i=3;i<f.size();i+=4)n+=f[i]>80;return n;}
static size_t changed(const Frame& a,const Frame& b){if(a.size()!=b.size()||a.empty())return 0;size_t n=0;for(size_t i=0;i<a.size();i+=4){bool diff=false;for(size_t c=0;c<4;++c)diff|=std::abs(int(a[i+c])-int(b[i+c]))>12;n+=diff;}return n;}
static bool transparent(const Frame& f){return f.size()==W*H*4&&f[3]==0&&f[W*4-1]==0&&f[f.size()-1]==0;}
static bool solid_color(const Frame& f){size_t n=0;for(size_t i=0;i<f.size();i+=4)n+=f[i+3]>200&&(std::abs(int(f[i])-int(f[i+2]))>15);return n>500;}
int main(int argc,char** argv){
    if(argc!=4||!obs_startup("en-US",nullptr,nullptr))return 2;
    obs_video_info v{};v.graphics_module="libobs-opengl";v.fps_num=30;v.fps_den=1;
    v.base_width=W;v.base_height=H;v.output_width=W;v.output_height=H;v.output_format=VIDEO_FORMAT_RGBA;v.scale_type=OBS_SCALE_DISABLE;
    if(obs_reset_video(&v)!=OBS_VIDEO_SUCCESS){obs_shutdown();return 3;}
    obs_module_t* mod=nullptr;
    if(obs_open_module(&mod,argv[1],argv[3])!=MODULE_SUCCESS||!obs_init_module(mod)){obs_shutdown();return 4;}
    auto make=[&](const char* name,const char* channel,const char* path){
        auto* c=obs_data_create();obs_data_set_string(c,"model_path",path);obs_data_set_string(c,"face_channel",channel);
        obs_data_set_int(c,"width",W);obs_data_set_int(c,"height",H);obs_data_set_bool(c,"idle",false);
        auto* s=obs_source_create("sight_native_live2d_source",name,c,nullptr);obs_data_release(c);return s;
    };
    auto* a=make("native Live2D A","live2d-a",argv[2]);
    auto* b=make("native Live2D B","live2d-b",argv[2]);
    bool registered=a&&b&&obs_source_get_width(a)==W&&obs_source_get_height(a)==H;
    const auto filename=std::filesystem::path(argv[2]).filename().string();
    Frame neutral,other,pose,missing;
    auto wait=[&](obs_source_t* s,Frame& frame,auto predicate){for(int i=0;i<100;++i){frame=draw(s);if(predicate(frame))return true;std::this_thread::sleep_for(std::chrono::milliseconds(45));}return false;};
    if(filename=="Ren.model3.json"){
        bool rejected=registered&&wait(a,missing,[&](const Frame& f){
            auto* props=obs_source_properties(a);auto* info=props?obs_properties_get(props,"status"):nullptr;
            const std::string status=info?obs_property_description(info):"";
            if(props)obs_properties_destroy(props);
            return f.size()==W*H*4&&visible(f)==0&&status.find("offscreen compositing")!=std::string::npos;
        });
        std::cout<<"SIGHT_NATIVE_LIVE2D_UNSUPPORTED rejected="<<rejected<<std::endl;
        if(a)obs_source_release(a);if(b)obs_source_release(b);obs_shutdown();return !rejected;
    }
    bool loaded=registered&&wait(a,neutral,[&](const Frame& f){return transparent(f)&&solid_color(f)&&visible(f)>3000&&visible(f)<W*H*9/10;});
    bool second=loaded&&wait(b,other,[&](const Frame& f){return transparent(f)&&visible(f)>3000;});
    sight::TrackingData face;face.landmarks.resize(478);
    face.blendshape_names={"jawOpen","eyeBlinkLeft","eyeBlinkRight","mouthSmileLeft","mouthSmileRight"};
    face.blendshapes={.95f,.95f,.8f,.8f,.8f};face.has_head_transform=true;
    face.head_transform={.8660254f,0,.5f,0,0,1,0,0,-.5f,0,.8660254f,0,0,0,0,1};
    auto owner=sight::TrackingSignalBus::new_publisher_id();
    bool tracking=false,independent=false,loss=false,invalid=false,reload=false,resize=false;
    size_t delta=0;
    if(second){
        if(auto* path=std::getenv("SIGHT_LIVE2D_RGBA")){
            std::ofstream f(path,std::ios::binary);f.write(reinterpret_cast<const char*>(neutral.data()),neutral.size());
        }
        sight::TrackingSignalBus::publish("live2d-a",owner,sight::TrackingKind::Face,face,W,H);
        tracking=wait(a,pose,[&](const Frame& f){delta=changed(neutral,f);return delta>500&&visible(f)>3000;});
        independent=changed(draw(b),other)==0;
        sight::TrackingSignalBus::withdraw("live2d-a",owner);
        loss=wait(a,missing,[&](const Frame& f){return changed(neutral,f)<20;});
        auto* c=obs_data_create();obs_data_set_int(c,"width",256);obs_data_set_int(c,"height",384);
        obs_source_update(a,c);obs_data_release(c);
        resize=wait(a,missing,[&](const Frame& f){
            return obs_source_get_width(a)==256&&obs_source_get_height(a)==384&&visible(f)>500&&visible(f)<visible(neutral)/2;
        });
        c=obs_data_create();obs_data_set_int(c,"width",W);obs_data_set_int(c,"height",H);
        obs_data_set_string(c,"model_path","/no-such/sight.model3.json");obs_source_update(a,c);obs_data_release(c);
        invalid=wait(a,missing,[](const Frame& f){return f.size()==W*H*4&&visible(f)==0;});
        c=obs_data_create();obs_data_set_string(c,"model_path",argv[2]);obs_source_update(a,c);obs_data_release(c);
        reload=wait(a,missing,[&](const Frame& f){return transparent(f)&&visible(f)>3000;});
    }
    bool authored=true,expression=true,cleared=true,choices=true;
    size_t motion_pixels=0,expression_pixels=0;
    if(filename=="Haru.model3.json"){
        authored=expression=cleared=choices=false;
        if(reload){
            // Properties are a snapshot, refreshed when reopened; verify the
            // choices describe successfully parsed authored resources.
            auto* props=obs_source_properties(a);
            auto has=[&](const char* property,const char* val){
                auto* list=props?obs_properties_get(props,property):nullptr;
                if(!list)return false;
                for(size_t i=0;i<obs_property_list_item_count(list);++i)
                    if(std::string(obs_property_list_item_string(list,i))==val)return true;
                return false;
            };
            choices=has("motion_group","Idle")&&has("motion_group","TapBody")&&has("expression","F03");
            if(props)obs_properties_destroy(props);
            Frame stable=draw(a);
            auto* c=obs_data_create();obs_data_set_string(c,"expression","F03");
            obs_source_update(a,c);obs_data_release(c);
            expression=wait(a,pose,[&](const Frame& f){expression_pixels=changed(stable,f);return expression_pixels>10&&visible(f)>3000;});
            c=obs_data_create();obs_data_set_string(c,"expression","");
            obs_source_update(a,c);obs_data_release(c);
            cleared=wait(a,pose,[&](const Frame& f){return changed(stable,f)<10;});
            c=obs_data_create();obs_data_set_bool(c,"idle",true);obs_data_set_string(c,"motion_group","TapBody");obs_data_set_double(c,"ease_ms",0);
            obs_source_update(a,c);obs_data_release(c);
            // Authored clip is substantially more than the procedural two-degree
            // sway and blink. Check changed visible geometry plus temporal change.
            Frame first;
            bool moving=wait(a,first,[&](const Frame& f){motion_pixels=changed(stable,f);return motion_pixels>3500&&visible(f)>3000;});
            authored=moving&&wait(a,pose,[&](const Frame& f){return changed(first,f)>300&&visible(f)>3000;});
            c=obs_data_create();obs_data_set_bool(c,"idle",false);obs_source_update(a,c);obs_data_release(c);
            cleared=cleared&&wait(a,pose,[&](const Frame& f){return changed(stable,f)<10;});
        }
        std::cout<<"SIGHT_LIVE2D_AUTHORED choices="<<choices<<" motion="<<authored<<" motion_pixels="<<motion_pixels
                 <<" expression="<<expression<<" expression_pixels="<<expression_pixels<<" cleared="<<cleared<<std::endl;
    }
    std::cout<<"SIGHT_NATIVE_LIVE2D registered="<<registered<<" loaded="<<loaded<<" pixels="<<visible(neutral)
        <<" second="<<second<<" tracked="<<tracking<<" changed="<<delta<<" independent="<<independent
        <<" loss="<<loss<<" resize="<<resize<<" invalid="<<invalid<<" reload="<<reload<<std::endl;
    if(a)obs_source_release(a);if(b)obs_source_release(b);obs_shutdown();
    return !(registered&&loaded&&second&&tracking&&independent&&loss&&resize&&invalid&&reload&&authored&&expression&&cleared&&choices);
}
