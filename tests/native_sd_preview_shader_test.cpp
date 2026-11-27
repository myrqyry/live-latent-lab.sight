#include <obs.h>
#include <graphics/graphics.h>
#include <graphics/vec2.h>
#include <graphics/vec4.h>
#include <util/bmem.h>
#include "core/native_sd_preview.hpp"
#include "core/sight_flow_v1.h"
#include <iostream>
#include <vector>
#include <cstring>
#include <cmath>
namespace {
constexpr unsigned W=128,H=96;
using Frame=std::vector<uint8_t>;
Frame draw(gs_effect_t*fx,gs_texture_t*a,gs_texture_t*b,float phase,TransformParams t,gs_texture_t*flow=nullptr,float enabled=0){
    Frame out;
    auto*target=gs_texrender_create(GS_RGBA,GS_ZS_NONE);
    auto*stage=gs_stagesurface_create(W,H,GS_RGBA);
    if(target&&stage&&gs_texrender_begin(target,W,H)){
        gs_projection_push();gs_matrix_push();gs_matrix_identity();
        gs_ortho(0,float(W),0,float(H),-100,100);
        gs_set_cull_mode(GS_NEITHER);
        gs_blend_state_push();gs_blend_function(GS_BLEND_ONE,GS_BLEND_ZERO);
        gs_effect_set_texture(gs_effect_get_param_by_name(fx,"image"),b);
        gs_effect_set_texture(gs_effect_get_param_by_name(fx,"previous_image"),a);
        vec2 size;vec2_set(&size,W,H);
        gs_effect_set_vec2(gs_effect_get_param_by_name(fx,"image_size"),&size);
        auto m=sight::sd::preview_mapping(t,phase);vec4 va,vb;
        vec4_set(&va,m.old_scale,m.old_angle,m.old_pan_x,m.old_pan_y);
        vec4_set(&vb,m.new_scale,m.new_angle,m.new_pan_x,m.new_pan_y);
        gs_effect_set_vec4(gs_effect_get_param_by_name(fx,"old_transform"),&va);
        gs_effect_set_vec4(gs_effect_get_param_by_name(fx,"new_transform"),&vb);
        gs_effect_set_float(gs_effect_get_param_by_name(fx,"phase"),phase);
        if(auto*p=gs_effect_get_param_by_name(fx,"residual_flow"))gs_effect_set_texture(p,flow?flow:b);
        if(auto*p=gs_effect_get_param_by_name(fx,"flow_enabled"))gs_effect_set_float(p,enabled);
        while(gs_effect_loop(fx,"Draw"))gs_draw_sprite(b,0,W,H);
        gs_blend_state_pop();gs_matrix_pop();gs_projection_pop();gs_texrender_end(target);
        gs_stage_texture(stage,gs_texrender_get_texture(target));
        uint8_t*bytes=nullptr;uint32_t pitch=0;
        if(gs_stagesurface_map(stage,&bytes,&pitch)){
            out.resize(W*H*4);
            for(unsigned y=0;y<H;++y)std::memcpy(out.data()+y*W*4,bytes+y*pitch,W*4);
            gs_stagesurface_unmap(stage);
        }
    }
    if(stage)gs_stagesurface_destroy(stage);if(target)gs_texrender_destroy(target);
    return out;
}
}
int main(int argc,char**argv){
    if(argc!=2||!obs_startup("en-US",nullptr,nullptr))return 2;
    obs_video_info v{};v.graphics_module="libobs-opengl";v.fps_num=30;v.fps_den=1;
    v.base_width=v.output_width=W;v.base_height=v.output_height=H;v.output_format=VIDEO_FORMAT_RGBA;v.scale_type=OBS_SCALE_DISABLE;
    if(obs_reset_video(&v)!=OBS_VIDEO_SUCCESS){obs_shutdown();return 2;}
    obs_enter_graphics();
    char*errors=nullptr;auto*fx=gs_effect_create_from_file(argv[1],&errors);
    if(errors){std::cerr<<errors<<"\n";bfree(errors);}
    Frame a(W*H*4),b(a.size()),expected(a.size());
    for(unsigned y=0;y<H;++y)for(unsigned x=0;x<W;++x){
        auto i=(y*W+x)*4;
        a[i]=(x>32&&x<74&&y>20&&y<65)?225:25;
        a[i+1]=(x*13+y*7)%200+20;a[i+2]=(x*3+y*11)%180+20;a[i+3]=255;
    }
    TransformParams t;t.pan_x=12;t.pan_y=4;
    transform_image(a.data(),a.size(),W,H,b.data(),b.size(),t);
    auto mid=t;mid.pan_x/=2;mid.pan_y/=2;
    transform_image(a.data(),a.size(),W,H,expected.data(),expected.size(),mid);
    const uint8_t*ap=a.data(),*bp=b.data();
    auto*at=gs_texture_create(W,H,GS_RGBA,1,&ap,0);
    auto*bt=gs_texture_create(W,H,GS_RGBA,1,&bp,0);
    bool pass=false;
    if(fx&&at&&bt){
        auto zero=draw(fx,at,bt,0,t),one=draw(fx,at,bt,1,t);
        auto guided=draw(fx,at,bt,.5f,t),linear=draw(fx,at,bt,.5f,{});
        if(zero.size()==a.size()&&one.size()==a.size()&&guided.size()==a.size()&&linear.size()==a.size()){
            double e=0,l=0;size_t n=0;bool alpha=true;
            for(unsigned y=12;y<H-12;++y)for(unsigned x=16;x<W-16;++x){
                size_t i=(y*W+x)*4;
                alpha&=guided[i+3]==255;
                for(unsigned c=0;c<3;++c){
                    e+=std::abs(int(guided[i+c])-int(expected[i+c]));
                    l+=std::abs(int(linear[i+c])-int(expected[i+c]));++n;
                }
            }
            std::cout<<"translation_mae_crossfade="<<l/n<<" translation_mae_camera="<<e/n
                <<" endpoints="<<(zero==a&&one==b)<<" alpha="<<alpha<<"\n";
            pass=zero==a&&one==b&&alpha&&l/n>5&&e/n<1;
        }
    }
    // Camera moves right by 4px while the residual scene moves another
    // 8px right. The ideal midpoint is the original translated by 6px.
    // A deterministic grid isolates compositor math from estimation.
    TransformParams camera;camera.pan_x=4;camera.pan_y=4;
    auto actual=camera;actual.pan_x+=8;
    auto half=actual;half.pan_x*=.5f;half.pan_y*=.5f;
    Frame moving(a.size()),goal(a.size());
    transform_image(a.data(),a.size(),W,H,moving.data(),moving.size(),actual);
    transform_image(a.data(),a.size(),W,H,goal.data(),goal.size(),half);
    const uint8_t*mp=moving.data();
    auto*mt=gs_texture_create(W,H,GS_RGBA,1,&mp,0);
    std::vector<float> cells(SIGHT_FLOW_CELLS*4,0.f);
    for(size_t i=0;i<SIGHT_FLOW_CELLS;++i){cells[i*4]=-8.f/W;cells[i*4+2]=1.f;}
    const uint8_t*cp=reinterpret_cast<const uint8_t*>(cells.data());
    auto*grid=gs_texture_create(SIGHT_FLOW_WIDTH,SIGHT_FLOW_HEIGHT,GS_RGBA32F,1,&cp,0);
    bool residual_ok=false;
    if(fx&&at&&mt&&grid){
        auto ordinary=draw(fx,at,mt,.5f,camera,grid,0);
        auto motion=draw(fx,at,mt,.5f,camera,grid,1);
        auto z=draw(fx,at,mt,0,camera,grid,1),o=draw(fx,at,mt,1,camera,grid,1);
        if(ordinary.size()==goal.size()&&motion.size()==goal.size()){
            double before=0,after=0;size_t n=0;
            for(unsigned y=12;y<H-12;++y)for(unsigned x=20;x<W-20;++x)for(unsigned c=0;c<3;++c){
                size_t i=(y*W+x)*4+c;
                before+=std::abs(int(ordinary[i])-int(goal[i]));after+=std::abs(int(motion[i])-int(goal[i]));++n;
            }
            std::fill(cells.begin(),cells.end(),0.f);
            // Use a new zero-confidence texture for the fallback fixture.
            auto*empty=gs_texture_create(SIGHT_FLOW_WIDTH,SIGHT_FLOW_HEIGHT,GS_RGBA32F,1,&cp,0);
            auto fallback=empty?draw(fx,at,mt,.5f,camera,empty,1):Frame{};
            if(empty)gs_texture_destroy(empty);
            residual_ok=before/n>5&&after/n<1&&z==a&&o==moving&&fallback==ordinary;
            std::cout<<"residual_camera_mae="<<before/n<<" residual_flow_mae="<<after/n
                <<" endpoints="<<(z==a&&o==moving)<<" fallback="<<(fallback==ordinary)<<"\n";
        }
    }
    if(mt)gs_texture_destroy(mt);if(grid)gs_texture_destroy(grid);
    pass &= residual_ok;
    if(at)gs_texture_destroy(at);if(bt)gs_texture_destroy(bt);if(fx)gs_effect_destroy(fx);
    obs_leave_graphics();obs_shutdown();return pass?0:1;
}
