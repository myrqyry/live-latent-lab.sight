#include <obs.h>
#include <graphics/graphics.h>
#include <graphics/vec2.h>
#include <util/bmem.h>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>
namespace {
constexpr unsigned W=128,H=96,PW=32,PH=24;
using Frame=std::vector<uint8_t>;
Frame render(gs_effect_t*fx,gs_texture_t*src,gs_texture_t*delta,float guided,float strength=1.f,gs_texture_t*depth=nullptr,float depth_strength=0.f){
    auto*target=gs_texrender_create(GS_RGBA,GS_ZS_NONE);
    auto*stage=gs_stagesurface_create(W,H,GS_RGBA);
    Frame out;
    gs_set_cull_mode(GS_NEITHER);
    if(target&&stage&&gs_texrender_begin(target,W,H)){
        gs_projection_push();gs_matrix_push();gs_matrix_identity();
        gs_ortho(0,float(W),0,float(H),-100,100);
        gs_blend_state_push();gs_blend_function(GS_BLEND_ONE,GS_BLEND_ZERO);
        for(auto name:{"image","correction","reference_image","motion_grid"})
            gs_effect_set_texture(gs_effect_get_param_by_name(fx,name),std::strcmp(name,"image")==0?src:delta);
        for(auto [name,value]:{std::pair{"strength",strength},{"relighting",1.f},{"surface_detail",1.f},
                {"color_strength",1.f},{"highlight_protection",0.f},{"flow_enabled",0.f},{"edge_aware",guided}})
            if(auto*p=gs_effect_get_param_by_name(fx,name))gs_effect_set_float(p,value);
        if(auto*p=gs_effect_get_param_by_name(fx,"depth_image"))gs_effect_set_texture(p,depth?depth:src);
        if(auto*p=gs_effect_get_param_by_name(fx,"depth_guidance"))gs_effect_set_float(p,depth_strength);
        gs_effect_set_int(gs_effect_get_param_by_name(fx,"sync_mode"),2);
        vec2 size{};vec2_set(&size,PW,PH);
        gs_effect_set_vec2(gs_effect_get_param_by_name(fx,"correction_size"),&size);
        gs_effect_set_vec2(gs_effect_get_param_by_name(fx,"proxy_size"),&size);
        while(gs_effect_loop(fx,"Draw"))gs_draw_sprite(src,0,W,H);
        gs_blend_state_pop();gs_matrix_pop();gs_projection_pop();gs_texrender_end(target);
        gs_stage_texture(stage,gs_texrender_get_texture(target));
        uint8_t*bytes=nullptr;uint32_t pitch=0;
        if(gs_stagesurface_map(stage,&bytes,&pitch)){
            out.resize(W*H*4);
            for(unsigned y=0;y<H;++y)std::memcpy(out.data()+y*W*4,bytes+y*pitch,W*4);
            gs_stagesurface_unmap(stage);
        }
    }
    if(stage)gs_stagesurface_destroy(stage);
    if(target)gs_texrender_destroy(target);
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
    Frame src(W*H*4);
    std::vector<float> correction(PW*PH*4),constant(PW*PH*4);
    for(unsigned y=0;y<H;++y)for(unsigned x=0;x<W;++x){
        size_t i=(y*W+x)*4;
        // Sharp RGB edge aligned to proxy cell boundary; fine detail in blue.
        src[i]=x<W/2?50:190;src[i+1]=x<W/2?65:175;
        src[i+2]=uint8_t((x<W/2?70:165)+(y%2)*10);src[i+3]=255;
    }
    for(unsigned y=0;y<PH;++y)for(unsigned x=0;x<PW;++x)for(unsigned c=0;c<3;++c){
        correction[(y*PW+x)*4+c]=x<PW/2?.12f:-.12f;
        constant[(y*PW+x)*4+c]=.04f;
    }
    const uint8_t*p=src.data(),*q=reinterpret_cast<const uint8_t*>(correction.data()),*k=reinterpret_cast<const uint8_t*>(constant.data());
    auto*image=gs_texture_create(W,H,GS_RGBA,1,&p,0);
    auto*residual=gs_texture_create(PW,PH,GS_RGBA32F,1,&q,0);
    auto*flat=gs_texture_create(PW,PH,GS_RGBA32F,1,&k,0);
    bool pass=false;
    if(fx&&image&&residual&&flat){
        auto a=render(fx,image,residual,0),b=render(fx,image,residual,1);
        auto neutral=render(fx,image,residual,1,0);
        auto half=render(fx,image,residual,.5f);
        auto fa=render(fx,image,flat,0),fb=render(fx,image,flat,1);
        if(a.size()==src.size()&&b.size()==src.size()&&neutral.size()==src.size()&&half.size()==src.size()&&fa.size()==src.size()&&fb.size()==src.size()){
            double before=0,after=0;size_t n=0;
            bool unchanged=true,constant_ok=true,alpha=true,half_ok=true;
            for(unsigned y=8;y<H-8;++y)for(unsigned x=0;x<W;++x){
                size_t i=(y*W+x)*4;
                alpha&=b[i+3]==src[i+3];
                for(unsigned c=0;c<3;++c){
                    const auto j=i+c;double expected=src[j]+(x<W/2?1:-1)*.12*255;
                    if(x>=W/2-2&&x<W/2+2){before+=std::abs(a[j]-expected);after+=std::abs(b[j]-expected);++n;}
                    else unchanged&=std::abs(int(a[j])-int(b[j]))<=1;
                    constant_ok&=std::abs(int(fa[j])-int(fb[j]))<=1;
                    half_ok&=std::abs(2*int(half[j])-int(a[j])-int(b[j]))<=2;
                }
            }
            const bool zero=neutral==src;
            std::cout<<"edge_mae_linear="<<before/n<<" edge_mae_guided="<<after/n
                <<" interior_unchanged="<<unchanged<<" constant_preserved="<<constant_ok<<" alpha="<<alpha<<" halfway="<<half_ok<<" zero="<<zero<<"\n";
            pass=before/n>3&&after<before*.35&&unchanged&&constant_ok&&alpha&&half_ok&&zero;
        }
    }
    // Identical colors on opposite sides: RGB guidance cannot detect this edge.
    Frame uniform_src(W*H*4,128);
    for(size_t i=3;i<uniform_src.size();i+=4)uniform_src[i]=255;
    std::vector<uint8_t> depths(PW*PH);
    for(unsigned y=0;y<PH;++y)for(unsigned x=0;x<PW;++x)depths[y*PW+x]=x<PW/2?230:25;
    const uint8_t* u=uniform_src.data(),* z=depths.data();
    auto*uniform=gs_texture_create(W,H,GS_RGBA,1,&u,0);
    auto*depth=gs_texture_create(PW,PH,GS_R8,1,&z,0);
    bool depth_ok=false;
    if(fx&&uniform&&depth&&residual&&flat){
        auto linear=render(fx,uniform,residual,1,1,depth,0);
        auto guided=render(fx,uniform,residual,1,1,depth,1);
        auto constant_a=render(fx,uniform,flat,0,1,depth,0);
        auto constant_b=render(fx,uniform,flat,1,1,depth,1);
        if(linear.size()==uniform_src.size()&&guided.size()==linear.size()){
            double before=0,after=0;size_t n=0;
            for(unsigned y=8;y<H-8;++y)for(unsigned x=W/2-2;x<W/2+2;++x)
                for(unsigned c=0;c<3;++c){
                    auto i=(y*W+x)*4+c;double expected=128+(x<W/2?1:-1)*.12*255;
                    before+=std::abs(linear[i]-expected);after+=std::abs(guided[i]-expected);++n;
                }
            depth_ok=before/n>3&&after<before*.35&&constant_a==constant_b;
            std::cout<<"depth_edge_linear="<<before/n<<" depth_edge_guided="<<after/n
                <<" depth_constant="<<(constant_a==constant_b)<<"\n";
        }
    }
    if(depth)gs_texture_destroy(depth);if(uniform)gs_texture_destroy(uniform);
    pass &= depth_ok;
    if(flat)gs_texture_destroy(flat);if(residual)gs_texture_destroy(residual);if(image)gs_texture_destroy(image);
    if(fx)gs_effect_destroy(fx);
    obs_leave_graphics();obs_shutdown();return pass?0:1;
}
