#include <obs.h>
#include <graphics/graphics.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>
constexpr unsigned W=320,H=180;
using Frame=std::vector<uint8_t>;
Frame render(gs_effect_t*fx,gs_texture_t*input,gs_texture_t*mask,gs_texture_t*text_mask) {
 auto* dst=gs_texrender_create(GS_RGBA,GS_ZS_NONE);
 auto* stage=gs_stagesurface_create(W,H,GS_RGBA);
 Frame pixels;
 gs_set_cull_mode(GS_NEITHER);
 if(dst&&stage&&gs_texrender_begin(dst,W,H)) {
  gs_projection_push();gs_matrix_push();gs_matrix_identity();
  gs_ortho(0,float(W),0,float(H),-100,100);
  gs_effect_set_texture(gs_effect_get_param_by_name(fx,"image"),input);
  gs_effect_set_texture(gs_effect_get_param_by_name(fx,"privacy_mask"),mask);
  gs_effect_set_texture(gs_effect_get_param_by_name(fx,"privacy_text_mask"),text_mask);
  vec2 resolution{};vec2_set(&resolution,float(W),float(H));
  gs_effect_set_vec2(gs_effect_get_param_by_name(fx,"image_size"),&resolution);
  gs_effect_set_float(gs_effect_get_param_by_name(fx,"block_px"),28.f);
  while(gs_effect_loop(fx,"Draw"))gs_draw_sprite(input,0,W,H);
  gs_matrix_pop();gs_projection_pop();gs_texrender_end(dst);
  gs_stage_texture(stage,gs_texrender_get_texture(dst));
  uint8_t*data=nullptr;uint32_t pitch=0;
  if(gs_stagesurface_map(stage,&data,&pitch)) {
   pixels.resize(size_t(W)*H*4);
   for(unsigned y=0;y<H;++y)
    std::memcpy(pixels.data()+size_t(y)*W*4,data+size_t(y)*pitch,W*4);
   gs_stagesurface_unmap(stage);
  }
 }
 if(stage)gs_stagesurface_destroy(stage);
 if(dst)gs_texrender_destroy(dst);
 return pixels;
}
int main(int argc,char**argv) {
 if(argc!=2)return 2;
 if(!obs_startup("en-US",nullptr,nullptr))return 1;
 obs_video_info v{};v.graphics_module="libobs-opengl";
 v.fps_num=30;v.fps_den=1;v.base_width=W;v.base_height=H;
 v.output_width=W;v.output_height=H;v.output_format=VIDEO_FORMAT_RGBA;
 v.scale_type=OBS_SCALE_DISABLE;
 if(obs_reset_video(&v)!=OBS_VIDEO_SUCCESS){obs_shutdown();return 1;}
 obs_enter_graphics();
 char*errors=nullptr;
 auto*fx=gs_effect_create_from_file(argv[1],&errors);
 if(errors){std::cerr<<"Shader compiler: "<<errors<<"\n";bfree(errors);}
 bool pass=false;
 if(fx) {
  Frame original(size_t(W)*H*4);
  std::vector<uint8_t> m(size_t(W)*H,0);
  std::vector<uint8_t> text_m(size_t(W)*H,0);
  for(unsigned y=0;y<H;++y)for(unsigned x=0;x<W;++x) {
   auto i=(size_t(y)*W+x)*4;
   original[i]=uint8_t((x*7+y*3)%255);
   original[i+1]=uint8_t((x*3+y*5)%255);
   original[i+2]=uint8_t((x*5+y*7)%255);
   original[i+3]=255;
   if((x>=35&&x<130&&y>=35&&y<145)||
      (x>=190&&x<290&&y>=35&&y<145))m[size_t(y)*W+x]=255;
   if(x>=140&&x<178&&y>=45&&y<135)text_m[size_t(y)*W+x]=255;
  }
  const uint8_t *source_data=original.data();
  auto*input=gs_texture_create(W,H,GS_RGBA,1,&source_data,0);
  const uint8_t *mask_data=m.data();
  auto*mask=gs_texture_create(W,H,GS_R8,1,&mask_data,0);
  const uint8_t*text_pixels=text_m.data();
  auto*text_mask=gs_texture_create(W,H,GS_R8,1,&text_pixels,0);
  if(input&&mask&&text_mask) {
   auto concealed=render(fx,input,mask,text_mask);
   std::vector<uint8_t> empty_mask(size_t(W)*H,0);
   const uint8_t*empty_pixels=empty_mask.data();
   auto passthrough_mask=gs_texture_create(W,H,GS_R8,1,&empty_pixels,0);
   std::vector<uint8_t> zero_text(size_t(W)*H,0);
   const uint8_t*empty_text=zero_text.data();
   auto no_text=gs_texture_create(W,H,GS_R8,1,&empty_text,0);
   auto passthrough=(passthrough_mask&&no_text)?render(fx,input,passthrough_mask,no_text):Frame{};
   if(no_text)gs_texture_destroy(no_text);
   if(passthrough_mask)gs_texture_destroy(passthrough_mask);
   size_t changed_faces=0,changed_background=0,solid_text=0;
   bool unchanged=passthrough.size()==original.size(),opaque=!concealed.empty();
   if(concealed.size()==original.size()&&passthrough.size()==original.size()) {
    for(size_t i=0;i<m.size();++i) {
     bool changed=false;
     for(int j=0;j<3;++j) {
      changed|=std::abs(int(concealed[4*i+j])-int(original[4*i+j]))>2;
      unchanged &= std::abs(int(passthrough[4*i+j])-int(original[4*i+j]))<=2;
     }
     unchanged &= passthrough[4*i+3]==original[4*i+3];
     opaque &= concealed[4*i+3]==255;
     if(text_m[i]) {
       solid_text+=concealed[4*i]==0&&concealed[4*i+1]==0&&concealed[4*i+2]==0;
     } else if(m[i])changed_faces+=changed;else changed_background+=changed;
    }
   }
   std::cout<<"PRIVACY_GPU_OPAQUE_TEXT="<<solid_text<<" PRIVACY_GPU_CONCEALED_FACES="<<changed_faces
      <<" BACKGROUND_CHANGED="<<changed_background
      <<" ZERO_MASK_PASSTHROUGH="<<unchanged<<" OPAQUE="<<opaque<<"\n";
   pass=changed_faces>12000 && changed_background==0 &&
        solid_text>=38*90-10 && unchanged && opaque;
  }
  if(text_mask)gs_texture_destroy(text_mask);
  if(mask)gs_texture_destroy(mask);
  if(input)gs_texture_destroy(input);
  gs_effect_destroy(fx);
 }
 obs_leave_graphics();obs_shutdown();
 std::cout<<(pass?"PASS: two masked regions and ordinary unmasked video":
                  "FAIL: privacy GPU behavior unverified")<<"\n";
 return pass?0:1;
}
