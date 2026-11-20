#include <obs.h>
#include <obs-module.h>
#include <graphics/graphics.h>
#include <util/platform.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <iostream>
#include <thread>
#include <vector>
namespace {
#ifndef SIGHT_TEST_SOURCE_W
#define SIGHT_TEST_SOURCE_W 256
#define SIGHT_TEST_SOURCE_H 144
#define SIGHT_TEST_CAPTURE_EDGE 256
#endif
constexpr uint32_t W=SIGHT_TEST_SOURCE_W,H=SIGHT_TEST_SOURCE_H;
using Frame=std::vector<uint8_t>;
struct Image {gs_texture_t* t=nullptr;};
const char* name(void*){return "noisy camera fixture";}
uint32_t w(void*){return W;}uint32_t h(void*){return H;}
Frame picture(){
 Frame img(size_t(W)*H*4);
 uint32_t rng=0x7af635ac;
 for(uint32_t y=0;y<H;++y)for(uint32_t x=0;x<W;++x){
  size_t i=(size_t(y)*W+x)*4;
  rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;
  int n=int(rng%61)-30;
  int b=(x>85&&x<170?145:65)+(y>80?20:0);
  for(int c=0;c<3;++c)img[i+c]=uint8_t(std::clamp(b+n,0,255));
  img[i+3]=255;
 }
 return img;
}
void* create(obs_data_t*,obs_source_t*){
 auto* out=new Image;
 auto pixels=picture();
 const uint8_t* ptr=pixels.data();
 obs_enter_graphics();out->t=gs_texture_create(W,H,GS_RGBA,1,&ptr,0);obs_leave_graphics();
 return out;
}
void destroy(void* p){
 auto* i=(Image*)p;
 obs_enter_graphics();if(i->t)gs_texture_destroy(i->t);obs_leave_graphics();delete i;
}
void render(void* p,gs_effect_t*){auto* i=(Image*)p;if(i->t)obs_source_draw(i->t,0,0,W,H,false);}
obs_source_info src{
 .id="ivme_smoke_source",.type=OBS_SOURCE_TYPE_INPUT,
 .output_flags=OBS_SOURCE_VIDEO,.get_name=name,
 .create=create,.destroy=destroy,.get_width=w,.get_height=h,
 .video_render=render
};
Frame grab(obs_source_t* source){
 obs_enter_graphics();Frame pixels;
 auto* tr=gs_texrender_create(GS_RGBA,GS_ZS_NONE);
 auto* st=gs_stagesurface_create(W,H,GS_RGBA);
 if(tr&&st&&gs_texrender_begin(tr,W,H)){
  gs_projection_push();gs_matrix_push();gs_matrix_identity();
  gs_ortho(0,float(W),0,float(H),-100,100);
  obs_source_video_render(source);
  gs_matrix_pop();gs_projection_pop();gs_texrender_end(tr);
  gs_stage_texture(st,gs_texrender_get_texture(tr));
  uint8_t* p=nullptr;uint32_t stride=0;
  if(gs_stagesurface_map(st,&p,&stride)){
   pixels.resize(size_t(W)*H*4);
   for(size_t row=0;row<H;++row)memcpy(pixels.data()+row*W*4,p+row*stride,W*4);
   gs_stagesurface_unmap(st);
  }
 }
 if(st)gs_stagesurface_destroy(st);if(tr)gs_texrender_destroy(tr);
 obs_leave_graphics();return pixels;
}
double difference(const Frame& a,const Frame& b){
 if(a.size()!=b.size()||a.empty())return -1;
 double total=0;
 for(size_t i=0;i<a.size();i+=4)for(int c=0;c<3;++c)
  total+=std::abs(int(a[i+c])-int(b[i+c]));
 return total / (a.size()/4*3);
}
}
int main(int argc,char**argv){
 if(argc!=4)return 2;
 if(!obs_startup("en-US",nullptr,nullptr))return 3;
 obs_video_info vi{};vi.graphics_module="libobs-opengl";vi.fps_num=30;vi.fps_den=1;
 vi.base_width=W;vi.base_height=H;vi.output_width=W;vi.output_height=H;
 vi.output_format=VIDEO_FORMAT_RGBA;vi.scale_type=OBS_SCALE_DISABLE;
 if(obs_reset_video(&vi)!=OBS_VIDEO_SUCCESS)return 4;
 obs_module_t* module=nullptr;
 if(obs_open_module(&module,argv[1],argv[2])!=MODULE_SUCCESS||
    !obs_init_module(module)){std::cerr<<"MODULE_LOAD_FAILED\n";return 5;}
 obs_register_source(&src);
 auto* source=obs_source_create("ivme_smoke_source","test source",nullptr,nullptr);
 if(!source)return 6;
 auto input=grab(source);
 // Fail-open when a model is not configured. Missing weights must not black
 // out or change the webcam (verified through the *actual* OBS render path).
 {
  auto* absent=obs_data_create();
  obs_data_set_bool(absent,"effect_enabled",true);
  obs_data_set_string(absent,"model_path","/tmp/nonexistent-ivme-denoiser-model.onnx");
  auto* missing=obs_source_create("sight_ivme_denoise_filter","Ivme missing",absent,nullptr);
  obs_data_release(absent);
  if(!missing)return 9;
  obs_source_filter_add(source,missing);
  for(int n=0;n<3;++n){
   auto raw=grab(source);
   if(difference(raw,input)>0.01){
    std::cerr<<"MISSING_MODEL_NOT_PASSTHROUGH\n";return 10;
   }
  }
  obs_source_filter_remove(source,missing);
  obs_source_release(missing);
 }
 auto* opts=obs_data_create();
 obs_data_set_string(opts,"model_path",argv[3]);
 obs_data_set_int(opts,"capture_edge",SIGHT_TEST_CAPTURE_EDGE);
 obs_data_set_double(opts,"noise_sigma",25.0);
 obs_data_set_double(opts,"blend_strength",1.0);
 // Stop-motion: explicitly retain last successful image across slow inference.
 obs_data_set_bool(opts,"hold_last_output",true);
 auto* filter=obs_source_create("sight_ivme_denoise_filter","Ivme Denoise",opts,nullptr);
 obs_data_release(opts);
 if(!filter){std::cerr<<"FILTER_NOT_REGISTERED\n";return 7;}
 obs_source_filter_add(source,filter);
 bool changed=false;
 for(int j=0;j<180;++j){
  auto output=grab(source);
  double delta=difference(input,output);
  if(j%10==0)std::cout<<"OBS_IVME_ATTEMPT "<<j<<" frame_diff="<<delta<<std::endl;
  if(delta>2 && output.size()==input.size()){changed=true;break;}
  std::this_thread::sleep_for(std::chrono::milliseconds(120));
 }
 if(changed){
  // Changing capture size and sigma must invalidate previously completed
  // output, then produce a newly computed result at the selected size.
  auto* settings=obs_source_get_settings(filter);
  obs_data_set_int(settings,"capture_edge",128);
  obs_data_set_double(settings,"noise_sigma",40.0);
  obs_source_update(filter,settings);
  obs_data_release(settings);
  bool cleared=false;
  for(int i=0;i<14;++i){
   const double d=difference(grab(source),input);
   if(d<0.01){cleared=true;break;}
   std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  if(!cleared){
   std::cerr<<"IVME_STALE_OUTPUT_AFTER_SETTINGS_UPDATE\n";return 12;
  }
  bool updated=false;
  for(int i=0;i<90;++i){
   if(difference(grab(source),input)>2){updated=true;break;}
   std::this_thread::sleep_for(std::chrono::milliseconds(120));
  }
  if(!updated){
   std::cerr<<"IVME_NO_OUTPUT_AFTER_SETTINGS_UPDATE\n";return 13;
  }
  std::cout<<"IVME_SETTINGS_INVALIDATION_PASS\n";
  // No rendering while sleeping: this tests output persistence, not
  // whether a worker can generate another frame before the stale timeout.
  std::this_thread::sleep_for(std::chrono::milliseconds(3200));
  if(difference(grab(source),input)<=2){
   std::cerr<<"IVME_SLOW_FRAME_NOT_HELD\n";return 14;
  }
  std::cout<<"IVME_SLOW_FRAME_HOLD_PASS\n";
  // Critical safety: disabled effect must show current source even in Hold mode.
  settings=obs_source_get_settings(filter);
  obs_data_set_bool(settings,"effect_enabled",false);
  obs_source_update(filter,settings);
  obs_data_release(settings);
  bool disabled=false;
  for(int k=0;k<20;++k){
   if(difference(grab(source),input)<0.01){disabled=true;break;}
   std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  if(!disabled){std::cerr<<"IVME_HOLD_IGNORED_DISABLE\n";return 15;}
  std::cout<<"IVME_HOLD_DISABLE_PASSTHROUGH_PASS\n";
 }
 obs_source_filter_remove(source,filter);
 obs_source_release(filter);obs_source_release(source);
 obs_shutdown();
 std::cout<<"REAL_IVME_OBS_FILTER "<<(changed?"PASS":"FAIL")<<std::endl;
 return changed?0:8;
}
