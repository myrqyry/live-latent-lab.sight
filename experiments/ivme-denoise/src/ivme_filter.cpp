#include <obs-module.h>
#include <graphics/graphics.h>
#include <util/platform.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include "core/onnx_ivme_denoiser.hpp"

namespace {
constexpr auto NAME="Sight: Camera Denoise (Ivme)";
constexpr auto MODEL="model_path";
constexpr auto EDGE="capture_edge";
constexpr auto SIGMA="noise_sigma";
constexpr auto STRENGTH="blend_strength";
constexpr auto ENABLED="effect_enabled";
constexpr auto HOLD="hold_last_output";
constexpr uint64_t MAX_AGE_MS=2500;
uint64_t now_ms(){
 return std::chrono::duration_cast<std::chrono::milliseconds>(
   std::chrono::steady_clock::now().time_since_epoch()).count();
}
uint32_t cap_edge(int64_t value){
 switch(value){
 case 0:case 128:case 192:case 256:case 384:case 512:
 case 640:case 768:case 1024:case 1280:return uint32_t(value);
 default:return 256;
 }
}
std::pair<uint32_t,uint32_t> size_for(uint32_t w,uint32_t h,uint32_t requested){
 if(!w||!h)return {0,0};
 // This CPU ONNX model consumed ~4GB and ~8s at 1280x720 on this PC.
 // The 1024px original per-axis cap was arbitrary; the 1MP budget is not.
 constexpr uint32_t max_dimension=1280;
 constexpr uint64_t max_pixels=1024ULL*1024ULL;
 if(requested==0){
  if(w>max_dimension||h>max_dimension||uint64_t(w)*h>max_pixels||
     (w%4)||(h%4))return {0,0};
  return {w,h};
 }
 const uint32_t large=std::max(w,h);
 const uint32_t effective=std::min(large,requested);
 auto align=[&](uint32_t v){
  uint32_t raw=uint32_t((uint64_t(v)*effective+large/2)/large);
  return std::max(32u,(raw+2u)/4u*4u);
 };
 const auto out_w=align(w),out_h=align(h);
 if(out_w>max_dimension||out_h>max_dimension||
    uint64_t(out_w)*out_h>max_pixels)return {0,0};
 return {out_w,out_h};
}
struct Filter {
 obs_source_t* source=nullptr;
 std::mutex control_mutex;
 std::condition_variable cv;
 std::thread worker;
 bool stopping=false, load_required=false, frame_ready=false;
 std::string wanted_path;
 uint64_t generation=0;
 uint32_t input_w=0,input_h=0;
 std::vector<uint8_t> input_rgba;
 std::atomic<bool> ready{false};
 std::atomic<bool> wants_frame{false};
 std::atomic<bool> enabled{true};
 std::atomic<bool> hold_last_output{false};
 std::atomic<uint32_t> edge{256};
 std::atomic<float> sigma{25.f};
 std::atomic<float> strength{1.f};
 std::atomic<bool> reset_texture{false};
 // Worker produces pixels; render thread consumes them.
 std::mutex result_mutex;
 std::vector<uint8_t> output_rgba;
 uint32_t output_w=0,output_h=0;
 uint64_t output_stamp=0;
 bool new_output=false;
 // GPU resources on the graphics thread only.
 gs_texrender_t* capture=nullptr;
 gs_stagesurf_t* stage=nullptr;
 gs_texrender_t* original=nullptr;
 gs_texture_t* output=nullptr;
 gs_effect_t* effect=nullptr;
 uint32_t stage_w=0,stage_h=0;
 bool announced=false;
};
void worker_main(Filter* f){
 lll::OnnxIvmeDenoiser processor;
 std::string loaded_model;
 uint32_t previous_w=0,previous_h=0;
 for(;;){
  std::string path;
  uint64_t generation=0;
  std::vector<uint8_t> rgba;
  uint32_t w=0,h=0;
  bool load=false;
  {
   std::unique_lock lk(f->control_mutex);
   f->cv.wait(lk,[&]{return f->stopping||f->load_required||f->frame_ready;});
   if(f->stopping)return;
   generation=f->generation;
   if(f->load_required){
    load=true;path=f->wanted_path;
    f->load_required=false;f->frame_ready=false;
   }else{
    rgba=std::move(f->input_rgba);
    w=f->input_w;h=f->input_h;
    f->frame_ready=false;
   }
  }
  if(load){
   previous_w=0;previous_h=0;
   const bool ok=!path.empty()&&processor.load(path);
   loaded_model=ok?path:"";
   {
    std::lock_guard lk(f->control_mutex);
    if(f->generation!=generation)continue;
    f->ready=ok;
    f->wants_frame=ok;
   }
   if(ok)blog(LOG_INFO,"[Sight/Ivme] Loaded native ONNX checkpoint: %s",path.c_str());
   else blog(LOG_WARNING,"[Sight/Ivme] Checkpoint unavailable: %s",
              processor.last_error().c_str());
   continue;
  }
  if(!processor.ready()||rgba.size()!=size_t(w)*h*4||!w||!h){
   if(f->ready)f->wants_frame=true;
   continue;
  }
  // ONNX Runtime NCHWc's shape-specialized kernels may reject shape changes
  // on the same session. Rebuild only after a *real* capture geometry change.
  if(previous_w!=w||previous_h!=h){
   previous_w=w;previous_h=h;
   if(loaded_model.empty()||!processor.load(loaded_model)){
    blog(LOG_WARNING,"[Sight/Ivme] Could not reset ORT session for new capture geometry: %s",
         processor.last_error().c_str());
    f->ready=false;f->wants_frame=false;
    continue;
   }
  }
  std::vector<uint8_t> rgb(size_t(w)*h*3);
  for(size_t i=0;i<size_t(w)*h;++i)
   memcpy(rgb.data()+i*3,rgba.data()+i*4,3);
  std::vector<uint8_t> denoised;
  const float sigma=f->sigma.load();
  const auto start=now_ms();
  const bool ok=processor.denoise(rgb.data(),w,h,sigma,denoised);
  const uint64_t elapsed=now_ms()-start;
  if(!ok){
   blog(LOG_ERROR,"[Sight/Ivme] Native inference failed: %s",processor.last_error().c_str());
  }else if(denoised.size()==size_t(w)*h*3){
   // Keep the source alpha for the compositor; do not generate or fill alpha.
   std::vector<uint8_t> out(size_t(w)*h*4);
   for(size_t i=0;i<size_t(w)*h;++i){
    memcpy(out.data()+i*4,denoised.data()+i*3,3);
    out[i*4+3]=rgba[i*4+3];
   }
   {
    std::lock_guard lk(f->control_mutex);
    if(generation==f->generation && f->ready){
     std::lock_guard outlk(f->result_mutex);
     f->output_rgba=std::move(out);
     f->output_w=w;f->output_h=h;
     f->output_stamp=now_ms();f->new_output=true;
    }
   }
   if(!f->announced){
    blog(LOG_INFO,"[Sight/Ivme] First REAL native ONNX denoised frame %ux%u (%llu ms)",
         w,h,static_cast<unsigned long long>(elapsed));
    f->announced=true;
   }
  }
  if(f->ready)f->wants_frame=true;
 }
}
const char* get_name(void*){return NAME;}
void* create(obs_data_t* settings,obs_source_t* context){
 auto* f=new Filter;
 f->source=context;
 const char* p=obs_data_get_string(settings,MODEL);
 f->wanted_path=p?p:"";
 f->edge=cap_edge(obs_data_get_int(settings,EDGE));
 f->sigma=float(std::clamp(obs_data_get_double(settings,SIGMA),1.,80.));
 f->strength=float(std::clamp(obs_data_get_double(settings,STRENGTH),0.,1.));
 f->enabled=obs_data_get_bool(settings,ENABLED);
 f->hold_last_output=obs_data_get_bool(settings,HOLD);
 f->generation=1;
 f->load_required=!f->wanted_path.empty();
 f->worker=std::thread(worker_main,f);
 if(f->load_required)f->cv.notify_one();
 return f;
}
void destroy(void* data){
 auto* f=static_cast<Filter*>(data);
 if(!f)return;
 {
  std::lock_guard lk(f->control_mutex);f->stopping=true;
 }
 f->cv.notify_one();
 if(f->worker.joinable())f->worker.join();
 obs_enter_graphics();
 if(f->capture)gs_texrender_destroy(f->capture);
 if(f->stage)gs_stagesurface_destroy(f->stage);
 if(f->original)gs_texrender_destroy(f->original);
 if(f->output)gs_texture_destroy(f->output);
 if(f->effect)gs_effect_destroy(f->effect);
 obs_leave_graphics();
 delete f;
}
void update(void* data,obs_data_t* settings){
 auto* f=static_cast<Filter*>(data);
 const char* path=obs_data_get_string(settings,MODEL);
 const std::string requested=path?path:"";
 const bool enable=obs_data_get_bool(settings,ENABLED);
 const uint32_t edge=cap_edge(obs_data_get_int(settings,EDGE));
 const float sigma=float(std::clamp(obs_data_get_double(settings,SIGMA),1.,80.));
 const float strength=float(std::clamp(obs_data_get_double(settings,STRENGTH),0.,1.));
 const bool enabled_changed=f->enabled.exchange(enable)!=enable;
 const bool edge_changed=f->edge.exchange(edge)!=edge;
 const bool sigma_changed=f->sigma.exchange(sigma)!=sigma;
 f->strength=strength;
 f->hold_last_output=obs_data_get_bool(settings,HOLD);
 bool reload=false;
 {
  std::lock_guard lk(f->control_mutex);
  if(requested!=f->wanted_path){
   f->wanted_path=requested;f->load_required=true;reload=true;f->ready=false;
  }
  if(reload||enabled_changed||edge_changed||sigma_changed){
   ++f->generation;f->frame_ready=false;
   f->wants_frame=f->ready.load();
   f->reset_texture=true;
   std::lock_guard outlk(f->result_mutex);
   f->new_output=false;f->output_rgba.clear();f->output_stamp=0;
  }
 }
 if(reload)f->cv.notify_one();
}
void defaults(obs_data_t* d){
 // Stable user-local model path, separate from the existing Sight plugin.
 const char* home=std::getenv("HOME");
 const std::string default_model=home?std::string(home)+
  "/.config/obs-studio/plugins/sight-ivme-denoise/models/ivme-v1.onnx":"";
 obs_data_set_default_string(d,MODEL,default_model.c_str());
 obs_data_set_default_int(d,EDGE,256);
 obs_data_set_default_double(d,SIGMA,25.);
 obs_data_set_default_double(d,STRENGTH,1.);
 obs_data_set_default_bool(d,ENABLED,true);
 obs_data_set_default_bool(d,HOLD,false);
}
obs_properties_t* properties(void*){
 auto* p=obs_properties_create();
 obs_properties_add_bool(p,ENABLED,"Enable denoising");
 obs_properties_add_bool(p,HOLD,"Hold last processed frame (stop-motion; opt-in)");
 obs_properties_add_path(p,MODEL,"Ivme ONNX model",OBS_PATH_FILE,
                         "ONNX model (*.onnx)",nullptr);
 auto* edge=obs_properties_add_list(p,EDGE,"Processing resolution (max edge)",
                                   OBS_COMBO_TYPE_LIST,OBS_COMBO_FORMAT_INT);
 obs_property_list_add_int(edge,"Native source (1MP safety budget; very slow)",0);
 for(auto value:{128,192,256,384,512,640,768,1024,1280})
  obs_property_list_add_int(edge,std::to_string(value).c_str(),value);
 obs_properties_add_float_slider(p,SIGMA,"Noise level (sigma)",1,80,1);
 obs_properties_add_float_slider(p,STRENGTH,"Mix with original",0,1,.01);
 return p;
}
bool draw_target(Filter* f,obs_source_t* target,gs_texrender_t* renderer,
                 uint32_t source_w,uint32_t source_h,uint32_t render_w,uint32_t render_h){
 if(!renderer)return false;
 gs_texrender_reset(renderer);
 if(!gs_texrender_begin(renderer,render_w,render_h))return false;
 gs_projection_push();gs_matrix_push();gs_matrix_identity();
 gs_ortho(0,float(source_w),0,float(source_h),-100,100);
 auto* effect=obs_get_base_effect(OBS_EFFECT_DEFAULT);
 if(effect)while(gs_effect_loop(effect,"Draw"))obs_source_video_render(target);
 gs_matrix_pop();gs_projection_pop();
 gs_texrender_end(renderer);
 return gs_texrender_get_texture(renderer)!=nullptr;
}
void render(void* data,gs_effect_t*){
 auto* f=static_cast<Filter*>(data);
 auto* target=obs_filter_get_target(f->source);
 if(!target || !obs_filter_get_parent(f->source)){
  obs_source_skip_video_filter(f->source);return;
 }
 uint32_t width=obs_source_get_base_width(target);
 uint32_t height=obs_source_get_base_height(target);
 if(!width||!height||!f->enabled||!f->ready||f->strength<=0){
  obs_source_skip_video_filter(f->source);return;
 }
 if(f->reset_texture.exchange(false)&&f->output){
  gs_texture_destroy(f->output);f->output=nullptr;
 }
 if(!f->effect){
  char* path=obs_module_file("sight-ivme-denoise.effect");
  if(path){
   char* error=nullptr;
   f->effect=gs_effect_create_from_file(path,&error);
   if(error){blog(LOG_ERROR,"[Sight/Ivme] Shader: %s",error);bfree(error);}
   bfree(path);
  }
  if(!f->effect){obs_source_skip_video_filter(f->source);return;}
 }
 const auto [w,h]=size_for(width,height,f->edge);
 if(!w||!h){obs_source_skip_video_filter(f->source);return;}
 if(f->wants_frame.exchange(false)){
  bool submitted=false;
  if(!f->capture)f->capture=gs_texrender_create(GS_RGBA,GS_ZS_NONE);
  if(!f->stage||f->stage_w!=w||f->stage_h!=h){
   if(f->stage)gs_stagesurface_destroy(f->stage);
   f->stage=gs_stagesurface_create(w,h,GS_RGBA);
   f->stage_w=w;f->stage_h=h;
   // No old-resolution result can be composited on a resized source.
   if(f->output){gs_texture_destroy(f->output);f->output=nullptr;}
  }
  if(f->capture&&f->stage&&draw_target(f,target,f->capture,width,height,w,h)){
   auto* texture=gs_texrender_get_texture(f->capture);
   gs_stage_texture(f->stage,texture);
   uint8_t* pixels=nullptr;uint32_t pitch=0;
   if(gs_stagesurface_map(f->stage,&pixels,&pitch)){
    if(pitch>=w*4){
     std::vector<uint8_t> frame(size_t(w)*h*4);
     for(size_t y=0;y<h;++y)
      memcpy(frame.data()+y*w*4,pixels+y*pitch,w*4);
     std::lock_guard lk(f->control_mutex);
     if(f->ready&&!f->load_required){
      f->input_rgba=std::move(frame);f->input_w=w;f->input_h=h;
      f->frame_ready=true;submitted=true;f->cv.notify_one();
     }
    }
    gs_stagesurface_unmap(f->stage);
   }
  }
  if(!submitted&&f->ready)f->wants_frame=true;
 }
 {
  std::lock_guard lk(f->result_mutex);
  if(f->new_output){
   if(!f->output||gs_texture_get_width(f->output)!=f->output_w||
      gs_texture_get_height(f->output)!=f->output_h){
    if(f->output)gs_texture_destroy(f->output);
    f->output=gs_texture_create(f->output_w,f->output_h,GS_RGBA,1,nullptr,GS_DYNAMIC);
   }
   if(f->output&&!f->output_rgba.empty())
    gs_texture_set_image(f->output,f->output_rgba.data(),f->output_w*4,false);
   f->new_output=false;
  }
  // Opt-in stop-motion: a slow inference result remains visible until the
  // next succeeds. Invalid model, disabled filter or changed geometry still
  // fail open. The default keeps the previous freshness timeout.
  if((!f->hold_last_output && now_ms()-f->output_stamp>MAX_AGE_MS)||
     w!=f->output_w||h!=f->output_h){
   obs_source_skip_video_filter(f->source);return;
  }
 }
 if(!f->output){obs_source_skip_video_filter(f->source);return;}
 if(!f->original)f->original=gs_texrender_create(GS_RGBA,GS_ZS_NONE);
 if(!draw_target(f,target,f->original,width,height,width,height)){
  obs_source_skip_video_filter(f->source);return;
 }
 auto* source_tex=gs_texrender_get_texture(f->original);
 auto* orig=gs_effect_get_param_by_name(f->effect,"original_frame");
 auto* clean=gs_effect_get_param_by_name(f->effect,"denoised_frame");
 auto* blend=gs_effect_get_param_by_name(f->effect,"mix_amount");
 if(!source_tex||!orig||!clean||!blend){obs_source_skip_video_filter(f->source);return;}
 gs_effect_set_texture(orig,source_tex);
 gs_effect_set_texture(clean,f->output);
 gs_effect_set_float(blend,f->strength);
 while(gs_effect_loop(f->effect,"Draw"))
  gs_draw_sprite(source_tex,0,width,height);
}
}
obs_source_info sight_ivme_denoise_filter_info={
 .id="sight_ivme_denoise_filter",
 .type=OBS_SOURCE_TYPE_FILTER,
 .output_flags=OBS_SOURCE_VIDEO | OBS_SOURCE_CUSTOM_DRAW,
 .get_name=get_name,.create=create,.destroy=destroy,
 .get_defaults=defaults,.get_properties=properties,
 .update=update,.video_render=render
};
