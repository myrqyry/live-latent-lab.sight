#include <obs.h>
#include <obs-module.h>
#include <cstdio>
#include <cstring>
#define CHECK(x) do {if(!(x)){fprintf(stderr,"FAIL: %s\n",#x);return 1;}}while(false)
int main(int argc,char**argv){
 CHECK(argc==4&&obs_startup("en-US",argv[3],nullptr));
 obs_video_info v{};v.graphics_module="libobs-opengl";v.fps_num=30;v.fps_den=1;
 v.base_width=v.output_width=320;v.base_height=v.output_height=192;v.output_format=VIDEO_FORMAT_RGBA;
 v.colorspace=VIDEO_CS_DEFAULT;v.range=VIDEO_RANGE_DEFAULT;v.scale_type=OBS_SCALE_DISABLE;
 CHECK(obs_reset_video(&v)==OBS_VIDEO_SUCCESS);
 obs_module_t*m=nullptr;CHECK(obs_open_module(&m,argv[1],argv[2])==MODULE_SUCCESS&&obs_init_module(m));
 const char* ids[]={"sight_face_motion_filter","sight_body_motion_filter","sight_hand_motion_filter",
 "sight_holistic_motion_filter","sight_object_tracking_filter","sight_touch_field_filter"};
 auto*scene=obs_scene_create("Legacy scene fixture");CHECK(scene);
 auto*parent=obs_scene_get_source(scene);
 for(const auto*id:ids){
  CHECK(obs_source_get_display_name(id));
  CHECK(obs_get_source_output_flags(id)&OBS_SOURCE_CAP_OBSOLETE);
  auto*d=obs_data_create();
  obs_data_set_string(d,"model_path","");
  obs_data_set_string(d,"mediapipe_runtime","");
  obs_data_set_string(d,"signal_channel","legacy/hand");
  obs_data_set_string(d,"signal_channel_face","legacy/face");
  obs_data_set_string(d,"signal_channel_body","legacy/body");
  obs_data_set_string(d,"signal_channel_hand","legacy/holistic-hand");
  if(!strcmp(id,"sight_face_motion_filter"))obs_data_set_string(d,"signal_channel","legacy/face-single");
  if(!strcmp(id,"sight_body_motion_filter"))obs_data_set_string(d,"signal_channel","legacy/body-single");
  obs_data_set_int(d,"inference_cadence",4);
  obs_data_set_double(d,"touch_strength",1.4);
  auto*f=obs_source_create_private(id,id,d);obs_data_release(d);CHECK(f);
  CHECK(obs_source_get_type(f)==OBS_SOURCE_TYPE_FILTER);
  obs_source_filter_add(parent,f);obs_source_release(f);
 }
 // Serialize with libobs and reload the actual legacy IDs/settings into a fresh scene.
 auto*saved=obs_save_source(parent);CHECK(saved);
 auto*serialized=obs_data_create_from_json(obs_data_get_json(saved));CHECK(serialized);
 obs_data_release(saved);
 // Remove the original scene and its filters before loading the same UUIDs.
 for(const auto*id:ids){
  auto*f=obs_source_get_filter_by_name(parent,id);CHECK(f);
  obs_source_filter_remove(parent,f);obs_source_release(f);
 }
 obs_source_remove(parent);obs_scene_release(scene);obs_wait_for_destroy_queue();
 obs_data_set_string(serialized,"name","Reloaded legacy scene");
 auto*loaded=obs_load_source(serialized);obs_data_release(serialized);CHECK(loaded);
 for(const auto*id:ids){
  auto*f=obs_source_get_filter_by_name(loaded,id);CHECK(f);
  CHECK(!strcmp(obs_source_get_unversioned_id(f),id));
  auto*d=obs_source_get_settings(f);CHECK(d);
  auto*p=obs_source_properties(f);CHECK(p);
  if(!strcmp(id,"sight_touch_field_filter")){
   CHECK(obs_data_get_double(d,"touch_strength")==1.4);
   auto*ch=obs_properties_get(p,"signal_channel");CHECK(ch);
   bool discovered=false;
   for(size_t i=0;i<obs_property_list_item_count(ch);++i)
    if(!strcmp(obs_property_list_item_string(ch,i),"legacy/hand")&&!obs_property_list_item_disabled(ch,i))discovered=true;
   if(!discovered)for(size_t i=0;i<obs_property_list_item_count(ch);++i)
    fprintf(stderr,"channel %s disabled=%d\n",obs_property_list_item_string(ch,i),obs_property_list_item_disabled(ch,i));
   CHECK(discovered);
  }else{
   CHECK(obs_data_get_int(d,"inference_cadence")==4);
   CHECK(obs_properties_get(p,"model_path")&&!obs_properties_get(p,"tracking_mode"));
   CHECK(!*obs_data_get_string(d,"model_path"));
   obs_data_set_int(d,"inference_cadence",8);obs_source_update(f,d);
  }
  obs_properties_destroy(p);obs_data_release(d);obs_source_release(f);
 }
 obs_source_release(loaded);obs_wait_for_destroy_queue();obs_shutdown();
 puts("SIGHT_LEGACY_SCENE_PASS=1");
}
