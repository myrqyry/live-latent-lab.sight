#include <obs.h>
#include <obs-module.h>
#include <chrono>
#include <thread>
#include <string>
#include <cstdio>
#include <filesystem>
#include <cstdlib>
using namespace std::chrono_literals;
#define CHECK(x) do{if(!(x)){fprintf(stderr,"failed: %s\n",#x);return 1;}}while(false)
static bool read(obs_source_t*source,obs_properties_t*props,const char*expected){
 auto*button=obs_properties_get(props,"sight_backend_refresh");
 auto*status=obs_properties_get(props,"sight_backend_status");
 if(!button||!status)return false;
 obs_property_button_clicked(button,source);
 const char*desc=obs_property_description(status);
 return desc&&std::string(desc).find(expected)!=std::string::npos;
}
static bool wait(obs_source_t*s,obs_properties_t*p,const char*value){
 for(int i=0;i<200;++i){if(read(s,p,value))return true;std::this_thread::sleep_for(25ms);}
 return false;
}
static void model(obs_source_t*s,const char*path){
 auto*d=obs_data_create();obs_data_set_string(d,"model_path",path);obs_source_update(s,d);obs_data_release(d);
}
int main(int argc,char**argv){
 CHECK(argc==4);
 char temp[]="/tmp/sight-backend-status-XXXXXX";CHECK(mkdtemp(temp));
 struct Cleanup {const char*path;~Cleanup(){std::error_code ec;std::filesystem::remove_all(path,ec);}} cleanup{temp};
 setenv("XDG_CONFIG_HOME",temp,1);
 CHECK(obs_startup("en-US",temp,nullptr));
 obs_video_info v{};v.graphics_module="libobs-opengl";v.fps_num=30;v.fps_den=1;
 v.base_width=v.output_width=64;v.base_height=v.output_height=64;v.output_format=VIDEO_FORMAT_RGBA;
 CHECK(obs_reset_video(&v)==OBS_VIDEO_SUCCESS);
 obs_module_t*m=nullptr;CHECK(obs_open_module(&m,argv[1],argv[2])==MODULE_SUCCESS&&obs_init_module(m));
 auto*d=obs_data_create();obs_data_set_string(d,"model_path","");auto*s=obs_source_create_private("sight_depth_filter","diagnostic test",d);obs_data_release(d);CHECK(s);
 auto*p=obs_source_properties(s);CHECK(p&&read(s,p,"No model selected"));
 model(s,"/nonexistent/sight-diagnostic-model.onnx");
 CHECK(wait(s,p,"Load failed"));
 CHECK(std::string(obs_property_description(obs_properties_get(p,"sight_backend_status"))).find("nonexistent")!=std::string::npos);
 model(s,argv[3]);CHECK(wait(s,p,"Loaded — ONNX Runtime CPU"));
 model(s,"");CHECK(wait(s,p,"No model selected"));
 obs_properties_destroy(p);obs_source_release(s);
 // Verify actual failed runtime load and explicit off mode on unified tracking.
 d=obs_data_create();obs_data_set_string(d,"tracking_mode","face");
 obs_data_set_string(d,"face_model_path",argv[3]);
 obs_data_set_string(d,"face_mediapipe_runtime","/nonexistent/sight-runtime.so");
 s=obs_source_create_private("sight_unified_tracking_filter","tracking diagnostic test",d);obs_data_release(d);CHECK(s);
 p=obs_source_properties(s);CHECK(p&&wait(s,p,"Load failed"));
 CHECK(read(s,p,"MediaPipe library missing or unverified ABI"));
 d=obs_data_create();obs_data_set_string(d,"tracking_mode","none");obs_source_update(s,d);obs_data_release(d);
 CHECK(wait(s,p,"Tracking off"));
 obs_properties_destroy(p);obs_source_release(s);obs_wait_for_destroy_queue();obs_shutdown();
 puts("SIGHT_BACKEND_STATUS_PASS=1");
}
