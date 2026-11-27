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
static bool wait_status(obs_source_t*source,obs_properties_t*props,const char*expected){
    auto*button=obs_properties_get(props,"sight_backend_refresh");
    auto*status=obs_properties_get(props,"sight_backend_status");
    if(!button||!status)return false;
    for(int i=0;i<400;++i){
        obs_property_button_clicked(button,source);
        const char*desc=obs_property_description(status);
        if(desc&&std::string(desc).find(expected)!=std::string::npos){puts(desc);return true;}
        std::this_thread::sleep_for(25ms);
    }
    fprintf(stderr,"Expected %s; actual: %s\n",expected,obs_property_description(status));
    return false;
}
static void device(obs_source_t*source,int value){
    auto*d=obs_data_create();obs_data_set_int(d,"cutout_device",value);
    obs_source_update(source,d);obs_data_release(d);
}
int main(int argc,char**argv){
    CHECK(argc==5);
    const bool gpu=std::string(argv[4])=="cuda";
    char temp[]="/tmp/sight-cutout-device-XXXXXX";CHECK(mkdtemp(temp));
    struct Cleanup{const char*p;~Cleanup(){std::error_code ec;std::filesystem::remove_all(p,ec);}}cleanup{temp};
    setenv("XDG_CONFIG_HOME",temp,1);
    if(gpu)unsetenv("SIGHT_ONNX_DEVICE");else setenv("SIGHT_ONNX_DEVICE","cpu",1);
    CHECK(obs_startup("en-US",temp,nullptr));
    obs_video_info v{};v.graphics_module="libobs-opengl";v.fps_num=30;v.fps_den=1;
    v.base_width=v.output_width=64;v.base_height=v.output_height=64;v.output_format=VIDEO_FORMAT_RGBA;
    CHECK(obs_reset_video(&v)==OBS_VIDEO_SUCCESS);
    obs_module_t*m=nullptr;CHECK(obs_open_module(&m,argv[1],argv[2])==MODULE_SUCCESS&&obs_init_module(m));
    auto*defaults=obs_get_source_defaults("sight_u2net_filter");
    CHECK(defaults&&obs_data_get_int(defaults,"cutout_device")==0);obs_data_release(defaults);
    auto*d=obs_data_create();obs_data_set_string(d,"model_path",argv[3]);obs_data_set_int(d,"cutout_device",1);
    auto*a=obs_source_create_private("sight_u2net_filter","explicit CPU",d);
    auto*b=obs_source_create_private("sight_u2net_filter","independent CPU",d);obs_data_release(d);
    CHECK(a&&b);
    auto*p=obs_source_properties(a),*q=obs_source_properties(b);CHECK(p&&q);
    auto*list=obs_properties_get(p,"cutout_device");
    CHECK(list&&obs_property_list_item_count(list)==3);
    CHECK(wait_status(a,p,"Loaded — ONNX Runtime CPU"));
    CHECK(wait_status(b,q,"Loaded — ONNX Runtime CPU"));
    device(a,2);
    CHECK(wait_status(a,p,gpu?"Loaded — ONNX Runtime CUDA":"CUDA disabled by SIGHT_ONNX_DEVICE=cpu"));
    CHECK(wait_status(b,q,"Loaded — ONNX Runtime CPU"));
    device(a,0);
    CHECK(wait_status(a,p,gpu?"Loaded — ONNX Runtime CUDA":"Loaded — ONNX Runtime CPU"));
    device(a,1);
    CHECK(wait_status(a,p,"Loaded — ONNX Runtime CPU"));
    obs_properties_destroy(p);obs_properties_destroy(q);obs_source_release(a);obs_source_release(b);
    obs_wait_for_destroy_queue();obs_shutdown();puts("SIGHT_CUTOUT_DEVICE_PASS=1");return 0;
}
