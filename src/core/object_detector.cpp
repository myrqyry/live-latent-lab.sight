#include "object_detector.hpp"
#include "tasks_abi_123.hpp"
#include "tasks_runtime_abi.hpp"
#include <mediapipe/tasks/c/vision/core/image.h>
#include <mediapipe/tasks/c/vision/core/image_processing_options.h>
#include "tasks_library.hpp"
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <utility>
namespace sight {
namespace {
// Mirrors the exact official MediaPipe Tasks 1.2.3 C object detector ABI.
// Independently compared with upstream v1.2.3 headers in real model tests.
struct Options123 {
    mp123::BaseOptions base_options{};
    MpRunningMode running_mode=MP_RUNNING_MODE_VIDEO;
    const char* display_names_locale=nullptr;
    int max_results=16;
    float score_threshold=.35f;
    const char** category_allowlist=nullptr;
    uint32_t category_allowlist_count=0;
    const char** category_denylist=nullptr;
    uint32_t category_denylist_count=0;
    void (*result_callback)(MpStatus,const void*,MpImagePtr,int64_t)=nullptr;
};
struct Rect123 {int left,top,bottom,right;};
struct Detection123 {
    MpCategory* categories=nullptr;
    uint32_t categories_count=0;
    Rect123 bounding_box{};
    void* keypoints=nullptr;
    uint32_t keypoints_count=0;
};
struct Result123 {
    Detection123* detections=nullptr;
    uint32_t detections_count=0;
};
static_assert(sizeof(Options123)==144);
static_assert(offsetof(Options123,base_options)==0);
static_assert(offsetof(Options123,running_mode)==80);
static_assert(offsetof(Options123,display_names_locale)==88);
static_assert(offsetof(Options123,max_results)==96);
static_assert(offsetof(Options123,score_threshold)==100);
static_assert(offsetof(Options123,category_allowlist)==104);
static_assert(offsetof(Options123,category_denylist)==120);
static_assert(offsetof(Options123,result_callback)==136);
static_assert(sizeof(Detection123)==48 && offsetof(Detection123,bounding_box)==12);
static_assert(sizeof(Result123)==16);
}
struct ObjectDetector::Symbols {
    decltype(&MpImageCreateFromUint8Data) create_image=nullptr;
    decltype(&MpImageFree) free_image=nullptr;
    decltype(&MpErrorFree) free_error=nullptr;
    MpStatus (*create)(Options123*,void**,char**)=nullptr;
    MpStatus (*video)(void*,MpImagePtr,const MpImageProcessingOptions*,int64_t,
                      Result123*,char**)=nullptr;
    void (*close_result)(Result123*)=nullptr;
    MpStatus (*close)(void*,char**)=nullptr;
};
ObjectDetector::~ObjectDetector(){unload();}
void ObjectDetector::unload(){
    if(handle_&&api_&&api_->close){
        char* error=nullptr;
        api_->close(handle_,&error);
        if(error&&api_->free_error)api_->free_error(error);
    }
    handle_=nullptr;
    delete api_;api_=nullptr;
    // XNNPACK native TLS can outlive the classifier on worker teardown.
    // RTLD_NODELETE keeps code mapped until OBS exits.
    if(library_)close_tasks_library(library_);
    library_=nullptr;
    last_timestamp_=-1;
}
bool ObjectDetector::load(const std::string& library_path,
                          const std::string& model_path,float score_threshold,int max_results){
    unload();last_error_.clear();
    if(!tasks_runtime_host_supported){last_error_="No ABI-qualified native MediaPipe runtime for this host";return false;}
    std::error_code ec;
    if(library_path.empty() ||
       !std::filesystem::is_regular_file(library_path,ec)||ec ||
       tasks_runtime_for_host(std::filesystem::file_size(library_path,ec))!=TasksRuntimeAbi::V123||ec){
        last_error_="Object detector requires Sight's verified native MediaPipe 1.2.3 runtime";
        return false;
    }
    ec.clear();
    if(model_path.empty()||!std::filesystem::is_regular_file(model_path,ec)||
       ec||std::filesystem::file_size(model_path,ec)==0||ec){
        last_error_="Missing or empty compatible object detection model";
        return false;
    }
    library_=open_tasks_library(library_path,last_error_);
    if(!library_){return false;}
    api_=new Symbols{};
#define MP_GET(field,symbol) do {api_->field=reinterpret_cast<decltype(api_->field)>(tasks_library_symbol(library_,#symbol));if(!api_->field){last_error_="Missing symbol "#symbol;unload();return false;}}while(0)
    MP_GET(create_image,MpImageCreateFromUint8Data);
    MP_GET(free_image,MpImageFree);
    MP_GET(free_error,MpErrorFree);
    MP_GET(create,MpObjectDetectorCreate);
    MP_GET(video,MpObjectDetectorDetectForVideo);
    MP_GET(close_result,MpObjectDetectorCloseResult);
    MP_GET(close,MpObjectDetectorClose);
#undef MP_GET
    Options123 options{};
    options.score_threshold=std::clamp(score_threshold,.25f,.95f);
    options.max_results=std::clamp(max_results,1,16);
    options.base_options.model_asset_path=model_path.c_str();
    options.base_options.file_descriptor=-1;
    options.base_options.delegate=mp123::kCpuDelegate;
    char* error=nullptr;
    const auto status=api_->create(&options,&handle_,&error);
    if(status!=kMpOk || !handle_){
        last_error_=error?error:"Native object detector initialization failed";
        if(error)api_->free_error(error);
        unload();return false;
    }
    if(error)api_->free_error(error);
    return true;
}
bool ObjectDetector::detect(const uint8_t* rgba,uint32_t width,uint32_t height,
                            int64_t timestamp_ms,std::vector<ObjectDetection>& output){
    output.clear();last_error_.clear();
    if(!handle_||!rgba||width<8||height<8||width>4096||height>4096||
       uint64_t(width)*height*4>INT_MAX||timestamp_ms<=last_timestamp_){
        last_error_="Invalid object detection frame, time or state";
        return false;
    }
    MpImagePtr image=nullptr;char* error=nullptr;
    auto status=api_->create_image(kMpImageFormatSrgba,int(width),int(height),rgba,
                          int(uint64_t(width)*height*4),&image,&error);
    if(status!=kMpOk||!image){
        last_error_=error?error:"Could not create detector image";
        if(error)api_->free_error(error);
        return false;
    }
    if(error){api_->free_error(error);error=nullptr;}
    Result123 result{};
    last_timestamp_=timestamp_ms;
    status=api_->video(handle_,image,nullptr,timestamp_ms,&result,&error);
    api_->free_image(image);
    if(status!=kMpOk){
        last_error_=error?error:"MediaPipe object VIDEO inference failed";
        if(error)api_->free_error(error);
        return false;
    }
    if(error)api_->free_error(error);
    if(result.detections && result.detections_count<=256){
        for(uint32_t i=0;i<result.detections_count;++i){
            const auto& d=result.detections[i];
            if(!d.categories||d.categories_count<1||d.categories_count>32)continue;
            const auto& c=d.categories[0];
            const char* label=c.category_name?c.category_name:c.display_name;
            if(!label)continue;
            const size_t size=strnlen(label,97);
            if(size==0||size>96)continue;
            const float x=std::clamp(float(d.bounding_box.left)/width,0.f,1.f);
            const float y=std::clamp(float(d.bounding_box.top)/height,0.f,1.f);
            const float right=std::clamp(float(d.bounding_box.right)/width,0.f,1.f);
            const float bottom=std::clamp(float(d.bounding_box.bottom)/height,0.f,1.f);
            ObjectDetection item{x,y,right-x,bottom-y,c.score,std::string(label,size),c.index};
            if(valid_object(item))output.push_back(std::move(item));
        }
    }
    api_->close_result(&result); // closes all MediaPipe-owned labels/boxes
    return true;
}
}
