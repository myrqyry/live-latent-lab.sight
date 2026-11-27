#include "holistic_tracker.hpp"
#include "tasks_abi_123.hpp"
#include "tasks_runtime_abi.hpp"
#include "tasks_library.hpp"
#include <climits>
#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <string>

namespace sight {
namespace {
// This is an ABI copy of MpHolisticLandmarkerOptions from the official
// MediaPipe 1.2.3 manylinux_2_28_x86_64 wheel. Verified against the official
// v1.2.3 header, not against legacy Tasks headers.
struct HolisticOptions123 {
    mp123::BaseOptions base_options{};
    MpRunningMode running_mode = MP_RUNNING_MODE_IMAGE;
    float min_face_detection_confidence=.5f;
    float min_face_suppression_threshold=.3f;
    float min_face_presence_confidence=.5f;
    float min_hand_landmarks_confidence=.5f;
    float min_pose_detection_confidence=.5f;
    float min_pose_suppression_threshold=.3f;
    float min_pose_presence_confidence=.5f;
    bool output_face_blendshapes=false;
    bool output_pose_segmentation_masks=false;
    void (*result_callback)(MpStatus,const MpHolisticLandmarkerResult*,
                            MpImagePtr,int64_t)=nullptr;
};
static_assert(sizeof(HolisticOptions123)==128);
static_assert(offsetof(HolisticOptions123,running_mode)==80);
static_assert(offsetof(HolisticOptions123,min_face_detection_confidence)==84);
static_assert(offsetof(HolisticOptions123,min_pose_presence_confidence)==108);
static_assert(offsetof(HolisticOptions123,result_callback)==120);
}

struct HolisticTracker::Symbols {
    decltype(&MpImageCreateFromUint8Data) create_image=nullptr;
    decltype(&MpImageFree) free_image=nullptr;
    decltype(&MpImageGetWidth) image_width=nullptr;
    decltype(&MpImageGetHeight) image_height=nullptr;
    decltype(&MpImageGetFormat) image_format=nullptr;
    decltype(&MpImageGetChannels) image_channels=nullptr;
    decltype(&MpImageGetByteDepth) image_depth=nullptr;
    decltype(&MpImageGetWidthStep) image_stride=nullptr;
    decltype(&MpImageDataFloat32) image_float_data=nullptr;
    decltype(&MpErrorFree) free_error=nullptr;
    MpStatus (*create)(HolisticOptions123*,void**,char**)=nullptr;
    MpStatus (*video)(void*,MpImagePtr,const MpImageProcessingOptions*,int64_t,
                      MpHolisticLandmarkerResult*,char**)=nullptr;
    void (*close_result)(MpHolisticLandmarkerResult*)=nullptr;
    MpStatus (*close)(void*,char**)=nullptr;
};

HolisticTracker::~HolisticTracker(){unload();}

void HolisticTracker::unload() {
    if(handle_ && api_ && api_->close) {
        char* error=nullptr;
        api_->close(handle_,&error);
        if(error && api_->free_error)api_->free_error(error);
    }
    handle_=nullptr;
    delete api_;api_=nullptr;
    // The MediaPipe library has background TLS destructors: unloading its
    // text before OBS shutdown has caused SIGSEGV. RTLD_NODELETE mirrors the
    // existing verified TasksTracker ownership policy.
    if(library_) close_tasks_library(library_);
    library_=nullptr;
    last_timestamp_ms_=-1;
    mask_enabled_=false;
}

bool HolisticTracker::load(const std::string& library_path,
                           const std::string& model_path,
                           bool enable_person_mask,const HolisticOptions& config) {
    unload();
    last_error_.clear();
    if(!tasks_runtime_host_supported){last_error_="No ABI-qualified native MediaPipe runtime for this host";return false;}
    std::error_code ec;
    if(library_path.empty() ||
       !std::filesystem::is_regular_file(library_path,ec) ||
       ec ||
       tasks_runtime_for_host(std::filesystem::file_size(library_path,ec))!=
           TasksRuntimeAbi::V123 || ec) {
        last_error_="Holistic requires official MediaPipe 1.2.3 Linux x86_64 libmediapipe.so";
        return false;
    }
    if(model_path.empty() || !std::filesystem::is_regular_file(model_path,ec) ||
       ec || std::filesystem::file_size(model_path,ec)==0 || ec) {
        last_error_="Holistic model .task missing or empty";
        return false;
    }
    library_=open_tasks_library(library_path,last_error_);
    if(!library_) {return false;}
    api_=new Symbols{};
#define MP_GET(field,symbol) do {api_->field=reinterpret_cast<decltype(api_->field)>(tasks_library_symbol(library_,#symbol)); if(!api_->field){last_error_="Missing "#symbol;unload();return false;}}while(0)
    MP_GET(create_image,MpImageCreateFromUint8Data);
    MP_GET(free_image,MpImageFree);
    if(enable_person_mask) {
        MP_GET(image_width,MpImageGetWidth);
        MP_GET(image_height,MpImageGetHeight);
        MP_GET(image_format,MpImageGetFormat);
        MP_GET(image_channels,MpImageGetChannels);
        MP_GET(image_depth,MpImageGetByteDepth);
        MP_GET(image_stride,MpImageGetWidthStep);
        MP_GET(image_float_data,MpImageDataFloat32);
    }
    MP_GET(free_error,MpErrorFree);
    MP_GET(create,MpHolisticLandmarkerCreate);
    MP_GET(video,MpHolisticLandmarkerDetectForVideo);
    MP_GET(close_result,MpHolisticLandmarkerCloseResult);
    MP_GET(close,MpHolisticLandmarkerClose);
#undef MP_GET
    HolisticOptions123 options{};
    options.base_options.model_asset_path=model_path.c_str();
    options.base_options.file_descriptor=-1;
    options.base_options.delegate=mp123::kCpuDelegate;
    options.running_mode=MP_RUNNING_MODE_VIDEO;
    options.min_face_detection_confidence=std::clamp(config.face_detection,0.f,1.f);
    options.min_face_suppression_threshold=std::clamp(config.face_suppression,0.f,1.f);
    options.min_face_presence_confidence=std::clamp(config.face_presence,0.f,1.f);
    options.min_hand_landmarks_confidence=std::clamp(config.hand_landmarks,0.f,1.f);
    options.min_pose_detection_confidence=std::clamp(config.pose_detection,0.f,1.f);
    options.min_pose_suppression_threshold=std::clamp(config.pose_suppression,0.f,1.f);
    options.min_pose_presence_confidence=std::clamp(config.pose_presence,0.f,1.f);
    options.output_face_blendshapes=config.blendshapes;
    options.output_pose_segmentation_masks=enable_person_mask;
    char* error=nullptr;
    const MpStatus status=api_->create(&options,&handle_,&error);
    if(status!=kMpOk || !handle_) {
        last_error_=error?error:"Holistic graph creation failed";
        if(error)api_->free_error(error);
        unload();
        return false;
    }
    if(error)api_->free_error(error);
    mask_enabled_=enable_person_mask;
    return true;
}

bool HolisticTracker::detect(const uint8_t* rgba,uint32_t width,uint32_t height,
                             int64_t timestamp_ms,HolisticSignals& out,
                             HolisticPersonMask* person_mask) {
    out={}; last_error_.clear();
    if(person_mask)*person_mask={};
    if(!handle_ || !rgba || width==0 || height==0 ||
       width>4096 || height>4096 || timestamp_ms<=last_timestamp_ms_ ||
       uint64_t(width)*height*4 > INT_MAX) {
        last_error_="Invalid Holistic frame, timestamp or tracker state";
        return false;
    }
    MpImagePtr image=nullptr;
    char* error=nullptr;
    MpStatus status=api_->create_image(kMpImageFormatSrgba,
                                      static_cast<int>(width),
                                      static_cast<int>(height),rgba,
                                      static_cast<int>(uint64_t(width)*height*4),
                                      &image,&error);
    if(status!=kMpOk || !image) {
        last_error_=error?error:"Cannot allocate Holistic image";
        if(error)api_->free_error(error);
        return false;
    }
    if(error)api_->free_error(error);
    last_timestamp_ms_=timestamp_ms;
    MpHolisticLandmarkerResult result{};
    error=nullptr;
    status=api_->video(handle_,image,nullptr,timestamp_ms,&result,&error);
    if(status==kMpOk) {
        out=holistic_to_tracking(result);
        if(mask_enabled_ && person_mask && result.pose_segmentation_mask) {
            // Read only while result still owns MpImagePtr; never store
            // pointers into the signal bus or worker across close_result.
            const auto mask=result.pose_segmentation_mask;
            const int mw=api_->image_width(mask),mh=api_->image_height(mask);
            const int stride=api_->image_stride(mask);
            if(api_->image_format(mask)==kMpImageFormatVec32F1 &&
               api_->image_channels(mask)==1 && api_->image_depth(mask)==4 &&
               mw>=8 && mh>=8 && mw<=4096 && mh<=4096 &&
               stride>=mw*int(sizeof(float)) && stride%int(sizeof(float))==0 &&
               stride<=4096*int(sizeof(float))) {
                const float* pixels=nullptr;
                char* mask_error=nullptr;
                if(api_->image_float_data(mask,&pixels,&mask_error)==kMpOk && pixels) {
                    HolisticPersonMask owned;
                    owned.width=uint32_t(mw);owned.height=uint32_t(mh);
                    owned.values.reserve(size_t(mw)*mh);
                    const size_t float_stride=size_t(stride)/sizeof(float);
                    bool valid=true;
                    for(int y=0;y<mh && valid;++y)
                        for(int x=0;x<mw;++x) {
                            const float v=pixels[size_t(y)*float_stride+x];
                            if(!std::isfinite(v)||v<0.f||v>1.f) {
                                valid=false;break;
                            }
                            owned.values.push_back(v);
                        }
                    if(valid)*person_mask=std::move(owned);
                }
                if(mask_error)api_->free_error(mask_error);
            }
        }
    } else last_error_=error?error:"Holistic VIDEO inference failed";
    if(status==kMpOk)api_->close_result(&result);
    if(error)api_->free_error(error);
    api_->free_image(image);
    return status==kMpOk;
}
}
