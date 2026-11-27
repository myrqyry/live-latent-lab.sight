#include "tasks_tracker.hpp"
#include "tasks_abi_123.hpp"
#include "tasks_runtime_abi.hpp"
#include "tasks_world_landmarks.hpp"
#include "hand_signals.hpp"
#include "tasks_library.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <sstream>
#include <cstring>
#include "mediapipe/tasks/c/vision/face_landmarker/face_landmarker.h"
#include "mediapipe/tasks/c/vision/pose_landmarker/pose_landmarker.h"
#include "mediapipe/tasks/c/vision/gesture_recognizer/gesture_recognizer.h"

namespace sight {
namespace {
// Executing unknown ABI revisions of this C API risks corrupting memory.
// The single version we actually verified uses the older MpBaseOptions layout.
constexpr auto kVerifiedLibrarySize = uintmax_t{33886744};
constexpr auto kOfficial123LibrarySize = uintmax_t{127908512};
constexpr const char* kOfficial123LibrarySha =
    "7ce8b23159d3fd4e8b727e35afbfb35f3d45d6bde1ec11cc01c316b3bf469f63";
static_assert(tasks_runtime_abi(kVerifiedLibrarySize) == TasksRuntimeAbi::Legacy);
static_assert(tasks_runtime_abi(kOfficial123LibrarySize) == TasksRuntimeAbi::V123);
constexpr const char* kVerifiedLibrarySha = "7ba887d5e09d87440b3713816ec02897d6a849eadf2325c9aabe3707cbdac86b";
// Hash-checking a 33MB library at filter creation would block OBS. Runtime
// therefore gates on canonical size only; the full SHA256 is verified at
// package time by cmake/validated_effects.cmake and is echoed in the load
// error below so a mismatched build is identifiable from the OBS log alone.
// MediaPipe 1.2.3 has a different options layout (file_descriptor, delegate
// values, app info, NPU path). Both ABIs have independent real-model tests.
// Any other binary must be investigated before updating this contract.
}
struct TasksTracker::Symbols {
    decltype(&MpImageCreateFromUint8Data) create_image = nullptr;
    decltype(&MpImageFree) free_image = nullptr;
    decltype(&MpErrorFree) free_error = nullptr;
    decltype(&MpFaceLandmarkerCreate) face_create = nullptr;
    decltype(&MpFaceLandmarkerDetectForVideo) face_video = nullptr;
    decltype(&MpFaceLandmarkerCloseResult) face_result_free = nullptr;
    decltype(&MpFaceLandmarkerClose) face_close = nullptr;
    decltype(&MpPoseLandmarkerCreate) pose_create = nullptr;
    decltype(&MpPoseLandmarkerDetectForVideo) pose_video = nullptr;
    decltype(&MpPoseLandmarkerCloseResult) pose_result_free = nullptr;
    decltype(&MpPoseLandmarkerClose) pose_close = nullptr;
    decltype(&MpGestureRecognizerCreate) gesture_create = nullptr;
    decltype(&MpGestureRecognizerRecognizeForVideo) gesture_video = nullptr;
    decltype(&MpGestureRecognizerCloseResult) gesture_result_free = nullptr;
    decltype(&MpGestureRecognizerClose) gesture_close = nullptr;
};
TasksTracker::~TasksTracker() { unload(); }
void TasksTracker::unload()
{
    if (handle_ && api_) {
        char* err=nullptr;
        if (kind_ == TrackingKind::Face && api_->face_close)
            api_->face_close(reinterpret_cast<MpFaceLandmarkerPtr>(handle_), &err);
        else if (kind_ == TrackingKind::Body && api_->pose_close)
            api_->pose_close(reinterpret_cast<MpPoseLandmarkerPtr>(handle_), &err);
        else if (kind_ == TrackingKind::Hand && api_->gesture_close)
            api_->gesture_close(reinterpret_cast<MpGestureRecognizerPtr>(handle_), &err);
        if (err && api_->free_error) api_->free_error(err);
    }
    handle_=nullptr;
    delete api_; api_=nullptr;
    if (library_) close_tasks_library(library_);
    library_=nullptr;
    last_timestamp_ms_=-1;
}

bool TasksTracker::load(const std::string& library_path,
                        const std::string& model_path, TrackingKind kind,const TasksOptions& config)
{
    unload();
    last_error_.clear();
    kind_=kind;
    if(!tasks_runtime_host_supported){last_error_="No ABI-qualified native MediaPipe runtime for this host";return false;}
    std::error_code ec;
    const auto bytes = (!library_path.empty() &&
                        std::filesystem::is_regular_file(library_path,ec))
                           ? std::filesystem::file_size(library_path,ec)
                           : uintmax_t{0};
    const auto abi = ec ? TasksRuntimeAbi::Unknown : tasks_runtime_for_host(bytes);
    if (abi == TasksRuntimeAbi::Unknown) {
        last_error_ = std::string(
            "MediaPipe library missing or unverified ABI; supported: legacy "
            "(size 33886744, SHA256 ") + kVerifiedLibrarySha +
            ") or official 1.2.3 Linux x86_64 wheel (size " +
            std::to_string(kOfficial123LibrarySize) + ", SHA256 " +
            kOfficial123LibrarySha + ")";
        return false;
    }
    if (model_path.empty() || !std::filesystem::is_regular_file(model_path,ec)) {
        last_error_ = "Model .task not found";
        return false;
    }
    // The native MediaPipe runtime creates long-lived TensorFlow/EGL worker
    // threads. Their pthread TLS destructors still reference the library after
    // task graph close. Allowing dlclose to unmap it causes a reproducible
    // SIGSEGV during obs_shutdown (inside __nptl_deallocate_tsd). Keeping
    // this specific verified library mapped until process exit avoids the
    // use-after-dlclose while preserving per-filter task graph cleanup.
    library_=open_tasks_library(library_path,last_error_);
    if (!library_) {return false;}
    auto* api=new Symbols;
#define MP_GET(field,symbol) do { api->field = reinterpret_cast<decltype(api->field)>(tasks_library_symbol(library_,#symbol)); if(!api->field) {last_error_="Missing " #symbol; delete api; unload(); return false;} } while(0)
    MP_GET(create_image,MpImageCreateFromUint8Data);
    MP_GET(free_image,MpImageFree);
    MP_GET(free_error,MpErrorFree);
    MP_GET(face_create,MpFaceLandmarkerCreate);
    MP_GET(face_video,MpFaceLandmarkerDetectForVideo);
    MP_GET(face_result_free,MpFaceLandmarkerCloseResult);
    MP_GET(face_close,MpFaceLandmarkerClose);
    MP_GET(pose_create,MpPoseLandmarkerCreate);
    MP_GET(pose_video,MpPoseLandmarkerDetectForVideo);
    MP_GET(pose_result_free,MpPoseLandmarkerCloseResult);
    MP_GET(pose_close,MpPoseLandmarkerClose);
    MP_GET(gesture_create,MpGestureRecognizerCreate);
    MP_GET(gesture_video,MpGestureRecognizerRecognizeForVideo);
    MP_GET(gesture_result_free,MpGestureRecognizerCloseResult);
    MP_GET(gesture_close,MpGestureRecognizerClose);
#undef MP_GET
    api_=api;
    char* error=nullptr;
    MpStatus code{};
    // MediaPipe 1.2.3 changes both MpBaseOptions layout and the numerical
    // MP_DELEGATE_CPU value. Both C ABI layouts have been tested with real
    // tasks. Never reuse the legacy options pointer with the new runtime.
    const auto configure_base = [&](auto& options) {
        options.base_options.model_asset_path = model_path.c_str();
        if constexpr (requires { options.base_options.file_descriptor; }) {
            options.base_options.file_descriptor = -1;
            options.base_options.delegate = mp123::kCpuDelegate;
        } else {
            options.base_options.delegate = MP_DELEGATE_CPU;
        }
    };
    if(kind_==TrackingKind::Face) {
        auto configure = [&](auto& options) {
            configure_base(options);
            options.running_mode=MP_RUNNING_MODE_VIDEO;
            options.num_faces=1;
            options.min_face_detection_confidence=std::clamp(config.detection,0.f,1.f);
            options.min_face_presence_confidence=std::clamp(config.presence,0.f,1.f);
            options.min_tracking_confidence=std::clamp(config.tracking,0.f,1.f);
            options.output_face_blendshapes=config.blendshapes;
            options.output_facial_transformation_matrixes=config.head_transform;
        };
        MpFaceLandmarkerPtr handle=nullptr;
        if (abi == TasksRuntimeAbi::V123) {
            mp123::FaceOptions options{};
            configure(options);
            using Create = MpStatus (*)(mp123::FaceOptions*,
                                         MpFaceLandmarkerPtr*,char**);
            code=reinterpret_cast<Create>(api_->face_create)(&options,&handle,&error);
        } else {
            MpFaceLandmarkerOptions options{};
            configure(options);
            code=api_->face_create(&options,&handle,&error);
        }
        handle_=handle;
    } else if(kind_==TrackingKind::Body) {
        auto configure = [&](auto& options) {
            configure_base(options);
            options.running_mode=MP_RUNNING_MODE_VIDEO;
            options.num_poses=1;
            options.min_pose_detection_confidence=std::clamp(config.detection,0.f,1.f);
            options.min_pose_presence_confidence=std::clamp(config.presence,0.f,1.f);
            options.min_tracking_confidence=std::clamp(config.tracking,0.f,1.f);
            options.output_segmentation_masks=false;
        };
        MpPoseLandmarkerPtr handle=nullptr;
        if (abi == TasksRuntimeAbi::V123) {
            mp123::PoseOptions options{};
            configure(options);
            using Create = MpStatus (*)(mp123::PoseOptions*,
                                         MpPoseLandmarkerPtr*,char**);
            code=reinterpret_cast<Create>(api_->pose_create)(&options,&handle,&error);
        } else {
            MpPoseLandmarkerOptions options{};
            configure(options);
            code=api_->pose_create(&options,&handle,&error);
        }
        handle_=handle;
    } else {
        auto configure = [&](auto& options) {
            configure_base(options);
            options.running_mode=MP_RUNNING_MODE_VIDEO;
            options.num_hands=std::clamp(config.max_hands,1,2);
            options.min_hand_detection_confidence=std::clamp(config.detection,0.f,1.f);
            options.min_hand_presence_confidence=std::clamp(config.presence,0.f,1.f);
            options.min_tracking_confidence=std::clamp(config.tracking,0.f,1.f);
        };
        MpGestureRecognizerPtr handle=nullptr;
        if (abi == TasksRuntimeAbi::V123) {
            mp123::GestureOptions options{};
            configure(options);
            using Create = MpStatus (*)(const mp123::GestureOptions*,
                                         MpGestureRecognizerPtr*,char**);
            code=reinterpret_cast<Create>(api_->gesture_create)(&options,&handle,&error);
        } else {
            MpGestureRecognizerOptions options{};
            configure(options);
            code=api_->gesture_create(&options,&handle,&error);
        }
        handle_=handle;
    }
    if (code!=0 || !handle_) {
        last_error_ = error ? error : "MediaPipe graph creation failed";
        if(error)api_->free_error(error);
        unload();
        return false;
    }
    if(error)api_->free_error(error);
    return true;
}

bool TasksTracker::detect(const uint8_t* rgba,uint32_t w,uint32_t h,
                          int64_t timestamp,TrackingData& output)
{
    output={};last_error_.clear();
    if(!handle_||!rgba||w==0||h==0 ||
       w>4096||h>4096||timestamp<=last_timestamp_ms_) {
        last_error_="Invalid task frame, timestamp or tracker state";
        return false;
    }
    if(uint64_t(w)*h*4 > INT32_MAX) {last_error_="Frame too large";return false;}
    MpImagePtr image=nullptr;
    char* error=nullptr;
    const auto status=api_->create_image(kMpImageFormatSrgba,static_cast<int>(w),
        static_cast<int>(h),rgba,static_cast<int>(uint64_t(w)*h*4),&image,&error);
    if(status!=0||!image) {
        last_error_=error?error:"Could not create MediaPipe image";
        if(error) api_->free_error(error);
        return false;
    }
    if(error)api_->free_error(error);
    // Update only after a successful image allocation to preserve monotonicity.
    last_timestamp_ms_=timestamp;
    bool ok=false;
    error=nullptr;
    if(kind_==TrackingKind::Face) {
        MpFaceLandmarkerResult result{};
        const auto code=api_->face_video(
            reinterpret_cast<MpFaceLandmarkerPtr>(handle_),image,nullptr,timestamp,
            &result,&error);
        ok=(code==0);
        if(ok && result.face_landmarks_count>0 && result.face_landmarks &&
           result.face_landmarks[0].landmarks_count==478 &&
           result.face_landmarks[0].landmarks) {
            const auto& landmarks=result.face_landmarks[0];
            output.landmarks.reserve(landmarks.landmarks_count);
            for(unsigned i=0;i<landmarks.landmarks_count;++i) {
                const auto& p=landmarks.landmarks[i];
                if(!std::isfinite(p.x)||!std::isfinite(p.y)||!std::isfinite(p.z)){
                    ok=false;break;
                }
                output.landmarks.push_back({p.x,p.y,p.z,
                    p.has_visibility?std::clamp(p.visibility,0.f,1.f):1.f});
            }
            if(ok && result.face_blendshapes_count && result.face_blendshapes) {
                const auto& blend=result.face_blendshapes[0];
                for(unsigned i=0;i<blend.categories_count && i<64;++i) {
                    output.blendshapes.push_back(blend.categories[i].score);
                    const char *name=blend.categories[i].category_name;
                    output.blendshape_names.emplace_back(name?name:"");
                }
            }
            if(ok && result.facial_transformation_matrixes_count &&
               result.facial_transformation_matrixes) {
                const auto& matrix=result.facial_transformation_matrixes[0];
                if(matrix.rows==4 && matrix.cols==4 && matrix.data) {
                    std::copy_n(matrix.data,16,output.head_transform.begin());
                    output.has_head_transform=true;
                }
            }
        }
        api_->face_result_free(&result);
    } else if(kind_==TrackingKind::Body) {
        MpPoseLandmarkerResult result{};
        const auto code=api_->pose_video(
            reinterpret_cast<MpPoseLandmarkerPtr>(handle_),image,nullptr,timestamp,
            &result,&error);
        ok=(code==0);
        if(ok && result.pose_landmarks_count>0 && result.pose_landmarks &&
           result.pose_landmarks[0].landmarks_count==33 &&
           result.pose_landmarks[0].landmarks) {
            const auto& landmarks=result.pose_landmarks[0];
            output.landmarks.reserve(landmarks.landmarks_count);
            for(unsigned i=0;i<landmarks.landmarks_count;++i) {
                const auto& p=landmarks.landmarks[i];
                if(!std::isfinite(p.x)||!std::isfinite(p.y)||!std::isfinite(p.z)){
                    ok=false;break;
                }
                output.landmarks.push_back({p.x,p.y,p.z,
                    p.has_visibility?std::clamp(p.visibility,0.f,1.f):1.f});
            }
        }
        // Preserve the same person's metric coordinates only when complete
        // and valid. This is orthogonal to normalized image landmarks.
        if (ok && output.landmarks.size() == 33 &&
            result.pose_world_landmarks &&
            result.pose_world_landmarks_count > 0)
            output.world_landmarks =
                copy_world_landmarks(&result.pose_world_landmarks[0], 33);
        api_->pose_result_free(&result);
    } else {
        MpGestureRecognizerResult result{};
        const auto code=api_->gesture_video(
            reinterpret_cast<MpGestureRecognizerPtr>(handle_),image,nullptr,timestamp,
            &result,&error);
        ok=(code==0);
        if(ok && result.hand_landmarks_count>0 && result.hand_landmarks) {
            for(unsigned hidx=0;hidx<result.hand_landmarks_count && hidx<2;++hidx) {
                const auto& hand=result.hand_landmarks[hidx];
                if(hand.landmarks_count!=21 || !hand.landmarks){ok=false;break;}
                HandData detected;
                detected.landmarks.reserve(21);
                for(unsigned j=0;j<21;++j) {
                    const auto& pt=hand.landmarks[j];
                    if(!std::isfinite(pt.x)||!std::isfinite(pt.y)||!std::isfinite(pt.z)) {
                        ok=false;break;
                    }
                    detected.landmarks.push_back({pt.x,pt.y,pt.z,1.f});
                }
                if(!ok)break;
                // Result arrays share hand indices with the gesture and
                // handedness categories. Keep each hand's world points
                // attached to the same hand, not image-aligned body space.
                if (hidx < result.hand_world_landmarks_count &&
                    result.hand_world_landmarks)
                    detected.world_landmarks = copy_world_landmarks(
                        &result.hand_world_landmarks[hidx], 21);
                detected.pinch=normalized_pinch(detected.landmarks);
                if(hidx<result.gestures_count && result.gestures) {
                    const auto& categories=result.gestures[hidx];
                    if(categories.categories_count && categories.categories) {
                        const auto& best=categories.categories[0];
                        detected.gesture=best.category_name?best.category_name:"";
                        detected.gesture_score=std::isfinite(best.score)?
                            std::clamp(best.score,0.f,1.f):0.f;
                    }
                }
                if(hidx<result.handedness_count && result.handedness) {
                    const auto& categories=result.handedness[hidx];
                    if(categories.categories_count && categories.categories)
                        detected.handedness=categories.categories[0].category_name?
                            categories.categories[0].category_name:"";
                }
                output.hands.push_back(std::move(detected));
            }
            if(ok && !output.hands.empty())
                output.landmarks=output.hands[0].landmarks;
        }
        api_->gesture_result_free(&result);
    }
    if(error) {
        if(!ok)last_error_=error;
        api_->free_error(error);
    }
    api_->free_image(image);
    if(!ok) {
        output={};
        if(last_error_.empty())last_error_="MediaPipe video inference failed";
    }
    return ok;
}
} // namespace sight
