#pragma once
#include "litert_runtime.hpp"
#include "object_detector.hpp"
#include "tasks_tracker.hpp"
#include <string>
#include <vector>
namespace sight {
// Worker-owned, strict GPU RTMW-m 133-point COCO-WholeBody pipeline.
// Person detector supplies the crop. No inferred 3D points, gestures,
// blendshapes, or MediaPipe landmark index substitutions.
enum class RtmwPersonBackend { EfficientDet, YoloX };
class RtmwTracker {
public:
    bool load(const std::string& model,const std::string& detector,
              const std::string& mediapipe_runtime,
              RtmwPersonBackend backend=RtmwPersonBackend::EfficientDet,float person_confidence=.35f);
    bool detect(const uint8_t* rgba,uint32_t w,uint32_t h,int64_t timestamp,
                TrackingData& result);
    bool ready() const {return ready_;}
    const std::string& last_error() const {return error_;}
    size_t compilation_count() const {return runtime_.compilation_count();}
private:
    lll::LiteRtRuntime runtime_;
    ObjectDetector detector_;
    std::vector<float> input_, detector_input_, detector_output_;
    std::string detector_model_;
    RtmwPersonBackend backend_=RtmwPersonBackend::EfficientDet;
    bool detect_people_gpu(const uint8_t* rgba,uint32_t w,uint32_t h,
                           std::vector<ObjectDetection>& detections);
    std::string model_,error_;
    bool ready_=false;
    float person_confidence_=.35f;
};
}
