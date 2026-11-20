#pragma once
#include "holistic_signal_adapter.hpp"
#include "holistic_person_mask.hpp"
#include <cstdint>
#include <string>

namespace sight {

// Dedicated single-graph MediaPipe 1.2.3 Holistic VIDEO runner.
// All operations happen on one owning worker thread. No OBS dependencies.
struct HolisticOptions {
    float face_detection=.5f,face_suppression=.3f,face_presence=.5f;
    float hand_landmarks=.5f;
    float pose_detection=.5f,pose_suppression=.3f,pose_presence=.5f;
    bool blendshapes=true;
    bool operator==(const HolisticOptions&) const = default;
};
class HolisticTracker {
public:
    HolisticTracker() = default;
    ~HolisticTracker();
    HolisticTracker(const HolisticTracker&) = delete;
    HolisticTracker& operator=(const HolisticTracker&) = delete;
    bool load(const std::string& library_path, const std::string& model_path,
              bool enable_person_mask=false,const HolisticOptions& options={});
    bool detect(const uint8_t* rgba,uint32_t width,uint32_t height,
                int64_t timestamp_ms,HolisticSignals& out,
                HolisticPersonMask* person_mask=nullptr);
    void unload();
    bool ready() const {return handle_!=nullptr;}
    const std::string& last_error() const {return last_error_;}
private:
    void* library_=nullptr;
    void* handle_=nullptr;
    struct Symbols;
    Symbols* api_=nullptr;
    int64_t last_timestamp_ms_=-1;
    bool mask_enabled_=false;
    std::string last_error_;
};
}
