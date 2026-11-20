#pragma once
#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace sight {
enum class TrackingKind { Face, Body, Hand, WholeBody };
struct TrackedPoint {
    float x = 0, y = 0, z = 0, confidence = 1;
};
struct HandData {
    std::vector<TrackedPoint> landmarks;
    // Hand-centered meters; indices match this hand's image landmarks,
    // not the hip-centered body world coordinates.
    std::vector<TrackedPoint> world_landmarks;
    std::string gesture;
    float gesture_score = 0.f;
    // Raw handedness label from MediaPipe. Coordinates still follow camera.
    std::string handedness;
    float pinch = 0.f; // Thumb tip to index tip, normalized by palm length.
};
struct TrackingData {
    std::vector<TrackedPoint> landmarks;
    // Pose-only: hip-centered meters. Empty when unavailable/invalid.
    // Never substitute normalized image coordinates into this field.
    std::vector<TrackedPoint> world_landmarks;
    // Hand kind: one or two 21-point datasets; landmarks aliases hand[0]
    // for compatibility with existing consumers, not a third inference.
    std::vector<HandData> hands;
    // Face-only: valid when 52 blendshapes or a 4x4 matrix was provided.
    std::vector<float> blendshapes;
    // Preserve MediaPipe category names; numeric order is not a stable VRM rig contract.
    std::vector<std::string> blendshape_names;
    std::array<float,16> head_transform{};
    bool has_head_transform = false;
};

// CPU MediaPipe Tasks video graph. Supports the SHA-pinned legacy C ABI
// and the official MediaPipe 1.2.3 Linux x86_64 wheel ABI. The runtime
// library and .task models are external, not linked dependencies.
struct TasksOptions {
    float detection=.5f,presence=.5f,tracking=.5f;
    int max_hands=2;
    bool blendshapes=true,head_transform=true;
    bool operator==(const TasksOptions&) const = default;
};
class TasksTracker {
public:
    TasksTracker() = default;
    ~TasksTracker();
    TasksTracker(const TasksTracker&) = delete;
    TasksTracker& operator=(const TasksTracker&) = delete;
    bool load(const std::string& library_path, const std::string& model_path,
              TrackingKind kind,const TasksOptions& options={});
    bool detect(const uint8_t* rgba, uint32_t width, uint32_t height,
                int64_t timestamp_ms, TrackingData& out);
    bool ready() const { return handle_ != nullptr; }
    const std::string& last_error() const { return last_error_; }
    void unload();
private:
    void* library_ = nullptr;
    void* handle_ = nullptr;
    TrackingKind kind_ = TrackingKind::Face;
    std::string last_error_;
    struct Symbols;
    Symbols* api_ = nullptr;
    int64_t last_timestamp_ms_ = -1;
};
}
