#pragma once
#include "object_tracks.hpp"
#include <cstdint>
#include <string>
#include <vector>
namespace sight {
// Dedicated MediaPipe Tasks 1.2.3 VIDEO detector. Owned by a single worker,
// uses the separately installed official runtime and user-selected model.
// No OBS, model weights or native library is linked into Sight.
class ObjectDetector {
public:
    ObjectDetector()=default;
    ~ObjectDetector();
    ObjectDetector(const ObjectDetector&)=delete;
    ObjectDetector& operator=(const ObjectDetector&)=delete;
    bool load(const std::string& library_path,const std::string& model_path,
              float score_threshold=.35f,int max_results=16);
    bool detect(const uint8_t* rgba,uint32_t width,uint32_t height,
                int64_t timestamp_ms,std::vector<ObjectDetection>& output);
    void unload();
    bool ready() const{return handle_!=nullptr;}
    const std::string& last_error() const{return last_error_;}
private:
    struct Symbols;
    Symbols* api_=nullptr;
    void* library_=nullptr;
    void* handle_=nullptr;
    int64_t last_timestamp_=-1;
    std::string last_error_;
};
}
