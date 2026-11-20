#pragma once

#include <string>
#include <vector>
#include <mutex>
#include <atomic>
#include <memory>
#include <map>
#include <functional>
#include "stable-diffusion.h"
#include "framework/vision_engine.hpp"
#include "../mediapipe/mp_wrapper.h"

namespace lll {

// Forward declare to allow existing code to compile if it uses these structs
using EngineParams = framework::VisionEngine::EngineConfig;

class LatentEngine {
public:
    static LatentEngine& get();
    
    // Register backend factory
    using BackendFactory = std::function<std::unique_ptr<framework::VisionEngine>()>;
    void register_backend(const std::string& name, BackendFactory factory);
    
    bool load_model(const EngineParams& params);
    void unload_model();
    
    // New generate signature returning managed data
    framework::VisionEngine::ImageData generate(const framework::VisionEngine::GenerationParams& params);
    
    // VAE Interface
    framework::VisionEngine::LatentData encode(const framework::VisionEngine::ImageData& image);
    framework::VisionEngine::ImageData decode(const framework::VisionEngine::LatentData& latent);
    
    bool is_ready() const;
    std::string current_model_path() const;

private:
    LatentEngine();
    ~LatentEngine();
    
    std::recursive_mutex engine_mutex_;
    std::unique_ptr<framework::VisionEngine> active_engine_;
    std::map<std::string, BackendFactory> backends_;
    
    EngineParams current_params_;
    
    // MediaPipe Integration
public:
    void init_mediapipe(bool use_gpu);
    void run_mediapipe_preprocess(const uint8_t* data, int width, int height, mp_graph_type_t graph_type);
    mp_runner_t* get_mp_runner() const { return mp_runner_; }
    
private:
    mp_runner_t* mp_runner_ = nullptr;
};

} // namespace lll
