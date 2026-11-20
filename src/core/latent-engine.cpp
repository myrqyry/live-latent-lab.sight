#include "latent-engine.hpp"
#include "engines/stable_diffusion_backend.hpp"
#include <obs-module.h>
#include <util/platform.h>
#include <cstring>

namespace lll {

LatentEngine::LatentEngine() {
    // Register default backend
    register_backend("stable-diffusion.cpp", []() -> std::unique_ptr<framework::VisionEngine> {
        return std::make_unique<engines::StableDiffusionBackend>();
    });
}

LatentEngine::~LatentEngine() {
    unload_model();
    if (mp_runner_) {
        mp_destroy(mp_runner_);
        mp_runner_ = nullptr;
    }
}

LatentEngine& LatentEngine::get() {
    static LatentEngine instance;
    return instance;
}

void LatentEngine::register_backend(const std::string& name, BackendFactory factory) {
    std::lock_guard<std::recursive_mutex> lock(engine_mutex_);
    backends_[name] = factory;
}

void LatentEngine::unload_model() {
    std::lock_guard<std::recursive_mutex> lock(engine_mutex_);
    if (active_engine_) {
        active_engine_->shutdown();
        active_engine_.reset();
    }
    current_params_.model_path.clear();
}

bool LatentEngine::load_model(const EngineParams& params) {
    std::lock_guard<std::recursive_mutex> lock(engine_mutex_);
    
    // Check if we need to reload
    // Note: Simple check for now. Ideally VisionEngine has a simpler config comparison.
    if (active_engine_ && active_engine_->is_ready() && 
        current_params_.model_path == params.model_path && 
        current_params_.vae_path == params.vae_path &&
        current_params_.controlnet_path == params.controlnet_path &&
        current_params_.lora_path == params.lora_path) {
        return true;
    }
    
    unload_model();
    
    if (params.model_path.empty()) return false;
    
    // Select backend (could be parameterized later)
    auto it = backends_.find("stable-diffusion.cpp");
    if (it == backends_.end()) {
        blog(LOG_ERROR, "[LatentEngine] Default backend not found");
        return false;
    }
    
    blog(LOG_INFO, "[LatentEngine] Initializing backend 'stable-diffusion.cpp' for model: %s", params.model_path.c_str());
    
    active_engine_ = it->second();
    if (!active_engine_->initialize(params)) {
        blog(LOG_ERROR, "[LatentEngine] Failed to initialize backend");
        active_engine_.reset();
        return false;
    }
    
    current_params_ = params;
    return true;
}

bool LatentEngine::is_ready() const {
    // Note: not thread safe if called without lock, but usually called in guarded context or single thread
    return active_engine_ && active_engine_->is_ready();
}

std::string LatentEngine::current_model_path() const {
    return current_params_.model_path;
}

framework::VisionEngine::ImageData LatentEngine::generate(const framework::VisionEngine::GenerationParams& params) {
    std::lock_guard<std::recursive_mutex> lock(engine_mutex_);
    
    if (!active_engine_ || !active_engine_->is_ready()) {
        return {};
    }
    
    return active_engine_->generate(params);
}

framework::VisionEngine::LatentData LatentEngine::encode(const framework::VisionEngine::ImageData& image) {
    std::lock_guard<std::recursive_mutex> lock(engine_mutex_);
    if (!active_engine_ || !active_engine_->is_ready()) return {};
    return active_engine_->encode_latent(image);
}

framework::VisionEngine::ImageData LatentEngine::decode(const framework::VisionEngine::LatentData& latent) {
    std::lock_guard<std::recursive_mutex> lock(engine_mutex_);
    if (!active_engine_ || !active_engine_->is_ready()) return {};
    return active_engine_->decode_latent(latent);
}

void LatentEngine::init_mediapipe(bool use_gpu) {
    std::lock_guard<std::recursive_mutex> lock(engine_mutex_);
    if (mp_runner_) {
        mp_destroy(mp_runner_);
        mp_runner_ = nullptr;
    }
    
    // Try to find models relative to module binary or data dir
    // For now passing nullptr relies on mp_wrapper's internal search or hardcoded paths
    mp_runner_ = mp_create(use_gpu, nullptr);
    
    if (mp_runner_) {
        blog(LOG_INFO, "[LatentEngine] MediaPipe Runner initialized. Backend: %s", mp_get_backend_name());
    } else {
        blog(LOG_ERROR, "[LatentEngine] Failed to initialize MediaPipe Runner");
    }
}

void LatentEngine::run_mediapipe_preprocess(const uint8_t* data, int width, int height, mp_graph_type_t graph_type) {
    std::lock_guard<std::recursive_mutex> lock(engine_mutex_);
    if (!mp_runner_) return;
    
    mp_graph_type_t current = mp_get_current_graph(mp_runner_);
    if (current != graph_type) {
        mp_start_graph(mp_runner_, graph_type);
    }
    
    // Assume RGBA input (4 channels)
    mp_process_frame(mp_runner_, data, width, height, width * 4);
}

} // namespace lll
