#pragma once
#include <vector>
#include <string>
#include <cstdint>
#include <memory>
#include <functional>

namespace lll {
namespace framework {

// Abstract base class for vision inference engines
class VisionEngine {
public:
    virtual ~VisionEngine() = default;
    
    struct ImageData {
        std::vector<uint8_t> data;  // RGB or RGBA, managed by vector
        uint32_t width;
        uint32_t height;
        uint32_t channels;  // 3=RGB, 4=RGBA
    };
    
    struct LatentData {
        std::vector<float> data;
        uint32_t width;
        uint32_t height;
        uint32_t channels;
    };
    
    struct GenerationParams {
        std::string prompt;
        std::string negative_prompt;
        uint32_t width = 512;
        uint32_t height = 512;
        int steps = 20;
        float cfg_scale = 7.0f;
        float strength = 0.7f;
        int64_t seed = -1;
        
        // Optional init/control images
        const uint8_t* init_image = nullptr; // Raw buffer pointer
        const uint8_t* control_image = nullptr; // Raw buffer pointer
        float control_strength = 0.8f;
        
        // Backend specific (defaults to -1/default)
        int sample_method = -1;
        int scheduler = -1;
    };
    
    struct EngineConfig {
        std::string model_path;
        int num_threads = 4;
        bool use_gpu = true;
        size_t vram_limit_mb = 0;  // 0 = no limit
        
        // Extra backend-specific config
        std::string vae_path;
        std::string taesd_path;
        std::string controlnet_path;
        std::string lora_path;
        bool flash_attn = false;
    };
    
    // Lifecycle
    virtual bool initialize(const EngineConfig& config) = 0;
    virtual void shutdown() = 0;
    virtual bool is_ready() const = 0;
    
    // Inference
    // Returns a raw pointer that the caller might need to free? 
    // Or better, returns a helper struct or unique_ptr.
    // Assuming implementation returns ImageData or similar.
    // For compatibility with current codebase which expects sd_image_t*, we might need a bridge, 
    // but the abstraction should be clean.
    // Let's stick to the review's suggestion of ImageData.
    virtual ImageData generate(const GenerationParams& params) = 0;
    
    // VAE Interface
    virtual LatentData encode_latent(const ImageData& image) = 0;
    virtual ImageData decode_latent(const LatentData& latent) = 0;
    
    // Metadata
    virtual std::string backend_name() const = 0;
    virtual size_t get_vram_usage_mb() const = 0;
    
    // Optional progress callback
    using ProgressCallback = std::function<void(int step, int total_steps, float time_sec)>;
    virtual void set_progress_callback(ProgressCallback callback) {}
};

} // namespace framework
} // namespace lll
