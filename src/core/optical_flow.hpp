#pragma once
/**
 * Optical Flow - Block-Matching Motion Estimation
 * 
 * Provides sparse optical flow computation for:
 * - Frame interpolation via warping
 * - Keyframe propagation (run diffusion on keyframes, warp intermediate frames)
 * - Motion detection for adaptive processing
 * 
 * Algorithm: Block-matching with Sum of Absolute Differences (SAD)
 * - Divide image into blocks (8x8 or 16x16)
 * - Search for best match in reference frame within a window
 * - Interpolate block-level motion to pixel-level
 * 
 * This is a fast approximation suitable for real-time OBS use.
 */

#include <cstdint>
#include <vector>

namespace lll {

/**
 * Optical flow configuration
 */
struct OpticalFlowConfig {
    int block_size = 8;      // Block size for matching (8 or 16)
    int search_range = 16;   // Search window radius in pixels
    float smoothing = 0.5f;  // Temporal smoothing factor (0 = none, 1 = full)
    bool subpixel = false;   // Enable subpixel refinement (slower but smoother)
};

/**
 * Motion vector field
 */
struct FlowField {
    std::vector<float> dx;   // Horizontal displacement per pixel
    std::vector<float> dy;   // Vertical displacement per pixel
    int width = 0;
    int height = 0;
    
    void resize(int w, int h) {
        width = w;
        height = h;
        dx.resize(w * h, 0.0f);
        dy.resize(w * h, 0.0f);
    }
    
    void clear() {
        std::fill(dx.begin(), dx.end(), 0.0f);
        std::fill(dy.begin(), dy.end(), 0.0f);
    }
    
    // Get flow at a point (bilinear interpolation)
    void get(float x, float y, float& out_dx, float& out_dy) const;
};

/**
 * Optical Flow Engine
 * 
 * Computes motion between frames and provides warping functionality.
 */
class OpticalFlow {
public:
    OpticalFlow();
    ~OpticalFlow() = default;
    
    /**
     * Configure the optical flow parameters
     */
    void configure(const OpticalFlowConfig& config);
    
    /**
     * Compute flow between previous and current frame
     * 
     * @param prev Previous frame (grayscale, 1 byte per pixel)
     * @param curr Current frame (grayscale, 1 byte per pixel)
     * @param width Frame width
     * @param height Frame height
     * @return true if computation succeeded
     */
    bool compute(const uint8_t* prev, const uint8_t* curr, int width, int height);
    
    /**
     * Compute flow from RGBA frames (converts internally)
     */
    bool compute_rgba(const uint8_t* prev, const uint8_t* curr, int width, int height);
    
    /**
     * Warp an RGBA image using the computed flow field
     * Performs backward warping (samples from src at src + flow)
     * 
     * @param src Source image to warp (RGBA)
     * @param dst Destination buffer (RGBA)
     * @param width Image width
     * @param height Image height
     * @param scale Flow scale multiplier (1.0 = full motion, 0.5 = half)
     * @return true if warp succeeded
     */
    bool warp(const uint8_t* src, uint8_t* dst, int width, int height, float scale = 1.0f);
    
    /**
     * Get the computed flow field (read-only)
     */
    const FlowField& get_flow() const { return flow_; }
    
    /**
     * Calculate the average motion magnitude
     * Useful for detecting scene changes
     */
    float get_average_motion() const;
    
    /**
     * Check if flow is ready (computed at least once)
     */
    bool is_ready() const { return ready_; }

private:
    OpticalFlowConfig config_;
    FlowField flow_;
    FlowField flow_prev_;      // Previous flow for temporal smoothing
    std::vector<uint8_t> gray_prev_;
    std::vector<uint8_t> gray_curr_;
    bool ready_ = false;
    
    // Block-level flow (before interpolation)
    std::vector<float> block_dx_;
    std::vector<float> block_dy_;
    int blocks_x_ = 0;
    int blocks_y_ = 0;
    
    // Convert RGBA to grayscale
    void to_grayscale(const uint8_t* rgba, uint8_t* gray, int width, int height);
    
    // Compute SAD between two blocks
    int compute_sad(const uint8_t* prev, const uint8_t* curr, int width,
                    int bx, int by, int dx, int dy);
    
    // Search for best match in search window
    void search_block(const uint8_t* prev, const uint8_t* curr, int width, int height,
                      int bx, int by, float& out_dx, float& out_dy);
    
    // Interpolate block-level flow to pixel-level
    void interpolate_flow(int width, int height);
};

/**
 * Keyframe Manager
 * 
 * Manages keyframe-based processing with optical flow interpolation.
 */
class KeyframeManager {
public:
    /**
     * Configuration for keyframe processing
     */
    struct Config {
        int keyframe_interval = 5;       // Process every Nth frame
        float scene_change_threshold = 50.0f;  // MSE threshold for forcing keyframe
        bool enabled = true;
    };
    
    KeyframeManager();
    
    void configure(const Config& cfg) { config_ = cfg; }
    
    /**
     * Determine if current frame should be a keyframe
     * 
     * @param frame_number Current frame counter
     * @param motion_magnitude Average motion from optical flow
     * @param scene_mse MSE between current and previous frame
     * @return true if this frame should be processed with diffusion
     */
    bool is_keyframe(int frame_number, float motion_magnitude, float scene_mse);
    
    /**
     * Store a keyframe
     */
    void store_keyframe(const uint8_t* rgba, int width, int height);
    
    /**
     * Get the last keyframe
     */
    const std::vector<uint8_t>& get_keyframe() const { return keyframe_; }
    
    /**
     * Check if we have a stored keyframe
     */
    bool has_keyframe() const { return !keyframe_.empty(); }

private:
    Config config_;
    std::vector<uint8_t> keyframe_;
    int last_keyframe_number_ = -1;
};

} // namespace lll
