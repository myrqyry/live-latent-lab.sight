#pragma once
/**
 * Point Tracker - Sparse Feature Tracking
 * 
 * Detects and tracks key points across video frames for:
 * - Temporally stable control signals
 * - Motion trajectory visualization
 * - Scene change detection
 * 
 * Algorithm:
 * 1. Detect corners using Harris or FAST-like methods
 * 2. Track points frame-to-frame using template matching
 * 3. Remove lost points, add new ones in sparse regions
 * 4. Optional: Smooth trajectories with simple filtering
 */

#include <cstdint>
#include <vector>

namespace lll {

/**
 * Tracked point with position, velocity, and metadata
 */
struct TrackedPoint {
    float x = 0.0f;           // Current X position (normalized 0-1)
    float y = 0.0f;           // Current Y position (normalized 0-1)
    float vx = 0.0f;          // X velocity (pixels per frame)
    float vy = 0.0f;          // Y velocity (pixels per frame)
    int age = 0;              // Frames since detection
    float score = 0.0f;       // Tracking confidence
    bool valid = false;       // Is this point still being tracked?
    
    // History for trajectory rendering
    std::vector<float> history_x;
    std::vector<float> history_y;
    int max_history = 30;     // Max frames of history
    
    void add_to_history() {
        history_x.push_back(x);
        history_y.push_back(y);
        if (history_x.size() > static_cast<size_t>(max_history)) {
            history_x.erase(history_x.begin());
            history_y.erase(history_y.begin());
        }
    }
};

/**
 * Point tracker configuration
 */
struct PointTrackerConfig {
    int max_points = 100;           // Maximum points to track
    float detection_threshold = 30.0f;  // Corner detection threshold
    int detection_block_size = 3;   // Block size for corner detection
    int search_radius = 16;         // Template matching search radius
    int template_size = 7;          // Template size for matching (odd)
    float min_distance = 10.0f;     // Minimum distance between points
    int history_length = 30;        // Trajectory history length
    float velocity_smoothing = 0.3f;  // Velocity temporal smoothing
};

/**
 * Point Tracker Engine
 */
class PointTracker {
public:
    PointTracker();
    ~PointTracker() = default;
    
    /**
     * Configure tracker parameters
     */
    void configure(const PointTrackerConfig& cfg);
    
    /**
     * Process a new frame
     * 
     * @param rgba Current frame (RGBA, 4 bytes per pixel)
     * @param width Frame width
     * @param height Frame height
     * @return true if processing succeeded
     */
    bool update(const uint8_t* rgba, int width, int height);
    
    /**
     * Get currently tracked points
     */
    const std::vector<TrackedPoint>& get_points() const { return points_; }
    
    /**
     * Get number of valid (actively tracked) points
     */
    int get_valid_count() const;
    
    /**
     * Render tracked points as a control image
     * 
     * @param rgb_out Output RGB buffer (3 bytes per pixel)
     * @param width Output width
     * @param height Output height
     * @param draw_points Draw current positions as dots
     * @param draw_trails Draw motion trails
     * @param bg_r, bg_g, bg_b Background color
     * @param fg_r, fg_g, fg_b Foreground color
     */
    void render_control_image(
        uint8_t* rgb_out, int width, int height,
        bool draw_points = true,
        bool draw_trails = true,
        uint8_t bg_r = 0, uint8_t bg_g = 0, uint8_t bg_b = 0,
        uint8_t fg_r = 255, uint8_t fg_g = 255, uint8_t fg_b = 255
    ) const;
    
    /**
     * Get average motion magnitude
     */
    float get_average_motion() const;
    
    /**
     * Reset all tracking state
     */
    void reset();

private:
    PointTrackerConfig config_;
    std::vector<TrackedPoint> points_;
    std::vector<uint8_t> prev_gray_;
    std::vector<uint8_t> curr_gray_;
    std::vector<float> corner_response_;
    int last_width_ = 0;
    int last_height_ = 0;
    bool initialized_ = false;
    
    // Convert RGBA to grayscale
    void to_grayscale(const uint8_t* rgba, uint8_t* gray, int width, int height);
    
    // Detect corners using Shi-Tomasi/Harris-like method
    void detect_corners(const uint8_t* gray, int width, int height);
    
    // Track existing points to new frame
    void track_points(const uint8_t* prev, const uint8_t* curr, int width, int height);
    
    // Add new points in sparse regions
    void add_new_points(int width, int height);
    
    // Remove lost or low-quality points
    void prune_points();
    
    // Compute Sum of Squared Differences for template matching
    float compute_ssd(const uint8_t* img, int width, int height,
                      int cx, int cy, int tx, int ty);
};

} // namespace lll
