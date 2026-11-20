#pragma once

#include <cstdint>
#include <vector>
#include <cmath>
#include <algorithm>

/**
 * OpenPose-style skeleton renderer for ControlNet
 * 
 * Renders a colored skeleton image compatible with OpenPose ControlNet models.
 * Colors follow the standard OpenPose color scheme for body, face, and hands.
 */

// Standard COCO keypoint indices (17 points)
enum COCOKeypoint {
    COCO_NOSE = 0,
    COCO_LEFT_EYE = 1,
    COCO_RIGHT_EYE = 2,
    COCO_LEFT_EAR = 3,
    COCO_RIGHT_EAR = 4,
    COCO_LEFT_SHOULDER = 5,
    COCO_RIGHT_SHOULDER = 6,
    COCO_LEFT_ELBOW = 7,
    COCO_RIGHT_ELBOW = 8,
    COCO_LEFT_WRIST = 9,
    COCO_RIGHT_WRIST = 10,
    COCO_LEFT_HIP = 11,
    COCO_RIGHT_HIP = 12,
    COCO_LEFT_KNEE = 13,
    COCO_RIGHT_KNEE = 14,
    COCO_LEFT_ANKLE = 15,
    COCO_RIGHT_ANKLE = 16,
    COCO_KEYPOINT_COUNT = 17
};

// OpenPose limb connections (pairs of keypoint indices)
static const int OPENPOSE_LIMBS[][2] = {
    {COCO_NOSE, COCO_LEFT_EYE},
    {COCO_NOSE, COCO_RIGHT_EYE},
    {COCO_LEFT_EYE, COCO_LEFT_EAR},
    {COCO_RIGHT_EYE, COCO_RIGHT_EAR},
    {COCO_LEFT_SHOULDER, COCO_RIGHT_SHOULDER},
    {COCO_LEFT_SHOULDER, COCO_LEFT_ELBOW},
    {COCO_LEFT_ELBOW, COCO_LEFT_WRIST},
    {COCO_RIGHT_SHOULDER, COCO_RIGHT_ELBOW},
    {COCO_RIGHT_ELBOW, COCO_RIGHT_WRIST},
    {COCO_LEFT_SHOULDER, COCO_LEFT_HIP},
    {COCO_RIGHT_SHOULDER, COCO_RIGHT_HIP},
    {COCO_LEFT_HIP, COCO_RIGHT_HIP},
    {COCO_LEFT_HIP, COCO_LEFT_KNEE},
    {COCO_LEFT_KNEE, COCO_LEFT_ANKLE},
    {COCO_RIGHT_HIP, COCO_RIGHT_KNEE},
    {COCO_RIGHT_KNEE, COCO_RIGHT_ANKLE},
};

static const int OPENPOSE_LIMB_COUNT = sizeof(OPENPOSE_LIMBS) / sizeof(OPENPOSE_LIMBS[0]);

// OpenPose color scheme (BGR for each limb)
// These colors match the standard OpenPose visualization
static const uint8_t OPENPOSE_COLORS[][3] = {
    {255, 0, 85},      // Nose-LEye: magenta-ish
    {255, 0, 0},       // Nose-REye: blue-ish
    {255, 85, 0},      // LEye-LEar: cyan-ish
    {255, 170, 0},     // REye-REar: 
    {255, 255, 0},     // LShoulder-RShoulder: cyan
    {170, 255, 0},     // LShoulder-LElbow: 
    {85, 255, 0},      // LElbow-LWrist: green
    {0, 255, 0},       // RShoulder-RElbow: green
    {0, 255, 85},      // RElbow-RWrist: 
    {0, 255, 170},     // LShoulder-LHip: 
    {0, 255, 255},     // RShoulder-RHip: yellow
    {0, 170, 255},     // LHip-RHip: 
    {0, 85, 255},      // LHip-LKnee: 
    {0, 0, 255},       // LKnee-LAnkle: red
    {85, 0, 255},      // RHip-RKnee: 
    {170, 0, 255},     // RKnee-RAnkle: purple
};

struct Keypoint {
    float x;           // Normalized [0, 1] or pixel coordinates
    float y;           // Normalized [0, 1] or pixel coordinates
    float confidence;  // Detection confidence [0, 1]
    bool valid;        // Whether this keypoint was detected
    
    Keypoint() : x(0), y(0), confidence(0), valid(false) {}
    Keypoint(float x_, float y_, float conf = 1.0f) 
        : x(x_), y(y_), confidence(conf), valid(conf > 0.0f) {}
};

struct PoseKeypoints {
    Keypoint points[COCO_KEYPOINT_COUNT];
    
    PoseKeypoints() = default;
    
    // Set keypoint by index
    void set(int idx, float x, float y, float conf = 1.0f) {
        if (idx >= 0 && idx < COCO_KEYPOINT_COUNT) {
            points[idx] = Keypoint(x, y, conf);
        }
    }
    
    // Get keypoint by index
    const Keypoint& get(int idx) const {
        static Keypoint invalid;
        if (idx >= 0 && idx < COCO_KEYPOINT_COUNT) {
            return points[idx];
        }
        return invalid;
    }
};

/**
 * Draw a thick anti-aliased line on an RGB image
 */
inline void draw_line_rgb(
    uint8_t* img, int width, int height,
    int x1, int y1, int x2, int y2,
    uint8_t r, uint8_t g, uint8_t b,
    int thickness = 4)
{
    // Bresenham's line algorithm with thickness
    int dx = std::abs(x2 - x1);
    int dy = std::abs(y2 - y1);
    int sx = (x1 < x2) ? 1 : -1;
    int sy = (y1 < y2) ? 1 : -1;
    int err = dx - dy;
    
    int half_thick = thickness / 2;
    
    while (true) {
        // Draw a circle at each point for thickness
        for (int ty = -half_thick; ty <= half_thick; ty++) {
            for (int tx = -half_thick; tx <= half_thick; tx++) {
                if (tx*tx + ty*ty <= half_thick*half_thick) {
                    int px = x1 + tx;
                    int py = y1 + ty;
                    if (px >= 0 && px < width && py >= 0 && py < height) {
                        int idx = (py * width + px) * 3;
                        img[idx + 0] = r;
                        img[idx + 1] = g;
                        img[idx + 2] = b;
                    }
                }
            }
        }
        
        if (x1 == x2 && y1 == y2) break;
        
        int e2 = 2 * err;
        if (e2 > -dy) {
            err -= dy;
            x1 += sx;
        }
        if (e2 < dx) {
            err += dx;
            y1 += sy;
        }
    }
}

/**
 * Draw a filled circle (keypoint marker) on an RGB image
 */
inline void draw_circle_rgb(
    uint8_t* img, int width, int height,
    int cx, int cy, int radius,
    uint8_t r, uint8_t g, uint8_t b)
{
    for (int y = -radius; y <= radius; y++) {
        for (int x = -radius; x <= radius; x++) {
            if (x*x + y*y <= radius*radius) {
                int px = cx + x;
                int py = cy + y;
                if (px >= 0 && px < width && py >= 0 && py < height) {
                    int idx = (py * width + px) * 3;
                    img[idx + 0] = r;
                    img[idx + 1] = g;
                    img[idx + 2] = b;
                }
            }
        }
    }
}

/**
 * Render OpenPose-style skeleton for ControlNet
 * 
 * @param output     Output RGB image buffer (must be width * height * 3 bytes)
 * @param width      Image width
 * @param height     Image height
 * @param pose       Pose keypoints (normalized 0-1 or pixel coordinates)
 * @param normalized Whether keypoints are normalized (0-1) or pixel coordinates
 * @param line_width Skeleton line thickness
 * @param point_radius Keypoint circle radius
 */
inline void render_openpose_skeleton(
    uint8_t* output, int width, int height,
    const PoseKeypoints& pose,
    bool normalized = true,
    int line_width = 4,
    int point_radius = 6)
{
    // Clear to black
    std::memset(output, 0, width * height * 3);
    
    // Convert normalized coordinates to pixel coordinates
    auto to_pixel = [&](float x, float y, int& px, int& py) {
        if (normalized) {
            px = (int)(x * width);
            py = (int)(y * height);
        } else {
            px = (int)x;
            py = (int)y;
        }
    };
    
    // Draw limbs (lines between keypoints)
    for (int i = 0; i < OPENPOSE_LIMB_COUNT; i++) {
        int idx1 = OPENPOSE_LIMBS[i][0];
        int idx2 = OPENPOSE_LIMBS[i][1];
        
        const Keypoint& kp1 = pose.get(idx1);
        const Keypoint& kp2 = pose.get(idx2);
        
        // Only draw if both keypoints are valid
        if (kp1.valid && kp2.valid) {
            int x1, y1, x2, y2;
            to_pixel(kp1.x, kp1.y, x1, y1);
            to_pixel(kp2.x, kp2.y, x2, y2);
            
            // Use OpenPose color for this limb
            draw_line_rgb(output, width, height,
                         x1, y1, x2, y2,
                         OPENPOSE_COLORS[i][0],
                         OPENPOSE_COLORS[i][1],
                         OPENPOSE_COLORS[i][2],
                         line_width);
        }
    }
    
    // Draw keypoints (circles)
    for (int i = 0; i < COCO_KEYPOINT_COUNT; i++) {
        const Keypoint& kp = pose.get(i);
        if (kp.valid) {
            int px, py;
            to_pixel(kp.x, kp.y, px, py);
            
            // Find a color for this keypoint (use average of connected limbs)
            uint8_t r = 255, g = 255, b = 255;  // Default white
            
            // Use the color of the first limb connected to this keypoint
            for (int j = 0; j < OPENPOSE_LIMB_COUNT; j++) {
                if (OPENPOSE_LIMBS[j][0] == i || OPENPOSE_LIMBS[j][1] == i) {
                    r = OPENPOSE_COLORS[j][0];
                    g = OPENPOSE_COLORS[j][1];
                    b = OPENPOSE_COLORS[j][2];
                    break;
                }
            }
            
            draw_circle_rgb(output, width, height, px, py, point_radius, r, g, b);
        }
    }
}

/**
 * Simple motion-based pose estimation (placeholder until real pose estimation)
 * 
 * This creates a rough approximation of body position by analyzing
 * the center of motion in the frame. Not accurate, but provides
 * some structure guidance.
 * 
 * @param current_frame  Current RGB frame
 * @param prev_frame     Previous RGB frame (for motion detection)
 * @param width          Frame width
 * @param height         Frame height
 * @param output_pose    Output pose keypoints
 */
inline void estimate_pose_from_motion(
    const uint8_t* current_frame,
    const uint8_t* prev_frame,
    int width, int height,
    PoseKeypoints& output_pose)
{
    if (!current_frame || !prev_frame) return;
    
    // Find center of motion
    float motion_x = 0, motion_y = 0;
    float total_motion = 0;
    
    for (int y = 0; y < height; y += 4) {  // Sample every 4th pixel for speed
        for (int x = 0; x < width; x += 4) {
            int idx = (y * width + x) * 3;
            
            int diff = std::abs((int)current_frame[idx] - (int)prev_frame[idx]) +
                      std::abs((int)current_frame[idx+1] - (int)prev_frame[idx+1]) +
                      std::abs((int)current_frame[idx+2] - (int)prev_frame[idx+2]);
            
            if (diff > 30) {  // Motion threshold
                motion_x += x * diff;
                motion_y += y * diff;
                total_motion += diff;
            }
        }
    }
    
    if (total_motion > 0) {
        // Center of motion
        float cx = (motion_x / total_motion) / width;
        float cy = (motion_y / total_motion) / height;
        
        // Create a rough skeleton centered on motion
        // This is a very rough approximation!
        output_pose.set(COCO_NOSE, cx, cy - 0.15f, 0.8f);
        output_pose.set(COCO_LEFT_SHOULDER, cx - 0.1f, cy - 0.05f, 0.7f);
        output_pose.set(COCO_RIGHT_SHOULDER, cx + 0.1f, cy - 0.05f, 0.7f);
        output_pose.set(COCO_LEFT_ELBOW, cx - 0.15f, cy + 0.05f, 0.6f);
        output_pose.set(COCO_RIGHT_ELBOW, cx + 0.15f, cy + 0.05f, 0.6f);
        output_pose.set(COCO_LEFT_WRIST, cx - 0.2f, cy + 0.15f, 0.5f);
        output_pose.set(COCO_RIGHT_WRIST, cx + 0.2f, cy + 0.15f, 0.5f);
        output_pose.set(COCO_LEFT_HIP, cx - 0.08f, cy + 0.15f, 0.7f);
        output_pose.set(COCO_RIGHT_HIP, cx + 0.08f, cy + 0.15f, 0.7f);
        output_pose.set(COCO_LEFT_KNEE, cx - 0.08f, cy + 0.35f, 0.6f);
        output_pose.set(COCO_RIGHT_KNEE, cx + 0.08f, cy + 0.35f, 0.6f);
        output_pose.set(COCO_LEFT_ANKLE, cx - 0.08f, cy + 0.5f, 0.5f);
        output_pose.set(COCO_RIGHT_ANKLE, cx + 0.08f, cy + 0.5f, 0.5f);
    }
}
