/**
 * Mesh Renderer - Implementation
 * 
 * Renders MediaPipe face mesh as control images for ControlNet.
 */

#include "mesh_renderer.hpp"
#include <cmath>
#include <algorithm>
#include <cstring>

namespace lll {

// Static topology arrays (defined in header as constexpr)
constexpr int FaceMeshTopology::FACE_OVAL[];
constexpr int FaceMeshTopology::LIPS_OUTER[];
constexpr int FaceMeshTopology::LEFT_EYE[];
constexpr int FaceMeshTopology::RIGHT_EYE[];
constexpr int FaceMeshTopology::LEFT_EYEBROW[];
constexpr int FaceMeshTopology::RIGHT_EYEBROW[];
constexpr int FaceMeshTopology::NOSE[];

void draw_line(
    uint8_t* rgb, int width, int height,
    int x0, int y0, int x1, int y1,
    uint8_t r, uint8_t g, uint8_t b,
    int thickness)
{
    // Bresenham's line algorithm with thickness
    int dx = std::abs(x1 - x0);
    int dy = std::abs(y1 - y0);
    int sx = (x0 < x1) ? 1 : -1;
    int sy = (y0 < y1) ? 1 : -1;
    int err = dx - dy;
    
    int half_thick = thickness / 2;
    
    while (true) {
        // Draw thick point
        for (int ty = -half_thick; ty <= half_thick; ++ty) {
            for (int tx = -half_thick; tx <= half_thick; ++tx) {
                int px = x0 + tx;
                int py = y0 + ty;
                if (px >= 0 && px < width && py >= 0 && py < height) {
                    int idx = (py * width + px) * 3;
                    rgb[idx + 0] = r;
                    rgb[idx + 1] = g;
                    rgb[idx + 2] = b;
                }
            }
        }
        
        if (x0 == x1 && y0 == y1) break;
        
        int e2 = 2 * err;
        if (e2 > -dy) {
            err -= dy;
            x0 += sx;
        }
        if (e2 < dx) {
            err += dx;
            y0 += sy;
        }
    }
}

void render_landmark_point(
    uint8_t* rgb, int width, int height,
    float x, float y,
    uint8_t r, uint8_t g, uint8_t b,
    int radius)
{
    int px = static_cast<int>(x * width);
    int py = static_cast<int>(y * height);
    
    // Draw filled circle
    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            if (dx * dx + dy * dy <= radius * radius) {
                int cx = px + dx;
                int cy = py + dy;
                if (cx >= 0 && cx < width && cy >= 0 && cy < height) {
                    int idx = (cy * width + cx) * 3;
                    rgb[idx + 0] = r;
                    rgb[idx + 1] = g;
                    rgb[idx + 2] = b;
                }
            }
        }
    }
}

void render_landmark_polyline(
    uint8_t* rgb, int width, int height,
    const mp_landmark_t* landmarks,
    const int* indices, int count,
    bool closed,
    uint8_t r, uint8_t g, uint8_t b,
    int thickness)
{
    if (count < 2) return;
    
    for (int i = 0; i < count - 1; ++i) {
        int idx0 = indices[i];
        int idx1 = indices[i + 1];
        
        // Skip if indices are out of range
        if (idx0 < 0 || idx0 >= 468 || idx1 < 0 || idx1 >= 468) continue;
        
        // Skip if landmarks have low visibility
        if (landmarks[idx0].visibility < 0.5f || landmarks[idx1].visibility < 0.5f) continue;
        
        int x0 = static_cast<int>(landmarks[idx0].x * width);
        int y0 = static_cast<int>(landmarks[idx0].y * height);
        int x1 = static_cast<int>(landmarks[idx1].x * width);
        int y1 = static_cast<int>(landmarks[idx1].y * height);
        
        draw_line(rgb, width, height, x0, y0, x1, y1, r, g, b, thickness);
    }
    
    // Close the polyline if requested
    if (closed && count >= 2) {
        int idx0 = indices[count - 1];
        int idx1 = indices[0];
        
        if (idx0 >= 0 && idx0 < 468 && idx1 >= 0 && idx1 < 468 &&
            landmarks[idx0].visibility >= 0.5f && landmarks[idx1].visibility >= 0.5f) {
            int x0 = static_cast<int>(landmarks[idx0].x * width);
            int y0 = static_cast<int>(landmarks[idx0].y * height);
            int x1 = static_cast<int>(landmarks[idx1].x * width);
            int y1 = static_cast<int>(landmarks[idx1].y * height);
            
            draw_line(rgb, width, height, x0, y0, x1, y1, r, g, b, thickness);
        }
    }
}

void render_face_mesh_control(
    uint8_t* rgb_out,
    int width, int height,
    const mp_landmark_t* landmarks, int landmark_count,
    int mode,
    uint8_t bg_r, uint8_t bg_g, uint8_t bg_b,
    uint8_t fg_r, uint8_t fg_g, uint8_t fg_b)
{
    if (!rgb_out || !landmarks || landmark_count < 468 || width <= 0 || height <= 0) {
        return;
    }
    
    // Clear to background color
    for (int i = 0; i < width * height; ++i) {
        rgb_out[i * 3 + 0] = bg_r;
        rgb_out[i * 3 + 1] = bg_g;
        rgb_out[i * 3 + 2] = bg_b;
    }
    
    int thickness = std::max(1, std::min(width, height) / 150);  // Adaptive thickness
    int point_radius = thickness + 1;
    
    switch (mode) {
        case 0:  // Points only
            for (int i = 0; i < landmark_count; ++i) {
                if (landmarks[i].visibility >= 0.5f) {
                    render_landmark_point(rgb_out, width, height,
                                         landmarks[i].x, landmarks[i].y,
                                         fg_r, fg_g, fg_b, point_radius);
                }
            }
            break;
            
        case 1:  // Wireframe (all contours)
            // Face oval
            render_landmark_polyline(rgb_out, width, height, landmarks,
                                    FaceMeshTopology::FACE_OVAL, 
                                    FaceMeshTopology::FACE_OVAL_COUNT,
                                    true, fg_r, fg_g, fg_b, thickness);
            
            // Eyes
            render_landmark_polyline(rgb_out, width, height, landmarks,
                                    FaceMeshTopology::LEFT_EYE,
                                    FaceMeshTopology::LEFT_EYE_COUNT,
                                    true, fg_r, fg_g, fg_b, thickness);
            render_landmark_polyline(rgb_out, width, height, landmarks,
                                    FaceMeshTopology::RIGHT_EYE,
                                    FaceMeshTopology::RIGHT_EYE_COUNT,
                                    true, fg_r, fg_g, fg_b, thickness);
            
            // Eyebrows
            render_landmark_polyline(rgb_out, width, height, landmarks,
                                    FaceMeshTopology::LEFT_EYEBROW,
                                    FaceMeshTopology::LEFT_EYEBROW_COUNT,
                                    false, fg_r, fg_g, fg_b, thickness);
            render_landmark_polyline(rgb_out, width, height, landmarks,
                                    FaceMeshTopology::RIGHT_EYEBROW,
                                    FaceMeshTopology::RIGHT_EYEBROW_COUNT,
                                    false, fg_r, fg_g, fg_b, thickness);
            
            // Lips
            render_landmark_polyline(rgb_out, width, height, landmarks,
                                    FaceMeshTopology::LIPS_OUTER,
                                    FaceMeshTopology::LIPS_OUTER_COUNT,
                                    true, fg_r, fg_g, fg_b, thickness);
            
            // Nose
            render_landmark_polyline(rgb_out, width, height, landmarks,
                                    FaceMeshTopology::NOSE,
                                    FaceMeshTopology::NOSE_COUNT,
                                    false, fg_r, fg_g, fg_b, thickness);
            break;
            
        case 2:  // Contours only (just face oval and eyes - for lineart)
            render_landmark_polyline(rgb_out, width, height, landmarks,
                                    FaceMeshTopology::FACE_OVAL,
                                    FaceMeshTopology::FACE_OVAL_COUNT,
                                    true, fg_r, fg_g, fg_b, thickness);
            render_landmark_polyline(rgb_out, width, height, landmarks,
                                    FaceMeshTopology::LEFT_EYE,
                                    FaceMeshTopology::LEFT_EYE_COUNT,
                                    true, fg_r, fg_g, fg_b, thickness);
            render_landmark_polyline(rgb_out, width, height, landmarks,
                                    FaceMeshTopology::RIGHT_EYE,
                                    FaceMeshTopology::RIGHT_EYE_COUNT,
                                    true, fg_r, fg_g, fg_b, thickness);
            break;
            
        case 3:  // Full mesh - render all 468 points connected
            // For simplicity, just render all points with slightly larger radius
            for (int i = 0; i < landmark_count; ++i) {
                if (landmarks[i].visibility >= 0.3f) {
                    // Color based on z-depth for 3D effect
                    float depth = landmarks[i].z;
                    uint8_t intensity = static_cast<uint8_t>(
                        std::clamp((0.5f - depth) * 255.0f, 0.0f, 255.0f)
                    );
                    render_landmark_point(rgb_out, width, height,
                                         landmarks[i].x, landmarks[i].y,
                                         intensity, intensity, intensity, point_radius);
                }
            }
            
            // Also draw main contours
            render_landmark_polyline(rgb_out, width, height, landmarks,
                                    FaceMeshTopology::FACE_OVAL,
                                    FaceMeshTopology::FACE_OVAL_COUNT,
                                    true, fg_r, fg_g, fg_b, thickness);
            break;
            
        default:
            // Default to wireframe
            render_landmark_polyline(rgb_out, width, height, landmarks,
                                    FaceMeshTopology::FACE_OVAL,
                                    FaceMeshTopology::FACE_OVAL_COUNT,
                                    true, fg_r, fg_g, fg_b, thickness);
            break;
    }
}

} // namespace lll
