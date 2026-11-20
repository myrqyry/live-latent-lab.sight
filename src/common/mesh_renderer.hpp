#pragma once
/**
 * Mesh Renderer - MediaPipe Face Mesh Visualization
 * 
 * Renders MediaPipe 468-landmark face mesh as a control image for ControlNet.
 * This provides more temporally stable conditioning than raw RGB or edge detection.
 * 
 * Rendering modes:
 * - Points only: Draw each landmark as a colored dot
 * - Wireframe: Connect landmarks with lines based on face mesh topology
 * - Triangles: Fill triangles with gradient colors
 * - Contours: Draw only face outline (useful for lineart ControlNet)
 */

#include <cstdint>
#include <vector>
#include "mediapipe/mp_wrapper.h"

namespace lll {

/**
 * Face mesh connection indices for wireframe rendering
 * Based on MediaPipe's FACEMESH_TESSELATION connections
 */
struct FaceMeshTopology {
    // Key contour indices for silhouette
    static constexpr int FACE_OVAL[] = {
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
        172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
    };
    static constexpr int FACE_OVAL_COUNT = 36;
    
    // Lip outer contour
    static constexpr int LIPS_OUTER[] = {
        61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409,
        270, 269, 267, 0, 37, 39, 40, 185
    };
    static constexpr int LIPS_OUTER_COUNT = 20;
    
    // Left eye contour
    static constexpr int LEFT_EYE[] = {
        33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246
    };
    static constexpr int LEFT_EYE_COUNT = 16;
    
    // Right eye contour
    static constexpr int RIGHT_EYE[] = {
        362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398
    };
    static constexpr int RIGHT_EYE_COUNT = 16;
    
    // Left eyebrow
    static constexpr int LEFT_EYEBROW[] = {
        70, 63, 105, 66, 107, 55, 65, 52, 53, 46
    };
    static constexpr int LEFT_EYEBROW_COUNT = 10;
    
    // Right eyebrow
    static constexpr int RIGHT_EYEBROW[] = {
        300, 293, 334, 296, 336, 285, 295, 282, 283, 276
    };
    static constexpr int RIGHT_EYEBROW_COUNT = 10;
    
    // Nose bridge
    static constexpr int NOSE[] = {
        168, 6, 197, 195, 5, 4, 1, 19, 94, 2
    };
    static constexpr int NOSE_COUNT = 10;
};

/**
 * Render face mesh landmarks as a control image
 * 
 * @param rgb_out Output RGB image buffer (3 bytes per pixel)
 * @param width Output image width
 * @param height Output image height
 * @param landmarks Array of 468 landmarks from MediaPipe
 * @param landmark_count Number of landmarks (should be 468)
 * @param mode Rendering mode:
 *             0 = Points only (landmarks as dots)
 *             1 = Wireframe (connected lines)
 *             2 = Contours only (face outline for lineart)
 *             3 = Full mesh with triangles
 * @param bg_r, bg_g, bg_b Background color (usually 0,0,0 for black)
 * @param fg_r, fg_g, fg_b Foreground/line color (usually 255,255,255 for white)
 */
void render_face_mesh_control(
    uint8_t* rgb_out,
    int width, int height,
    const mp_landmark_t* landmarks, int landmark_count,
    int mode = 1,
    uint8_t bg_r = 0, uint8_t bg_g = 0, uint8_t bg_b = 0,
    uint8_t fg_r = 255, uint8_t fg_g = 255, uint8_t fg_b = 255
);

/**
 * Render a polyline connecting landmarks by index
 * 
 * @param rgb Image buffer
 * @param width Image width
 * @param height Image height
 * @param landmarks Landmark array
 * @param indices Array of landmark indices to connect
 * @param count Number of indices
 * @param closed If true, connects last point to first
 * @param r, g, b Line color
 * @param thickness Line thickness in pixels
 */
void render_landmark_polyline(
    uint8_t* rgb, int width, int height,
    const mp_landmark_t* landmarks,
    const int* indices, int count,
    bool closed,
    uint8_t r, uint8_t g, uint8_t b,
    int thickness = 1
);

/**
 * Draw a single point at a landmark position
 */
void render_landmark_point(
    uint8_t* rgb, int width, int height,
    float x, float y,  // Normalized coordinates [0, 1]
    uint8_t r, uint8_t g, uint8_t b,
    int radius = 2
);

/**
 * Draw a line between two points using Bresenham's algorithm
 */
void draw_line(
    uint8_t* rgb, int width, int height,
    int x0, int y0, int x1, int y1,
    uint8_t r, uint8_t g, uint8_t b,
    int thickness = 1
);

} // namespace lll
