#pragma once
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace lll {

inline void convert_rgba_to_rgb(const uint8_t* src, uint8_t* dst, size_t pixel_count) {
    const uint8_t* src_end = src + (pixel_count * 4);
    
    // Pointer arithmetic is generally faster than indexed access
    while (src < src_end) {
        *dst++ = *src++;  // R
        *dst++ = *src++;  // G
        *dst++ = *src++;  // B
        src++;            // Skip A
    }
}

inline void convert_rgb_to_rgba(const uint8_t* src, uint8_t* dst, size_t pixel_count) {
    const uint8_t* src_end = src + (pixel_count * 3);
    
    while (src < src_end) {
        *dst++ = *src++;  // R
        *dst++ = *src++;  // G
        *dst++ = *src++;  // B
        *dst++ = 255;     // A
    }
}


} // namespace lll

// Extended namespace lll coverage



// Parameters for endless morphing transformations
struct TransformParams {
    float zoom_factor = 1.0f;
    float pan_x = 0.0f;
    float pan_y = 0.0f;
    float rotation_deg = 0.0f;
};



/**
 * High-performance bilinear image scaling with buffer validation
 * Supports OpenMP parallelization for multi-core speedup
 * @return true if resize succeeded, false if buffer sizes are insufficient
 */
inline bool resize_bilinear(
    const uint8_t* src, size_t src_size, int src_w, int src_h,
    uint8_t* dst, size_t dst_size, int dst_w, int dst_h)
{
    // Validate dimensions
    if (src_w <= 0 || src_h <= 0 || dst_w <= 0 || dst_h <= 0) {
        return false;
    }
    
    // Validate buffer capacities (4 bytes per pixel for RGBA)
    const size_t required_src = static_cast<size_t>(src_w) * src_h * 4;
    const size_t required_dst = static_cast<size_t>(dst_w) * dst_h * 4;
    
    if (src_size < required_src || dst_size < required_dst) {
        return false;
    }

    const float x_ratio = (float)(src_w - 1) / (float)dst_w;
    const float y_ratio = (float)(src_h - 1) / (float)dst_h;

    // Use OpenMP for parallel row processing if available
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(dst_h > 64)
    #endif
    for (int i = 0; i < dst_h; i++) {
        const float y_float = y_ratio * i;
        const int y = (int)y_float;
        const int y_next = std::min(y + 1, src_h - 1);
        const float y_diff = y_float - y;
        const float y_diff_inv = 1.0f - y_diff;
        
        for (int j = 0; j < dst_w; j++) {
            const float x_float = x_ratio * j;
            const int x = (int)x_float;
            const int x_next = std::min(x + 1, src_w - 1);
            const float x_diff = x_float - x;
            const float x_diff_inv = 1.0f - x_diff;

            const int idx_tl = (y * src_w + x) * 4;        // Top-left
            const int idx_tr = (y * src_w + x_next) * 4;   // Top-right
            const int idx_bl = (y_next * src_w + x) * 4;   // Bottom-left
            const int idx_br = (y_next * src_w + x_next) * 4; // Bottom-right

            const int idx_dst = (i * dst_w + j) * 4;

            // Pre-calculate weights
            const float w_tl = x_diff_inv * y_diff_inv;
            const float w_tr = x_diff * y_diff_inv;
            const float w_bl = x_diff_inv * y_diff;
            const float w_br = x_diff * y_diff;

            // Process all 4 channels (RGBA)
            for (int k = 0; k < 4; k++) {
                float pixel = src[idx_tl + k] * w_tl +
                              src[idx_tr + k] * w_tr +
                              src[idx_bl + k] * w_bl +
                              src[idx_br + k] * w_br;

                dst[idx_dst + k] = (uint8_t)std::clamp(pixel, 0.0f, 255.0f);
            }
        }
    }
    
    return true;
}

/**
 * Calculates the Mean Squared Error (MSE) between two RGBA images.
 * Useful for detecting if an image has changed significantly.
 * @return MSE value (0.0 means identical, higher means more different)
 */
inline float calculate_image_difference(
    const uint8_t* img1, const uint8_t* img2, size_t size)
{
    if (size == 0) return 0.0f;
    
    double diff_sum = 0.0;
    
    // We sample every 4th pixel for performance (stride=16 bytes)
    // or just process every pixel if it's small enough.
    // For OBS filtering, we usually have enough CPU for full check.
    #ifdef _OPENMP
    #pragma omp parallel for reduction(+:diff_sum) schedule(static)
    #endif
    for (size_t i = 0; i < size; i += 4) {
        // Compare RGB channels, skip Alpha
        int r_diff = (int)img1[i] - (int)img2[i];
        int g_diff = (int)img1[i+1] - (int)img2[i+1];
        int b_diff = (int)img1[i+2] - (int)img2[i+2];
        
        diff_sum += (r_diff * r_diff + g_diff * g_diff + b_diff * b_diff);
    }
    
    // Mean of squared differences per channel (3 channels used)
    return (float)(diff_sum / (size / 4 * 3));
}

/**
 * Fast nearest-neighbor scaling for preview purposes with buffer validation
 * Much faster than bilinear but lower quality
 * @return true if resize succeeded, false if buffer sizes are insufficient
 */
inline bool resize_nearest(
    const uint8_t* src, size_t src_size, int src_w, int src_h,
    uint8_t* dst, size_t dst_size, int dst_w, int dst_h)
{
    // Validate dimensions
    if (src_w <= 0 || src_h <= 0 || dst_w <= 0 || dst_h <= 0) {
        return false;
    }
    
    // Validate buffer capacities (4 bytes per pixel for RGBA)
    const size_t required_src = static_cast<size_t>(src_w) * src_h * 4;
    const size_t required_dst = static_cast<size_t>(dst_w) * dst_h * 4;
    
    if (src_size < required_src || dst_size < required_dst) {
        return false;
    }

    const float x_ratio = (float)src_w / (float)dst_w;
    const float y_ratio = (float)src_h / (float)dst_h;

    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(dst_h > 64)
    #endif
    for (int i = 0; i < dst_h; i++) {
        const int y = (int)(y_ratio * i);
        for (int j = 0; j < dst_w; j++) {
            const int x = (int)(x_ratio * j);
            const int idx_src = (y * src_w + x) * 4;
            const int idx_dst = (i * dst_w + j) * 4;
            
            // Copy all 4 channels using memcpy (const-correct)
            std::memcpy(dst + idx_dst, src + idx_src, 4);
        }
    }
    
    return true;
}

/**
 * Transforms an RGBA image with Scale (Zoom) and Translation (Pan).
 * Uses bilinear interpolation for quality.
 *
 * @param params: Struct containing zoom_factor, pan_x, pan_y
 */
inline bool transform_image(
    const uint8_t* src, size_t src_size, int width, int height,
    uint8_t* dst, size_t dst_size,
    const TransformParams& params)
{
    if (width <= 0 || height <= 0) return false;

    const size_t required = static_cast<size_t>(width) * height * 4;
    if (src_size < required || dst_size < required) return false;

    // Center coordinates
    const float cx = width * 0.5f;
    const float cy = height * 0.5f;

    const float scale = params.zoom_factor;
    const float pan_x = params.pan_x;
    const float pan_y = params.pan_y;
    const float rot_rad = params.rotation_deg * (3.14159265359f / 180.0f);
    const float cos_a = std::cos(rot_rad);
    const float sin_a = std::sin(rot_rad);

    // Parallelize processing
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(height > 64)
    #endif
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // Inverse mapping: Map destination pixel (x, y) back to source pixel (src_x, src_y)
            // First apply translation relative to center and pan
            const float rel_x = x - cx - pan_x;
            const float rel_y = y - cy - pan_y;

            // Apply inverse rotation (rotate by -angle)
            // x' = x * cos(-a) - y * sin(-a) = x * cos(a) + y * sin(a)
            // y' = x * sin(-a) + y * cos(-a) = -x * sin(a) + y * cos(a)
            const float rot_x = rel_x * cos_a + rel_y * sin_a;
            const float rot_y = -rel_x * sin_a + rel_y * cos_a;

            // Apply inverse scale and translate back to center
            float src_x = rot_x / scale + cx;
            float src_y = rot_y / scale + cy;

            // Bilinear Interpolation
            int x0 = (int)std::floor(src_x);
            int y0 = (int)std::floor(src_y);
            int x1 = x0 + 1;
            int y1 = y0 + 1;

            // Clamping/Border handling (Clamp to Edge)
            float dx = src_x - x0;
            float dy = src_y - y0;

            // Clamp coordinates for sampling
            int x0_c = std::clamp(x0, 0, width - 1);
            int x1_c = std::clamp(x1, 0, width - 1);
            int y0_c = std::clamp(y0, 0, height - 1);
            int y1_c = std::clamp(y1, 0, height - 1);

            int idx_tl = (y0_c * width + x0_c) * 4;
            int idx_tr = (y0_c * width + x1_c) * 4;
            int idx_bl = (y1_c * width + x0_c) * 4;
            int idx_br = (y1_c * width + x1_c) * 4;

            int dst_idx = (y * width + x) * 4;

            for (int c = 0; c < 4; c++) {
                float val =
                    src[idx_tl + c] * (1.0f - dx) * (1.0f - dy) +
                    src[idx_tr + c] * dx * (1.0f - dy) +
                    src[idx_bl + c] * (1.0f - dx) * dy +
                    src[idx_br + c] * dx * dy;

                dst[dst_idx + c] = (uint8_t)std::clamp(val, 0.0f, 255.0f);
            }
        }
    }
    return true;
}

/**
 * Stack Blur Algorithm (Fast Gaussian approximation)
 * Radius = 0 to ~255
 */
inline void stack_blur_rgba(uint8_t* pix, int w, int h, int radius) {

    if (radius < 1) return;
    int wm = w - 1;
    int hm = h - 1;
    int wh = w * h;
    int div = radius + radius + 1;
    int r[wh], g[wh], b[wh];
    int rsum, gsum, bsum, x, y, i, p, yp, yi, yw;
    int vmin[std::max(w, h)];
    int divsum = (div + 1) >> 1;
    divsum *= divsum;
    int dv[256 * divsum];
    for (i = 0; i < 256 * divsum; i++) {
        dv[i] = (i / divsum);
    }
    yw = yi = 0;
    int stack[div][3];
    int stackpointer;
    int stackstart;
    int *sir;
    int rbs;
    int r1 = radius + 1;
    int routsum, goutsum, boutsum;
    int rinsum, ginsum, binsum;

    for (y = 0; y < h; y++) {
        rinsum = ginsum = binsum = routsum = goutsum = boutsum = rsum = gsum = bsum = 0;
        for (i = -radius; i <= radius; i++) {
            p = pix[std::min(wm, std::max(i, 0)) * 4 + y * w * 4];
            sir = stack[i + radius];
            sir[0] = (p >> 0) & 0xff;
            sir[1] = (p >> 8) & 0xff; // Wrong shift for uint8 array, fixing below
            sir[2] = (p >> 16) & 0xff; // Logic assumes int32 pixel? 
            
            // Correct logic for RGBA buffer
            int idx = std::min(wm, std::max(i, 0)) * 4 + y * w * 4;
            sir[0] = pix[idx];
            sir[1] = pix[idx+1];
            sir[2] = pix[idx+2];
            
            rbs = r1 - std::abs(i);
            rsum += sir[0] * rbs;
            gsum += sir[1] * rbs;
            bsum += sir[2] * rbs;
            if (i > 0) {
                rinsum += sir[0];
                ginsum += sir[1];
                binsum += sir[2];
            } else {
                routsum += sir[0];
                goutsum += sir[1];
                boutsum += sir[2];
            }
        }
        stackpointer = radius;

        for (x = 0; x < w; x++) {
            r[yi] = dv[rsum];
            g[yi] = dv[gsum];
            b[yi] = dv[bsum];

            rsum -= routsum;
            gsum -= goutsum;
            bsum -= boutsum;

            stackstart = stackpointer - radius + div;
            sir = stack[stackstart % div];

            routsum -= sir[0];
            goutsum -= sir[1];
            boutsum -= sir[2];

            if (y == 0) {
                vmin[x] = std::min(x + radius + 1, wm);
            }
            // Reading rightmost pixel
            int idx = vmin[x] * 4 + y * w * 4;
            sir[0] = pix[idx];
            sir[1] = pix[idx+1];
            sir[2] = pix[idx+2];

            rinsum += sir[0];
            ginsum += sir[1];
            binsum += sir[2];

            rsum += rinsum;
            gsum += ginsum;
            bsum += binsum;

            stackpointer = (stackpointer + 1) % div;
            sir = stack[(stackpointer) % div];

            routsum += sir[0];
            goutsum += sir[1];
            boutsum += sir[2];

            rinsum -= sir[0];
            ginsum -= sir[1];
            binsum -= sir[2];

            yi++;
        }
        yw += w;
    }
    
    // Vertical Pass
    for (x = 0; x < w; x++) {
        rinsum = ginsum = binsum = routsum = goutsum = boutsum = rsum = gsum = bsum = 0;
        yp = -radius * w;
        for (i = -radius; i <= radius; i++) {
            yi = std::max(0, yp) + x;

            sir = stack[i + radius];
            
            // Read pre-blurred intermediate result
            sir[0] = r[yi];
            sir[1] = g[yi];
            sir[2] = b[yi];

            rbs = r1 - std::abs(i);
            rsum += sir[0] * rbs;
            gsum += sir[1] * rbs;
            bsum += sir[2] * rbs;

            if (i > 0) {
                rinsum += sir[0];
                ginsum += sir[1];
                binsum += sir[2];
            } else {
                routsum += sir[0];
                goutsum += sir[1];
                boutsum += sir[2];
            }

            if (i < hm) {
                yp += w;
            }
        }
        yi = x;
        stackpointer = radius;

        for (y = 0; y < h; y++) {
            // Write to output buffer
            int idx = yi * 4;
            pix[idx] = (uint8_t)dv[rsum];
            pix[idx+1] = (uint8_t)dv[gsum];
            pix[idx+2] = (uint8_t)dv[bsum];
            pix[idx+3] = 255; // Alpha opaque

            rsum -= routsum;
            gsum -= goutsum;
            bsum -= boutsum;

            stackstart = stackpointer - radius + div;
            sir = stack[stackstart % div];

            routsum -= sir[0];
            goutsum -= sir[1];
            boutsum -= sir[2];

            if (x == 0) {
                vmin[y] = std::min(y + r1, hm) * w;
            }
            int p_idx = x + vmin[y];
            
            sir[0] = r[p_idx];
            sir[1] = g[p_idx];
            sir[2] = b[p_idx];

            rinsum += sir[0];
            ginsum += sir[1];
            binsum += sir[2];

            rsum += rinsum;
            gsum += ginsum;
            bsum += binsum;

            stackpointer = (stackpointer + 1) % div;
            sir = stack[stackpointer];

            routsum += sir[0];
            goutsum += sir[1];
            boutsum += sir[2];

            rinsum -= sir[0];
            ginsum -= sir[1];
            binsum -= sir[2];

            yi += w;
        }
    }
}



