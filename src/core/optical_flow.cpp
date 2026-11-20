/**
 * Optical Flow - Implementation
 * 
 * Block-matching optical flow with SAD metric.
 */

#include "optical_flow.hpp"
#include <cmath>
#include <algorithm>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace lll {

// FlowField bilinear interpolation
void FlowField::get(float x, float y, float& out_dx, float& out_dy) const {
    if (width <= 0 || height <= 0 || dx.empty()) {
        out_dx = out_dy = 0.0f;
        return;
    }
    
    // Clamp coordinates
    x = std::max(0.0f, std::min(x, (float)(width - 1)));
    y = std::max(0.0f, std::min(y, (float)(height - 1)));
    
    int x0 = (int)x;
    int y0 = (int)y;
    int x1 = std::min(x0 + 1, width - 1);
    int y1 = std::min(y0 + 1, height - 1);
    
    float fx = x - x0;
    float fy = y - y0;
    
    // Bilinear interpolation for dx
    float d00 = dx[y0 * width + x0];
    float d10 = dx[y0 * width + x1];
    float d01 = dx[y1 * width + x0];
    float d11 = dx[y1 * width + x1];
    
    out_dx = d00 * (1 - fx) * (1 - fy) +
             d10 * fx * (1 - fy) +
             d01 * (1 - fx) * fy +
             d11 * fx * fy;
    
    // Bilinear interpolation for dy
    d00 = dy[y0 * width + x0];
    d10 = dy[y0 * width + x1];
    d01 = dy[y1 * width + x0];
    d11 = dy[y1 * width + x1];
    
    out_dy = d00 * (1 - fx) * (1 - fy) +
             d10 * fx * (1 - fy) +
             d01 * (1 - fx) * fy +
             d11 * fx * fy;
}

OpticalFlow::OpticalFlow() {
    config_ = OpticalFlowConfig();
}

void OpticalFlow::configure(const OpticalFlowConfig& config) {
    config_ = config;
}

void OpticalFlow::to_grayscale(const uint8_t* rgba, uint8_t* gray, int width, int height) {
    const int count = width * height;
    
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(count > 10000)
    #endif
    for (int i = 0; i < count; ++i) {
        // ITU-R BT.601 luma
        gray[i] = static_cast<uint8_t>(
            0.299f * rgba[i * 4 + 0] +
            0.587f * rgba[i * 4 + 1] +
            0.114f * rgba[i * 4 + 2]
        );
    }
}

int OpticalFlow::compute_sad(const uint8_t* prev, const uint8_t* curr, int width,
                              int bx, int by, int dx, int dy) {
    const int bs = config_.block_size;
    int sad = 0;
    
    for (int y = 0; y < bs; ++y) {
        for (int x = 0; x < bs; ++x) {
            int px = bx + x;
            int py = by + y;
            int sx = px + dx;
            int sy = py + dy;
            
            // Bounds checking (treat out of bounds as high cost)
            if (sx < 0 || sx >= width || sy < 0) {
                sad += 255;
                continue;
            }
            
            int prev_idx = py * width + px;
            int curr_idx = sy * width + sx;
            
            sad += std::abs((int)prev[prev_idx] - (int)curr[curr_idx]);
        }
    }
    
    return sad;
}

void OpticalFlow::search_block(const uint8_t* prev, const uint8_t* curr, 
                                int width, int height,
                                int bx, int by, float& out_dx, float& out_dy) {
    const int sr = config_.search_range;
    int best_sad = INT32_MAX;
    int best_dx = 0;
    int best_dy = 0;
    
    // Full search in search range
    for (int dy = -sr; dy <= sr; ++dy) {
        for (int dx = -sr; dx <= sr; ++dx) {
            int sad = compute_sad(prev, curr, width, bx, by, dx, dy);
            if (sad < best_sad) {
                best_sad = sad;
                best_dx = dx;
                best_dy = dy;
            }
        }
    }
    
    out_dx = static_cast<float>(best_dx);
    out_dy = static_cast<float>(best_dy);
}

void OpticalFlow::interpolate_flow(int width, int height) {
    const int bs = config_.block_size;
    
    flow_.resize(width, height);
    
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(height > 64)
    #endif
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            // Find which block this pixel is in
            float block_x = static_cast<float>(x) / bs;
            float block_y = static_cast<float>(y) / bs;
            
            // Bilinear interpolation from block centers
            int bx0 = static_cast<int>(block_x);
            int by0 = static_cast<int>(block_y);
            int bx1 = std::min(bx0 + 1, blocks_x_ - 1);
            int by1 = std::min(by0 + 1, blocks_y_ - 1);
            
            // Clamp
            bx0 = std::max(0, std::min(bx0, blocks_x_ - 1));
            by0 = std::max(0, std::min(by0, blocks_y_ - 1));
            
            float fx = block_x - bx0;
            float fy = block_y - by0;
            
            // Get block flow values
            float dx00 = block_dx_[by0 * blocks_x_ + bx0];
            float dx10 = block_dx_[by0 * blocks_x_ + bx1];
            float dx01 = block_dx_[by1 * blocks_x_ + bx0];
            float dx11 = block_dx_[by1 * blocks_x_ + bx1];
            
            float dy00 = block_dy_[by0 * blocks_x_ + bx0];
            float dy10 = block_dy_[by0 * blocks_x_ + bx1];
            float dy01 = block_dy_[by1 * blocks_x_ + bx0];
            float dy11 = block_dy_[by1 * blocks_x_ + bx1];
            
            // Bilinear interpolation
            float pixel_dx = dx00 * (1 - fx) * (1 - fy) +
                            dx10 * fx * (1 - fy) +
                            dx01 * (1 - fx) * fy +
                            dx11 * fx * fy;
            
            float pixel_dy = dy00 * (1 - fx) * (1 - fy) +
                            dy10 * fx * (1 - fy) +
                            dy01 * (1 - fx) * fy +
                            dy11 * fx * fy;
            
            // Temporal smoothing
            if (config_.smoothing > 0.0f && ready_) {
                int idx = y * width + x;
                pixel_dx = pixel_dx * (1 - config_.smoothing) + flow_prev_.dx[idx] * config_.smoothing;
                pixel_dy = pixel_dy * (1 - config_.smoothing) + flow_prev_.dy[idx] * config_.smoothing;
            }
            
            int idx = y * width + x;
            flow_.dx[idx] = pixel_dx;
            flow_.dy[idx] = pixel_dy;
        }
    }
}

bool OpticalFlow::compute(const uint8_t* prev, const uint8_t* curr, int width, int height) {
    if (!prev || !curr || width <= 0 || height <= 0) {
        return false;
    }
    
    const int bs = config_.block_size;
    blocks_x_ = (width + bs - 1) / bs;
    blocks_y_ = (height + bs - 1) / bs;
    
    block_dx_.resize(blocks_x_ * blocks_y_);
    block_dy_.resize(blocks_x_ * blocks_y_);
    
    // Save previous flow for temporal smoothing
    if (ready_) {
        flow_prev_ = flow_;
    }
    
    // Compute block-level flow
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic) collapse(2) if(blocks_y_ > 8)
    #endif
    for (int by = 0; by < blocks_y_; ++by) {
        for (int bx = 0; bx < blocks_x_; ++bx) {
            float dx, dy;
            search_block(prev, curr, width, height, bx * bs, by * bs, dx, dy);
            block_dx_[by * blocks_x_ + bx] = dx;
            block_dy_[by * blocks_x_ + bx] = dy;
        }
    }
    
    // Interpolate to pixel level
    interpolate_flow(width, height);
    
    ready_ = true;
    return true;
}

bool OpticalFlow::compute_rgba(const uint8_t* prev, const uint8_t* curr, int width, int height) {
    if (!prev || !curr || width <= 0 || height <= 0) {
        return false;
    }
    
    // Resize grayscale buffers
    size_t size = width * height;
    if (gray_prev_.size() != size) {
        gray_prev_.resize(size);
        gray_curr_.resize(size);
    }
    
    // Convert to grayscale
    to_grayscale(prev, gray_prev_.data(), width, height);
    to_grayscale(curr, gray_curr_.data(), width, height);
    
    return compute(gray_prev_.data(), gray_curr_.data(), width, height);
}

bool OpticalFlow::warp(const uint8_t* src, uint8_t* dst, int width, int height, float scale) {
    if (!ready_ || !src || !dst || width != flow_.width || height != flow_.height) {
        // Fallback: just copy
        if (src && dst && width > 0 && height > 0) {
            std::memcpy(dst, src, width * height * 4);
        }
        return false;
    }
    
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(height > 64)
    #endif
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = y * width + x;
            
            // Backward warp: where does this destination pixel come from?
            float src_x = x + flow_.dx[idx] * scale;
            float src_y = y + flow_.dy[idx] * scale;
            
            // Bilinear sampling from source
            int x0 = static_cast<int>(std::floor(src_x));
            int y0 = static_cast<int>(std::floor(src_y));
            int x1 = x0 + 1;
            int y1 = y0 + 1;
            
            // Clamp to image bounds
            x0 = std::max(0, std::min(x0, width - 1));
            x1 = std::max(0, std::min(x1, width - 1));
            y0 = std::max(0, std::min(y0, height - 1));
            y1 = std::max(0, std::min(y1, height - 1));
            
            float fx = src_x - std::floor(src_x);
            float fy = src_y - std::floor(src_y);
            
            // Sample all 4 channels (RGBA)
            for (int c = 0; c < 4; ++c) {
                float v00 = src[(y0 * width + x0) * 4 + c];
                float v10 = src[(y0 * width + x1) * 4 + c];
                float v01 = src[(y1 * width + x0) * 4 + c];
                float v11 = src[(y1 * width + x1) * 4 + c];
                
                float val = v00 * (1 - fx) * (1 - fy) +
                           v10 * fx * (1 - fy) +
                           v01 * (1 - fx) * fy +
                           v11 * fx * fy;
                
                dst[idx * 4 + c] = static_cast<uint8_t>(std::clamp(val, 0.0f, 255.0f));
            }
        }
    }
    
    return true;
}

float OpticalFlow::get_average_motion() const {
    if (!ready_ || flow_.dx.empty()) {
        return 0.0f;
    }
    
    double total = 0.0;
    const size_t count = flow_.dx.size();
    
    for (size_t i = 0; i < count; ++i) {
        float mag = std::sqrt(flow_.dx[i] * flow_.dx[i] + flow_.dy[i] * flow_.dy[i]);
        total += mag;
    }
    
    return static_cast<float>(total / count);
}

// KeyframeManager implementation

KeyframeManager::KeyframeManager() {
    config_ = Config();
}

bool KeyframeManager::is_keyframe(int frame_number, float motion_magnitude, float scene_mse) {
    if (!config_.enabled) {
        return true;  // Process every frame
    }
    
    // Scene change detection
    if (scene_mse > config_.scene_change_threshold) {
        last_keyframe_number_ = frame_number;
        return true;
    }
    
    // Interval-based keyframes
    if (last_keyframe_number_ < 0 || 
        (frame_number - last_keyframe_number_) >= config_.keyframe_interval) {
        last_keyframe_number_ = frame_number;
        return true;
    }
    
    return false;
}

void KeyframeManager::store_keyframe(const uint8_t* rgba, int width, int height) {
    size_t size = static_cast<size_t>(width) * height * 4;
    if (keyframe_.size() != size) {
        keyframe_.resize(size);
    }
    std::memcpy(keyframe_.data(), rgba, size);
}

} // namespace lll
