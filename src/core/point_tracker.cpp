/**
 * Point Tracker - Implementation
 * 
 * Sparse feature detection and tracking.
 */

#include "point_tracker.hpp"
#include <cmath>
#include <algorithm>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace lll {

PointTracker::PointTracker() {
    config_ = PointTrackerConfig();
    points_.reserve(config_.max_points);
}

void PointTracker::configure(const PointTrackerConfig& cfg) {
    config_ = cfg;
    points_.reserve(config_.max_points);
    for (auto& pt : points_) {
        pt.max_history = config_.history_length;
    }
}

void PointTracker::to_grayscale(const uint8_t* rgba, uint8_t* gray, int width, int height) {
    const int count = width * height;
    
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(count > 10000)
    #endif
    for (int i = 0; i < count; ++i) {
        gray[i] = static_cast<uint8_t>(
            0.299f * rgba[i * 4 + 0] +
            0.587f * rgba[i * 4 + 1] +
            0.114f * rgba[i * 4 + 2]
        );
    }
}

void PointTracker::detect_corners(const uint8_t* gray, int width, int height) {
    // Shi-Tomasi corner detection (approximation)
    // Compute gradient magnitude as simple corner response
    
    corner_response_.resize(width * height);
    std::fill(corner_response_.begin(), corner_response_.end(), 0.0f);
    
    const int bs = config_.detection_block_size;
    const int hbs = bs / 2;
    
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(height > 64)
    #endif
    for (int y = hbs; y < height - hbs; ++y) {
        for (int x = hbs; x < width - hbs; ++x) {
            // Compute structure tensor elements
            float Ixx = 0, Iyy = 0, Ixy = 0;
            
            for (int dy = -hbs; dy <= hbs; ++dy) {
                for (int dx = -hbs; dx <= hbs; ++dx) {
                    int px = x + dx;
                    int py = y + dy;
                    
                    // Sobel-like gradients
                    float gx = 0, gy = 0;
                    if (px > 0 && px < width - 1) {
                        gx = static_cast<float>(gray[py * width + px + 1]) -
                             static_cast<float>(gray[py * width + px - 1]);
                    }
                    if (py > 0 && py < height - 1) {
                        gy = static_cast<float>(gray[(py + 1) * width + px]) -
                             static_cast<float>(gray[(py - 1) * width + px]);
                    }
                    
                    Ixx += gx * gx;
                    Iyy += gy * gy;
                    Ixy += gx * gy;
                }
            }
            
            // Shi-Tomasi: min eigenvalue
            float trace = Ixx + Iyy;
            float det = Ixx * Iyy - Ixy * Ixy;
            float discriminant = trace * trace - 4.0f * det;
            
            float response = 0.0f;
            if (discriminant >= 0) {
                float sqrtD = std::sqrt(discriminant);
                float lambda1 = (trace + sqrtD) * 0.5f;
                float lambda2 = (trace - sqrtD) * 0.5f;
                response = std::min(lambda1, lambda2);
            }
            
            corner_response_[y * width + x] = response;
        }
    }
}

float PointTracker::compute_ssd(const uint8_t* img, int width, int height,
                                 int cx, int cy, int tx, int ty) {
    const int ts = config_.template_size;
    const int hts = ts / 2;
    
    float ssd = 0.0f;
    int count = 0;
    
    for (int dy = -hts; dy <= hts; ++dy) {
        for (int dx = -hts; dx <= hts; ++dx) {
            int cx_p = cx + dx;
            int cy_p = cy + dy;
            int tx_p = tx + dx;
            int ty_p = ty + dy;
            
            // Bounds check
            if (cx_p < 0 || cx_p >= width || cy_p < 0 || cy_p >= height ||
                tx_p < 0 || tx_p >= width || ty_p < 0 || ty_p >= height) {
                continue;
            }
            
            float diff = static_cast<float>(img[cy_p * width + cx_p]) -
                        static_cast<float>(prev_gray_[ty_p * width + tx_p]);
            ssd += diff * diff;
            count++;
        }
    }
    
    return count > 0 ? ssd / count : 1e9f;
}

void PointTracker::track_points(const uint8_t* prev, const uint8_t* curr, 
                                 int width, int height) {
    const int sr = config_.search_radius;
    
    for (auto& pt : points_) {
        if (!pt.valid) continue;
        
        // Current position in pixels
        int px = static_cast<int>(pt.x * width);
        int py = static_cast<int>(pt.y * height);
        
        // Search for best match
        float best_ssd = 1e9f;
        int best_dx = 0, best_dy = 0;
        
        for (int dy = -sr; dy <= sr; ++dy) {
            for (int dx = -sr; dx <= sr; ++dx) {
                int nx = px + dx;
                int ny = py + dy;
                
                if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
                
                float ssd = compute_ssd(curr, width, height, nx, ny, px, py);
                if (ssd < best_ssd) {
                    best_ssd = ssd;
                    best_dx = dx;
                    best_dy = dy;
                }
            }
        }
        
        // Update if match is good enough
        if (best_ssd < config_.detection_threshold * 100) {
            // Smooth velocity update
            float new_vx = static_cast<float>(best_dx);
            float new_vy = static_cast<float>(best_dy);
            pt.vx = pt.vx * config_.velocity_smoothing + 
                    new_vx * (1.0f - config_.velocity_smoothing);
            pt.vy = pt.vy * config_.velocity_smoothing + 
                    new_vy * (1.0f - config_.velocity_smoothing);
            
            // Update position
            pt.x = static_cast<float>(px + best_dx) / width;
            pt.y = static_cast<float>(py + best_dy) / height;
            pt.score = 1.0f / (1.0f + best_ssd * 0.001f);
            pt.age++;
            pt.add_to_history();
        } else {
            // Lost tracking
            pt.valid = false;
        }
        
        // Bounds check
        if (pt.x < 0 || pt.x >= 1 || pt.y < 0 || pt.y >= 1) {
            pt.valid = false;
        }
    }
}

void PointTracker::add_new_points(int width, int height) {
    // Count valid points
    int valid_count = get_valid_count();
    if (valid_count >= config_.max_points) return;
    
    // Find corners with non-maximum suppression
    std::vector<std::pair<float, int>> candidates;
    
    const int margin = config_.template_size;
    for (int y = margin; y < height - margin; ++y) {
        for (int x = margin; x < width - margin; ++x) {
            float response = corner_response_[y * width + x];
            if (response > config_.detection_threshold) {
                // Check if too close to existing points
                float nx = static_cast<float>(x) / width;
                float ny = static_cast<float>(y) / height;
                
                bool too_close = false;
                for (const auto& pt : points_) {
                    if (!pt.valid) continue;
                    float dx = (pt.x - nx) * width;
                    float dy = (pt.y - ny) * height;
                    if (dx * dx + dy * dy < config_.min_distance * config_.min_distance) {
                        too_close = true;
                        break;
                    }
                }
                
                if (!too_close) {
                    candidates.push_back({response, y * width + x});
                }
            }
        }
    }
    
    // Sort by response (highest first)
    std::sort(candidates.begin(), candidates.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // Add new points
    for (const auto& cand : candidates) {
        if (valid_count >= config_.max_points) break;
        
        int idx = cand.second;
        int x = idx % width;
        int y = idx / width;
        
        // Find empty slot or add new
        bool added = false;
        for (auto& pt : points_) {
            if (!pt.valid) {
                pt.x = static_cast<float>(x) / width;
                pt.y = static_cast<float>(y) / height;
                pt.vx = 0;
                pt.vy = 0;
                pt.age = 0;
                pt.score = 1.0f;
                pt.valid = true;
                pt.history_x.clear();
                pt.history_y.clear();
                pt.max_history = config_.history_length;
                pt.add_to_history();
                added = true;
                valid_count++;
                break;
            }
        }
        
        if (!added && static_cast<int>(points_.size()) < config_.max_points) {
            TrackedPoint pt;
            pt.x = static_cast<float>(x) / width;
            pt.y = static_cast<float>(y) / height;
            pt.vx = 0;
            pt.vy = 0;
            pt.age = 0;
            pt.score = 1.0f;
            pt.valid = true;
            pt.max_history = config_.history_length;
            pt.add_to_history();
            points_.push_back(pt);
            valid_count++;
        }
    }
}

void PointTracker::prune_points() {
    // Remove very old points with low scores
    for (auto& pt : points_) {
        if (pt.valid && pt.age > 100 && pt.score < 0.3f) {
            pt.valid = false;
        }
    }
}

bool PointTracker::update(const uint8_t* rgba, int width, int height) {
    if (!rgba || width <= 0 || height <= 0) return false;
    
    // Resize buffers
    size_t size = static_cast<size_t>(width) * height;
    if (curr_gray_.size() != size) {
        prev_gray_.resize(size);
        curr_gray_.resize(size);
    }
    
    // Convert to grayscale
    to_grayscale(rgba, curr_gray_.data(), width, height);
    
    if (initialized_ && last_width_ == width && last_height_ == height) {
        // Track existing points
        track_points(prev_gray_.data(), curr_gray_.data(), width, height);
        
        // Prune lost points
        prune_points();
    }
    
    // Detect corners
    detect_corners(curr_gray_.data(), width, height);
    
    // Add new points
    add_new_points(width, height);
    
    // Swap buffers
    std::swap(prev_gray_, curr_gray_);
    last_width_ = width;
    last_height_ = height;
    initialized_ = true;
    
    return true;
}

int PointTracker::get_valid_count() const {
    int count = 0;
    for (const auto& pt : points_) {
        if (pt.valid) count++;
    }
    return count;
}

float PointTracker::get_average_motion() const {
    float total = 0.0f;
    int count = 0;
    for (const auto& pt : points_) {
        if (pt.valid) {
            total += std::sqrt(pt.vx * pt.vx + pt.vy * pt.vy);
            count++;
        }
    }
    return count > 0 ? total / count : 0.0f;
}

void PointTracker::reset() {
    points_.clear();
    prev_gray_.clear();
    curr_gray_.clear();
    corner_response_.clear();
    initialized_ = false;
}

void PointTracker::render_control_image(
    uint8_t* rgb_out, int width, int height,
    bool draw_points, bool draw_trails,
    uint8_t bg_r, uint8_t bg_g, uint8_t bg_b,
    uint8_t fg_r, uint8_t fg_g, uint8_t fg_b) const 
{
    if (!rgb_out || width <= 0 || height <= 0) return;
    
    // Clear to background
    for (int i = 0; i < width * height; ++i) {
        rgb_out[i * 3 + 0] = bg_r;
        rgb_out[i * 3 + 1] = bg_g;
        rgb_out[i * 3 + 2] = bg_b;
    }
    
    for (const auto& pt : points_) {
        if (!pt.valid) continue;
        
        // Draw trails
        if (draw_trails && pt.history_x.size() > 1) {
            for (size_t i = 1; i < pt.history_x.size(); ++i) {
                int x0 = static_cast<int>(pt.history_x[i - 1] * width);
                int y0 = static_cast<int>(pt.history_y[i - 1] * height);
                int x1 = static_cast<int>(pt.history_x[i] * width);
                int y1 = static_cast<int>(pt.history_y[i] * height);
                
                // Bresenham line
                int dx = std::abs(x1 - x0);
                int dy = std::abs(y1 - y0);
                int sx = (x0 < x1) ? 1 : -1;
                int sy = (y0 < y1) ? 1 : -1;
                int err = dx - dy;
                
                // Fade based on age in trail
                float fade = static_cast<float>(i) / pt.history_x.size();
                uint8_t r = static_cast<uint8_t>(fg_r * fade);
                uint8_t g = static_cast<uint8_t>(fg_g * fade);
                uint8_t b = static_cast<uint8_t>(fg_b * fade);
                
                while (true) {
                    if (x0 >= 0 && x0 < width && y0 >= 0 && y0 < height) {
                        int idx = (y0 * width + x0) * 3;
                        rgb_out[idx + 0] = std::max(rgb_out[idx + 0], r);
                        rgb_out[idx + 1] = std::max(rgb_out[idx + 1], g);
                        rgb_out[idx + 2] = std::max(rgb_out[idx + 2], b);
                    }
                    if (x0 == x1 && y0 == y1) break;
                    int e2 = 2 * err;
                    if (e2 > -dy) { err -= dy; x0 += sx; }
                    if (e2 < dx) { err += dx; y0 += sy; }
                }
            }
        }
        
        // Draw current point
        if (draw_points) {
            int px = static_cast<int>(pt.x * width);
            int py = static_cast<int>(pt.y * height);
            int radius = 3;
            
            for (int dy = -radius; dy <= radius; ++dy) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    if (dx * dx + dy * dy <= radius * radius) {
                        int nx = px + dx;
                        int ny = py + dy;
                        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                            int idx = (ny * width + nx) * 3;
                            rgb_out[idx + 0] = fg_r;
                            rgb_out[idx + 1] = fg_g;
                            rgb_out[idx + 2] = fg_b;
                        }
                    }
                }
            }
        }
    }
}

} // namespace lll
