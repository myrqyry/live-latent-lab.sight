#pragma once
#include <entt/entt.hpp>
#include "mediapipe/mp_wrapper.h"
#include <vector>
#include <random>
#include <cmath>

namespace lll {

// ===================================
// Components
// ===================================

struct Transform {
    float x = 0.5f; // Normalized 0-1
    float y = 0.5f;
    float scale = 0.1f;
    float rotation = 0.0f;
    int type = 0; // 0=Circle, 1=Sprite...
};

struct Physics {
    float vx = 0.0f;
    float vy = 0.0f;
    float gravity = 0.0f; 
    float elasticity = 0.9f; 
};

struct FaceAnchor {
    int landmark_index = 0; // e.g. 4 (Tip of nose)
    float offset_x = 0.0f;
    float offset_y = 0.0f;
};

// ===================================
// Scene Manager
// ===================================

class SceneManager {
public:
    entt::registry registry;
    bool initialized = false;

    SceneManager() {
        init_demo();
    }

    void init_demo() {
        if (initialized) return;
        
        // Spawn bouncing balls
        std::mt19937 rng(1234);
        std::uniform_real_distribution<float> dist(0.1f, 0.9f);
        std::uniform_real_distribution<float> vel(-0.005f, 0.005f);

        for(int i=0; i<5; ++i) {
            auto entity = registry.create();
            registry.emplace<Transform>(entity, dist(rng), dist(rng), 0.05f, 0.0f, 0);
            registry.emplace<Physics>(entity, vel(rng), vel(rng), 0.0005f, 0.95f);
        }
        
        // Face Anchor Demo (Nose)
        auto nose = registry.create();
        registry.emplace<Transform>(nose, 0.5f, 0.5f, 0.08f, 0.0f, 1);
        registry.emplace<FaceAnchor>(nose, 4, 0.0f, 0.0f); // Tip of nose
        
        initialized = true;
    }

    void update_physics(float dt_ms) {
        // Physics Step (approximate)
        // dt_ms is usually ~16ms. 
        // We use discrete steps for stability
        float dt = 1.0f; // Multiplier
        
        auto view = registry.view<Transform, Physics>();
        view.each([dt](auto entity, Transform& t, Physics& p) {
            p.vy += p.gravity * dt;
            t.x += p.vx * dt;
            t.y += p.vy * dt;

            // Bounds collision (0-1) screen space
            if (t.y > 1.0f) { t.y = 1.0f; p.vy *= -p.elasticity; }
            if (t.y < 0.0f) { t.y = 0.0f; p.vy *= -p.elasticity; }
            if (t.x > 1.0f) { t.x = 1.0f; p.vx *= -p.elasticity; }
            if (t.x < 0.0f) { t.x = 0.0f; p.vx *= -p.elasticity; }
        });
    }

    void sync_mediapipe(mp_runner_t *runner) {
        if (!runner) return;
        
        // Get landmarks from MP wrapper
        mp_landmark_list_t landmarks = {0};
        if (mp_get_landmarks(runner, &landmarks) && landmarks.count > 0) {
            
            auto view = registry.view<Transform, FaceAnchor>();
            view.each([&landmarks](auto entity, Transform& t, const FaceAnchor& anchor) {
                if (anchor.landmark_index < landmarks.count) {
                    float lx = landmarks.landmarks[anchor.landmark_index].x;
                    float ly = landmarks.landmarks[anchor.landmark_index].y;
                    
                    // Simple smoothing could go here
                    t.x = lx + anchor.offset_x;
                    t.y = ly + anchor.offset_y;
                }
            });
            
            // Interaction: Physics objects bounce off Face objects?
            // Advanced: Iterate Physics objects, check distance to Face Anchor objects
            auto ph_view = registry.view<Transform, Physics>();
            auto fa_view = registry.view<Transform, FaceAnchor>();
            
            ph_view.each([&fa_view](auto p_ent, Transform& pt, Physics& pp) {
                fa_view.each([&pt, &pp](auto f_ent, Transform& ft, FaceAnchor& fa) {
                     float dx = pt.x - ft.x;
                     float dy = pt.y - ft.y;
                     float dist2 = dx*dx + dy*dy;
                     float min_dist = (pt.scale + ft.scale) * 0.5f; // Collide radii
                     
                     if (dist2 < min_dist*min_dist && dist2 > 0.0001f) {
                         // Simple circle collision response
                         float dist = std::sqrt(dist2);
                         float nx = dx / dist;
                         float ny = dy / dist;
                         
                         // Push out
                         float overlap = min_dist - dist;
                         pt.x += nx * overlap;
                         pt.y += ny * overlap;
                         
                         // Bounce velocity
                         // v' = v - 2 * (v . n) * n
                         float dot = pp.vx * nx + pp.vy * ny;
                         pp.vx -= 2 * dot * nx;
                         pp.vy -= 2 * dot * ny;
                         
                         // Add energy (kick)
                         pp.vx += nx * 0.01f;
                         pp.vy += ny * 0.01f;
                     }
                });
            });
        }
    }
    
    // Serialization for Shader
    // returns a flat vector. Format: [x, y, scale, type,  x, y, scale, type, ...]
    // Max entities logic handled by caller or fixed size
    void get_render_data(std::vector<float>& data, int max_entities) {
        data.clear();
        data.reserve(max_entities * 4);
        
        int count = 0;
        auto view = registry.view<Transform>();
        for(auto entity : view) {
            if (count >= max_entities) break;
            const auto& t = view.get<Transform>(entity);
            data.push_back(t.x);
            data.push_back(t.y);
            data.push_back(t.scale);
            data.push_back((float)t.type);
            count++;
        }
        
        // Pad with zeros
        while(count < max_entities) {
             data.push_back(-1.0f); // Hidden
             data.push_back(-1.0f);
             data.push_back(0.0f);
             data.push_back(0.0f);
             count++;
        }
    }
};

} // namespace lll
