#include <obs-module.h>
#include <graphics/graphics.h>
#include <util/platform.h>
#include <util/dstr.h>
#include <vector>
#include <algorithm>
#include <mutex>
#include <string>
#include <cmath>

#include "core/latent-engine.hpp"
#include "core/segmentation_processor.hpp"
#include "mediapipe/mp_wrapper.h"
#include "fast-face-detector.hpp"

// Constants
#define S_BLUR_STRENGTH "blur_strength"
#define S_SHOW_MASK "show_mask"
#define S_DETECTION_MODEL "detection_model"
#define S_BG_MODE "bg_mode"
#define S_SEG_MODEL "seg_model"

enum class BgMode {
    Blur = 0,
    RemoveGreen = 1,
    RemoveTransparent = 2,
    Replace = 3
};

enum class SegModel {
    U2Net = 0,
    RMBG = 1,
    InSPyReNet = 2
};

enum class ModelType {
    MediaPipe = 0,
    Fast = 1
};

struct privacy_filter_data {
    obs_source_t *context;
    
    // Properties
    float blur_strength = 50.0f;
    bool show_mask = false;
    ModelType detection_model = ModelType::MediaPipe;
    
    // Detectors
    std::unique_ptr<FastFaceDetector> fast_detector;
    std::unique_ptr<lll::SegmentationProcessor> seg_processor;
    
    // Config
    BgMode bg_mode = BgMode::Blur;
    SegModel seg_model = SegModel::U2Net;
    std::string seg_model_path;

    // State
    uint32_t width = 0;
    uint32_t height = 0;
    
    std::vector<uint8_t> input_rgb_small;
    std::mutex filter_mutex;
    
    // Rendering
    gs_stagesurf_t *stagesurf = nullptr;
    gs_texture_t *out_tex = nullptr;
    std::vector<uint8_t> texture_data;
    std::vector<uint8_t> mask_data;
};

static const char *get_name(void *unused) {
    UNUSED_PARAMETER(unused);
    return "LLL.sight Privacy Filter (MediaPipe)";
}

static void apply_privacy_blur(uint8_t* rgb, int w, int h, const uint8_t* mask, int mw, int mh, float strength) {
    if (strength <= 0.0f) return;
    
    // Fast blur by downscale/upscale (approx)
    int scale = 8;
    int sw = w / scale;
    int sh = h / scale;
    if (sw < 1) sw = 1; 
    if (sh < 1) sh = 1;
    std::vector<uint8_t> small(sw * sh * 3);
    
    // Nearest neighbor downscale for blur approximation
    for(int y=0; y<sh; y++) {
        for(int x=0; x<sw; x++) {
            int sx = x * scale;
            int sy = y * scale;
            if (sx >= w) sx = w - 1;
            if (sy >= h) sy = h - 1;
            
            int sidx = (sy * w + sx) * 3;
            int didx = (y * sw + x) * 3;
            
            small[didx]   = rgb[sidx];
            small[didx+1] = rgb[sidx+1];
            small[didx+2] = rgb[sidx+2];
        }
    }
    
    // Apply blur where mask indicates
    for(int y=0; y<h; y++) {
        for(int x=0; x<w; x++) {
            // Sample mask (nearest)
            int mx = (int)((float)x / w * mw);
            int my = (int)((float)y / h * mh);
            mx = std::clamp(mx, 0, mw-1);
            my = std::clamp(my, 0, mh-1);
            
            uint8_t m_val = mask[my * mw + mx];
            
            // m_val: 255 = primary face (keep sharp), < 255 = background/bystander
            if (m_val < 255) { 
                float alpha = (255.0f - m_val) / 255.0f * (strength / 100.0f);
                alpha = std::clamp(alpha, 0.0f, 1.0f);
                
                if (alpha > 0.0f) {
                   // Sample blurred version (nearest from small buffer)
                   int sx = (int)((float)x / w * sw);
                   int sy = (int)((float)y / h * sh);
                   sx = std::clamp(sx, 0, sw-1);
                   sy = std::clamp(sy, 0, sh-1);
                   int sidx = (sy * sw + sx) * 3;
                   
                   int didx = (y * w + x) * 3;
                   
                   rgb[didx]   = (uint8_t)((1.0f - alpha) * rgb[didx] + alpha * small[sidx]);
                   rgb[didx+1] = (uint8_t)((1.0f - alpha) * rgb[didx+1] + alpha * small[sidx+1]);
                   rgb[didx+2] = (uint8_t)((1.0f - alpha) * rgb[didx+2] + alpha * small[sidx+2]);
                }
            }
        }
    }
}

static void video_render(void *data, gs_effect_t *effect) {
    auto *filter = static_cast<privacy_filter_data*>(data);
    
    obs_source_t *target = obs_filter_get_target(filter->context);
    if (!target) {
        obs_source_skip_video_filter(filter->context);
        return;
    }

    uint32_t width = obs_source_get_base_width(target);
    uint32_t height = obs_source_get_base_height(target);
    
    if (width == 0 || height == 0) {
         obs_source_skip_video_filter(filter->context);
         return;
    }
    
    // Default pass-through if not in AI Segmentation mode
    if ((int)filter->detection_model != 2) {
        obs_source_skip_video_filter(filter->context);
        return;
    }

    // --- AI Segmentation Logic ---
    
    // 1. Prepare/Resize Staging Surface
    if (filter->stagesurf) {
        if (gs_stagesurface_get_width(filter->stagesurf) != width || 
            gs_stagesurface_get_height(filter->stagesurf) != height) {
            gs_stagesurface_destroy(filter->stagesurf);
            filter->stagesurf = nullptr;
        }
    }
    if (!filter->stagesurf) {
        filter->stagesurf = gs_stagesurface_create(width, height, GS_RGBA);
    }
    
    // 2. Capture Frame
    static gs_texrender_t *texrender = nullptr;
    if (!texrender) texrender = gs_texrender_create(GS_RGBA, GS_ZS_NONE);
    
    gs_texrender_reset(texrender);
    if (gs_texrender_begin(texrender, width, height)) {
        vec4 clear_color;
        vec4_zero(&clear_color);
        gs_clear(GS_CLEAR_COLOR, &clear_color, 0.0f, 0);
        gs_ortho(0.0f, (float)width, 0.0f, (float)height, -100.0f, 100.0f);
        
        obs_source_video_render(target);
        gs_texrender_end(texrender);
        
        gs_texture_t *tex = gs_texrender_get_texture(texrender);
        
        // 3. Download from GPU
        if (filter->stagesurf && tex) {
            gs_stage_texture(filter->stagesurf, tex);
            
            uint8_t *ptr;
            uint32_t linesize;
            if (gs_stagesurface_map(filter->stagesurf, &ptr, &linesize)) {
                // 4. Process
                if (filter->texture_data.size() != width * height * 4) {
                     filter->texture_data.resize(width * height * 4);
                     filter->mask_data.resize(width * height);
                }
                
                // Copy to buffer
                for(uint32_t y=0; y<height; ++y) {
                    const uint8_t* row = ptr + y * linesize;
                    uint8_t* dst_row = filter->texture_data.data() + y * width * 4;
                    memcpy(dst_row, row, width * 4); 
                }
                
                gs_stagesurface_unmap(filter->stagesurf);
                
                // Inference (expects dense RGB)
                static std::vector<uint8_t> rgb_temp;
                if (rgb_temp.size() != width * height * 3) rgb_temp.resize(width * height * 3);
                
                for(uint32_t i=0; i<width*height; ++i) {
                    rgb_temp[i*3+0] = filter->texture_data[i*4+0];
                    rgb_temp[i*3+1] = filter->texture_data[i*4+1];
                    rgb_temp[i*3+2] = filter->texture_data[i*4+2];
                }
                
                if (filter->seg_processor && filter->seg_processor->is_ready()) {
                    filter->seg_processor->process(rgb_temp.data(), width, height, filter->mask_data.data());
                }
                
                // 5. Apply Effect to `texture_data` (RGBA)
                bool remove = (filter->bg_mode == BgMode::RemoveTransparent || filter->bg_mode == BgMode::RemoveGreen);
                bool green = (filter->bg_mode == BgMode::RemoveGreen);
                
                if (remove) {
                    for(uint32_t i=0; i<width*height; ++i) {
                        uint8_t alpha = filter->mask_data[i];
                        if (green) {
                            if (alpha < 128) {
                                filter->texture_data[i*4+0] = 0;
                                filter->texture_data[i*4+1] = 255; 
                                filter->texture_data[i*4+2] = 0;
                                filter->texture_data[i*4+3] = 255;
                            }
                        } else {
                            filter->texture_data[i*4+3] = alpha;
                        }
                    }
                }
                
                // 6. Upload Result
                if (!filter->out_tex || gs_texture_get_width(filter->out_tex) != width || gs_texture_get_height(filter->out_tex) != height) {
                     if (filter->out_tex) gs_texture_destroy(filter->out_tex);
                     filter->out_tex = gs_texture_create(width, height, GS_RGBA, 1, nullptr, GS_DYNAMIC);
                }
                
                if (filter->out_tex) {
                    gs_texture_set_image(filter->out_tex, filter->texture_data.data(), width * 4, false);
                }
            }
        }
    }
    
    // 7. Draw Result
    if (filter->out_tex) {
        gs_effect_t *draw_effect = effect ? effect : obs_get_base_effect(OBS_EFFECT_DEFAULT);
        if (!draw_effect) {
            obs_source_skip_video_filter(filter->context);
            return;
        }
        gs_effect_set_texture(gs_effect_get_param_by_name(draw_effect, "image"),
                              filter->out_tex);
        while (gs_effect_loop(draw_effect, "Draw")) {
            gs_draw_sprite(filter->out_tex, 0, width, height);
        }
    } else {
        obs_source_skip_video_filter(filter->context);
    }
}

static obs_properties_t *get_properties(void *data) {
    UNUSED_PARAMETER(data);
    obs_properties_t *props = obs_properties_create();
    
    obs_properties_add_float_slider(props, S_BLUR_STRENGTH, "Blur Strength", 0.0, 100.0, 1.0);
    obs_properties_add_bool(props, S_SHOW_MASK, "Visualize Mask (Debug)");
    
    obs_property_t *p_model = obs_properties_add_list(props, S_DETECTION_MODEL, "Detection Model", OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
    obs_property_list_add_int(p_model, "MediaPipe (Detailed - Semantic Mask)", (int)ModelType::MediaPipe);
    obs_property_list_add_int(p_model, "Fast (libfacedetection - Bounding Box)", (int)ModelType::Fast);
    obs_property_list_add_int(p_model, "AI Segmentation (Background Removal)", 2); // New mode

    obs_property_t *p_bg = obs_properties_add_list(props, S_BG_MODE, "Background Mode", OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
    obs_property_list_add_int(p_bg, "Blur", (int)BgMode::Blur);
    obs_property_list_add_int(p_bg, "Remove (Green Screen)", (int)BgMode::RemoveGreen);
    obs_property_list_add_int(p_bg, "Remove (Transparent)", (int)BgMode::RemoveTransparent);
    obs_property_list_add_int(p_bg, "Replace (Not Implemented)", (int)BgMode::Replace);

    obs_property_t *p_seg = obs_properties_add_list(props, S_SEG_MODEL, "Segmentation Model", OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
    obs_property_list_add_int(p_seg, "U-2-Net (Portrait - Fast)", (int)SegModel::U2Net);
    obs_property_list_add_int(p_seg, "RMBG-1.4 (Background Removal - SOTA)", (int)SegModel::RMBG);
    obs_property_list_add_int(p_seg, "InSPyReNet (Gold Standard - Heavy)", (int)SegModel::InSPyReNet);

    obs_property_set_modified_callback(p_model, [](obs_properties_t *props, obs_property_t *p, obs_data_t *settings) {
        int mode = (int)obs_data_get_int(settings, S_DETECTION_MODEL);
        bool is_seg = (mode == 2);
        obs_property_set_visible(obs_properties_get(props, S_BG_MODE), is_seg);
        obs_property_set_visible(obs_properties_get(props, S_SEG_MODEL), is_seg);
        return true;
    });

    return props;
}

static void update(void *data, obs_data_t *settings) {
    auto *filter = static_cast<privacy_filter_data*>(data);
    filter->blur_strength = (float)obs_data_get_double(settings, S_BLUR_STRENGTH);
    filter->show_mask = obs_data_get_bool(settings, S_SHOW_MASK);
    filter->detection_model = (ModelType)obs_data_get_int(settings, S_DETECTION_MODEL);

    if (filter->detection_model == ModelType::Fast) {
        if (!filter->fast_detector) {
             filter->fast_detector = std::make_unique<FastFaceDetector>();
        }
    } else if ((int)filter->detection_model == 2) {
        // AI Segmentation Mode
        filter->bg_mode = (BgMode)obs_data_get_int(settings, S_BG_MODE);
        SegModel target_model = (SegModel)obs_data_get_int(settings, S_SEG_MODEL);
        
        if (!filter->seg_processor || filter->seg_model != target_model) {
            filter->seg_processor = std::make_unique<lll::SegmentationProcessor>();
            filter->seg_model = target_model;
            
            const char* filename = "models/u2net.onnx";
            lll::SegModelType type = lll::SegModelType::U2Net;

            if (target_model == SegModel::RMBG) {
                filename = "models/rmbg.onnx";
                type = lll::SegModelType::RMBG;
            } else if (target_model == SegModel::InSPyReNet) {
                filename = "models/inspyrenet.onnx";
                type = lll::SegModelType::InSPyReNet;
            }
            
            char* file = obs_module_file(filename);
            if (file) {
                 filter->seg_processor->load(file, type, true);
                 bfree(file);
            }
        }
    } else {
        // Ensure MP is initialized
        if (!lll::LatentEngine::get().get_mp_runner()) {
             lll::LatentEngine::get().init_mediapipe(true); 
        }
    }
}

static void *create(obs_data_t *settings, obs_source_t *context) {
    auto *filter = new privacy_filter_data();
    filter->context = context;
    obs_source_update(context, settings);
    return filter;
}

static void destroy(void *data) {
    auto *filter = static_cast<privacy_filter_data*>(data);
    if (filter->stagesurf) gs_stagesurface_destroy(filter->stagesurf);
    if (filter->out_tex) gs_texture_destroy(filter->out_tex);
    delete filter;
}

struct obs_source_info privacy_filter_info = {
    .id = "privacy_filter",
    .type = OBS_SOURCE_TYPE_FILTER,
    .output_flags = OBS_SOURCE_VIDEO,
    .get_name = get_name,
    .create = create,
    .destroy = destroy,
    .get_properties = get_properties,
    .update = update,
    .video_render = video_render
};
