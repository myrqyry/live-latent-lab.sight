#include <obs-module.h>
#include <graphics/graphics.h>
#include <util/platform.h>
#include <vector>
#include <algorithm>
#include <mutex>
#include <cmath>

#include "core/latent-engine.hpp"
#include "core/sam_predictor.hpp"

#define S_POINT_X "point_x"
#define S_POINT_Y "point_y"
#define S_SHOW_MASK "show_mask"

struct segment_filter_data {
    obs_source_t *context;
    
    // Properties
    float point_x = 0.5f;
    float point_y = 0.5f;
    bool show_mask = true;
    
    // Model
    std::unique_ptr<lll::SamPredictor> sam;
    
    // State
    uint32_t width = 0;
    uint32_t height = 0;
    
    // Rendering
    gs_stagesurf_t *stagesurf = nullptr;
    gs_texture_t *out_tex = nullptr;
    
    std::vector<uint8_t> texture_data; // RGBA
    std::vector<uint8_t> mask_data;    // 8-bit Alpha
    std::vector<uint8_t> rgb_temp;     // Packed RGB
};

static const char *get_name(void *unused) {
    UNUSED_PARAMETER(unused);
    return "Interactive Segment Filter (MobileSAM)";
}

static void video_render(void *data, gs_effect_t *effect) {
    auto *filter = static_cast<segment_filter_data*>(data);
    
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
    
    // Initialize Model if needed
    if (!filter->sam) {
         filter->sam = std::make_unique<lll::SamPredictor>();
         char* enc = obs_module_file("models/mobile_sam_encoder.onnx");
         char* dec = obs_module_file("models/mobile_sam_decoder.onnx");
         if (enc && dec) {
             filter->sam->load(enc, dec, true);
         }
         if (enc) bfree(enc);
         if (dec) bfree(dec);
    }
    
    if (!filter->sam->is_ready()) {
         obs_source_skip_video_filter(filter->context);
         return;
    }

    // 1. Prepare Staging Surface
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
                // Update buffers
                if (filter->texture_data.size() != width * height * 4) {
                     filter->texture_data.resize(width * height * 4);
                     filter->mask_data.resize(width * height);
                     filter->rgb_temp.resize(width * height * 3);
                }
                
                // Copy RGBA from mapped pointer
                for(uint32_t y=0; y<height; ++y) {
                    const uint8_t* row = ptr + y * linesize;
                    uint8_t* dst_row = filter->texture_data.data() + y * width * 4;
                    memcpy(dst_row, row, width * 4);
                }
                
                gs_stagesurface_unmap(filter->stagesurf);
                
                // Convert to RGB for inference
                for(uint32_t i=0; i<width*height; ++i) {
                    filter->rgb_temp[i*3+0] = filter->texture_data[i*4+0];
                    filter->rgb_temp[i*3+1] = filter->texture_data[i*4+1];
                    filter->rgb_temp[i*3+2] = filter->texture_data[i*4+2];
                }
                
                // SAM Inference
                // 1. Encode (Every frame for live video? Expensive but necessary)
                if (filter->sam->encode_image(filter->rgb_temp.data(), width, height)) {
                     // 2. Decode with prompt
                     lll::SamPoint pt = { filter->point_x, filter->point_y };
                     filter->sam->predict_mask(pt, filter->mask_data.data(), width, height);
                }
                
                // Apply Overlay locally
                for(uint32_t i=0; i<width*height; ++i) {
                    uint8_t m = filter->mask_data[i];
                    if (filter->show_mask) {
                        // Highlight mask in Red
                        if (m > 128) {
                            filter->texture_data[i*4+0] = 255;
                            filter->texture_data[i*4+1] = (uint8_t)(filter->texture_data[i*4+1] * 0.5f);
                            filter->texture_data[i*4+2] = (uint8_t)(filter->texture_data[i*4+2] * 0.5f);
                        }
                    } else {
                        // Alpha Masking (Keep only object)
                        filter->texture_data[i*4+3] = m;
                    }
                }
                
                // Upload
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
    
    // Draw
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
    
    obs_properties_add_float_slider(props, S_POINT_X, "Target X", 0.0, 1.0, 0.01);
    obs_properties_add_float_slider(props, S_POINT_Y, "Target Y", 0.0, 1.0, 0.01);
    obs_properties_add_bool(props, S_SHOW_MASK, "Show Mask Overlay");

    return props;
}

static void update(void *data, obs_data_t *settings) {
    auto *filter = static_cast<segment_filter_data*>(data);
    filter->point_x = (float)obs_data_get_double(settings, S_POINT_X);
    filter->point_y = (float)obs_data_get_double(settings, S_POINT_Y);
    filter->show_mask = obs_data_get_bool(settings, S_SHOW_MASK);
}

static void *create(obs_data_t *settings, obs_source_t *context) {
    auto *filter = new segment_filter_data();
    filter->context = context;
    obs_source_update(context, settings);
    return filter;
}

static void destroy(void *data) {
    auto *filter = static_cast<segment_filter_data*>(data);
    if (filter->stagesurf) gs_stagesurface_destroy(filter->stagesurf);
    if (filter->out_tex) gs_texture_destroy(filter->out_tex);
    delete filter;
}

struct obs_source_info mediapipe_filter_info = {
    .id = "interactive_segment_filter", // Renamed for clarity internally
    .type = OBS_SOURCE_TYPE_FILTER,
    .output_flags = OBS_SOURCE_VIDEO,
    .get_name = get_name,
    .create = create,
    .destroy = destroy,
    .get_properties = get_properties,
    .update = update,
    .video_render = video_render
};
