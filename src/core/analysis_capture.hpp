#pragma once
#include <obs.h>
#include <graphics/graphics.h>
// A filter or custom source owns its draw calls; entering a default-effect
// loop around a nested producer suppresses its own analysis capture.
// Ordinary input sources using obs_source_draw still need that loop.
inline void sight_draw_analysis_source(obs_source_t* target) {
    if(obs_source_get_type(target)==OBS_SOURCE_TYPE_FILTER ||
       (obs_source_get_output_flags(target)&OBS_SOURCE_CUSTOM_DRAW)) {
        obs_source_video_render(target);
    } else if(auto* effect=obs_get_base_effect(OBS_EFFECT_DEFAULT)) {
        while(gs_effect_loop(effect,"Draw"))obs_source_video_render(target);
    }
}
