#pragma once
#include "tasks_tracker.hpp"
#include <cstdint>
#include <vector>
namespace sight {
// Return premultiplied transparent RGBA; a missing subject yields all zeros.
std::vector<uint8_t> draw_tracking_overlay(
    uint32_t width, uint32_t height, TrackingKind kind,
    const TrackingData &result, bool confidence_alpha=false);

// Holistic is a *single* model result containing three independent modalities.
// Compose the existing per-kind overlays; no additional inference required.
enum class HolisticOverlayMode : uint8_t { All=0, Face=1, Body=2, Hands=3 };
std::vector<uint8_t> draw_holistic_overlay(
    uint32_t width, uint32_t height,
    const TrackingData& face, const TrackingData& body,
    const TrackingData& hands, HolisticOverlayMode mode,
    bool confidence_alpha=false);

}
