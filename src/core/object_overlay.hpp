#pragma once
#include "object_tracks.hpp"
#include <cstdint>
#include <vector>
namespace sight {
// Premultiplied, transparent RGBA used by Sight's existing OBS overlay shader.
// Shows category + persistent ID, not a substitute for detection metadata.
std::vector<uint8_t> draw_object_overlay(uint32_t width,uint32_t height,
                                        const std::vector<ObjectDetection>& objects);
}
