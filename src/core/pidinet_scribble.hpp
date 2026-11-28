#pragma once
#include <cstdint>
#include <vector>

namespace lll {
// Native PiDiNet directional NMS scribble, operating only on actual
// model-derived grayscale edges. Invalid arguments preserve input.
bool pidinet_scribble_edges(std::vector<uint8_t> &gray,
                           uint32_t width, uint32_t height);
} // namespace lll
