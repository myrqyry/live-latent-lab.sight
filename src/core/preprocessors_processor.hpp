#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

namespace preprocessors {

enum class Backend { Native, ModelRequired };
struct ModeSpec {
    const char *id;
    const char *label;
    const char *category;
    Backend backend;
};

struct Settings {
    int low_threshold = 64;
    int high_threshold = 128;
    double sigma = 2.0;
    bool invert = false;
    int palette_size = 32;
};

// Stable IDs: mode choice is saved by ID, not by display name or list index.
const std::vector<ModeSpec> &catalog();
const ModeSpec *find_mode(std::string_view id);
bool is_native(std::string_view id);

// Returns false and leaves result untouched on invalid input/unsupported
// models. This is not a silent neural or approximate substitute.
bool process_rgba(const uint8_t *input, int width, int height, int stride,
                  std::string_view mode_id, const Settings &settings,
                  std::vector<uint8_t> &result);

} // namespace preprocessors
