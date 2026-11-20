#pragma once
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <vector>

namespace lll {
// Gate new inference inputs by the number of OBS renders, not wall-clock
// spinning on the graphics thread. In-flight inference separately gates.
inline bool depth_capture_due(uint64_t render_index, uint64_t next_allowed)
{
    return render_index >= next_allowed;
}
inline uint64_t next_depth_capture(uint64_t render_index, unsigned cadence)
{
    return render_index + std::max(1u, cadence);
}

// A source's video dimensions identify the spatial coordinate system for
// depth. Initial observation is not a resize; unknown dimensions don't erase
// an established geometry. Caller must invalidate live depth on a real change.
struct DepthSourceGeometry {
    uint32_t width = 0;
    uint32_t height = 0;

    bool observe(uint32_t next_width, uint32_t next_height)
    {
        if (!next_width || !next_height)
            return false;
        const bool changed = width && height &&
            (width != next_width || height != next_height);
        width = next_width;
        height = next_height;
        return changed;
    }
};

// Monotonic milliseconds since process start. A missing success timestamp is
// never valid, even when expiration is disabled. A clock regression fails
// closed rather than accidentally extending the lifetime of stale results.
inline bool depth_result_recent(uint64_t now_ms, uint64_t successful_ms,
                                uint64_t max_age_ms)
{
    if (!successful_ms || now_ms < successful_ms)
        return false;
    return !max_age_ms || now_ms - successful_ms <= max_age_ms;
}

// Old-frame contribution is a percentage. This happens ONLY when new
// inference arrives; frames between updates reuse the existing GPU texture.
inline void blend_depth_result(std::vector<uint8_t> &fresh,
                               const std::vector<uint8_t> &prior,
                               unsigned prior_percent)
{
    if (fresh.size() != prior.size() || prior_percent == 0)
        return;
    prior_percent = std::min(95u, prior_percent);
    for (size_t i = 0; i < fresh.size(); ++i)
        fresh[i] = static_cast<uint8_t>(
            (unsigned(fresh[i]) * (100 - prior_percent) +
             unsigned(prior[i]) * prior_percent + 50u) / 100u);
}

// Optional appearance-change rejection, not optical flow or reprojection.
// Depth remains anchored to the latest real inference; previous depth is
// reused only when the corresponding model-input pixels look sufficiently
// similar. No extra model call or rendering-thread computation is required.
inline void blend_depth_result_change_aware(
    std::vector<uint8_t> &fresh, const std::vector<uint8_t> &prior,
    const std::vector<uint8_t> &current_rgb,
    const std::vector<uint8_t> &previous_rgb, unsigned prior_percent)
{
    if (prior_percent == 0 || fresh.size() != prior.size() ||
        current_rgb.size() != fresh.size() * 3 ||
        previous_rgb.size() != current_rgb.size())
        return;
    prior_percent = std::min(95u, prior_percent);
    constexpr unsigned keep_below = 12;   // Mean 8-bit RGB difference
    constexpr unsigned reject_above = 48;
    for (size_t i = 0; i < fresh.size(); ++i) {
        const size_t rgb = i * 3;
        const unsigned difference = static_cast<unsigned>(
            (std::abs(int(current_rgb[rgb]) - int(previous_rgb[rgb])) +
             std::abs(int(current_rgb[rgb + 1]) - int(previous_rgb[rgb + 1])) +
             std::abs(int(current_rgb[rgb + 2]) - int(previous_rgb[rgb + 2]))) / 3);
        const unsigned weight = difference <= keep_below ? prior_percent
            : difference >= reject_above ? 0u
            : prior_percent * (reject_above - difference) /
                (reject_above - keep_below);
        fresh[i] = static_cast<uint8_t>(
            (unsigned(fresh[i]) * (100u - weight) +
             unsigned(prior[i]) * weight + 50u) / 100u);
    }
}
}
