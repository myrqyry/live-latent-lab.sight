#include "core/depth_temporal_policy.hpp"
#include <cstdint>
#include <iostream>
#include <vector>
int main()
{
    if (!lll::depth_capture_due(1,0))
        return 1;
    const auto next = lll::next_depth_capture(10, 4);
    if (next != 14 || lll::depth_capture_due(11,next) ||
        lll::depth_capture_due(13,next) ||
        !lll::depth_capture_due(14,next))
        return 1;
    std::vector<uint8_t> fresh{100,255,0}, prior{200,0,255};
    lll::blend_depth_result(fresh, prior, 75);
    if (fresh != std::vector<uint8_t>{175,64,191})
        return 1;
    std::vector<uint8_t> passthrough{16,32,64};
    lll::blend_depth_result(passthrough, prior, 0);
    if (passthrough != std::vector<uint8_t>{16,32,64})
        return 1;
    std::vector<uint8_t> size_mismatch{12};
    lll::blend_depth_result(size_mismatch, prior, 95);
    if (size_mismatch != std::vector<uint8_t>{12})
        return 1;
    // The optional appearance-aware mode preserves smoothing in unchanged
    // pixels but rejects stale depth wherever captured RGB changed.
    const std::vector<uint8_t> before_rgb{
        20, 20, 20,  20, 20, 20,  20, 20, 20
    };
    const std::vector<uint8_t> after_rgb{
        20, 20, 20,  255, 255, 255,  50, 50, 50
    };
    std::vector<uint8_t> selective{100, 100, 100};
    const std::vector<uint8_t> prior_selective{200, 200, 200};
    lll::blend_depth_result_change_aware(
        selective, prior_selective, after_rgb, before_rgb, 75);
    // Unchanged pixel: normal 75% smoothing. High change: fresh-only.
    // Moderate change: partial smoothing rather than a sharp confidence edge.
    if (selective != std::vector<uint8_t>{175, 100, 137})
        return 1;

    // Disabled smoothing remains exact passthrough.
    std::vector<uint8_t> no_blend{100, 100, 100};
    lll::blend_depth_result_change_aware(
        no_blend, prior_selective, after_rgb, before_rgb, 0);
    if (no_blend != std::vector<uint8_t>{100, 100, 100})
        return 1;

    // Missing/misaligned captured RGB must never invent correspondence.
    std::vector<uint8_t> mismatched{100};
    lll::blend_depth_result_change_aware(
        mismatched, std::vector<uint8_t>{200},
        std::vector<uint8_t>{20, 20}, std::vector<uint8_t>{20, 20}, 75);
    if (mismatched != std::vector<uint8_t>{100})
        return 1;

    // Source resize must invalidate stale depth, but initial observation and
    // transient unknown dimensions must not disrupt the current model.
    lll::DepthSourceGeometry geometry;
    if (geometry.observe(640, 360) || geometry.observe(640, 360) ||
        geometry.observe(0, 0) || !geometry.observe(800, 450) ||
        geometry.observe(800, 450) || !geometry.observe(640, 360))
        return 1;

    // Last successful inference is eligible for a bounded time, even if the
    // producer stops, but missing timestamps / reversed time cannot qualify.
    if (!lll::depth_result_recent(5001, 5000, 5000) ||
        !lll::depth_result_recent(10000, 5000, 5000) ||
        lll::depth_result_recent(10001, 5000, 5000) ||
        lll::depth_result_recent(10, 0, 0) ||
        lll::depth_result_recent(100, 200, 0) ||
        !lll::depth_result_recent(999999, 200, 0))
        return 1;

    std::cout << "PASS: depth cadence, smoothing, geometry and output age\n";
}
