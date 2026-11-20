#pragma once
#include "sight_snapshot_v1.h"
#include "tracking_signal_bus.hpp"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <string>

namespace sight {
namespace detail {
inline sight_snapshot_point_v1 snapshot_point(const TrackedPoint& point)
{
    return {point.x, point.y, point.z, point.confidence};
}
template <size_t N> void copy_name(char (&dest)[N], const std::string& src)
{
    const auto count = std::min(src.size(), N - 1);
    std::memcpy(dest, src.data(), count);
    dest[count] = 0;
}
}

/* Caller supplies version+size. No allocations exposed, no inference.
 * Returns true for a fresh no-detection result (zero point_count). */
inline bool export_tracking_snapshot(const char* channel, unsigned max_age_ms,
                                     sight_snapshot_v1* out)
{
    if (!out || out->abi_version != SIGHT_SNAPSHOT_ABI_V1 ||
        out->struct_size != sizeof(sight_snapshot_v1))
        return false;
    std::memset(out, 0, sizeof(*out));
    out->abi_version = SIGHT_SNAPSHOT_ABI_V1;
    out->struct_size = sizeof(*out);
    if (!channel || !channel[0] || max_age_ms < 1 || max_age_ms > 5000)
        return false;
    TrackingSignal signal;
    if (!TrackingSignalBus::read(channel, std::chrono::milliseconds(max_age_ms), signal))
        return false;

    switch (signal.kind) {
    case TrackingKind::Face: out->kind = SIGHT_SNAPSHOT_FACE; break;
    case TrackingKind::Body: out->kind = SIGHT_SNAPSHOT_BODY; break;
    case TrackingKind::Hand: out->kind = SIGHT_SNAPSHOT_HAND; break;
    case TrackingKind::WholeBody: out->kind = SIGHT_SNAPSHOT_WHOLEBODY; break;
    default: return false;
    }
    out->source_width = signal.source_width;
    out->source_height = signal.source_height;
    out->sequence = signal.sequence;
    out->point_count = static_cast<uint32_t>(std::min(
        signal.data.landmarks.size(), size_t(SIGHT_SNAPSHOT_MAX_POINTS)));
    for (uint32_t i = 0; i < out->point_count; ++i)
        out->points[i] = detail::snapshot_point(signal.data.landmarks[i]);

    if (signal.kind == TrackingKind::Hand) {
        out->hands_count = static_cast<uint32_t>(std::min(
            signal.data.hands.size(), size_t(SIGHT_SNAPSHOT_MAX_HANDS)));
        for (uint32_t hand = 0; hand < out->hands_count; ++hand) {
            const auto& data = signal.data.hands[hand];
            for (size_t i = 0; i < std::min(data.landmarks.size(), size_t(SIGHT_SNAPSHOT_HAND_POINTS)); ++i)
                out->hands[hand][i] = detail::snapshot_point(data.landmarks[i]);
            out->pinch[hand] = data.pinch;
            out->gesture_score[hand] = data.gesture_score;
            detail::copy_name(out->gesture[hand], data.gesture);
            detail::copy_name(out->handedness[hand], data.handedness);
        }
    } else if (signal.kind == TrackingKind::Face) {
        out->blendshape_count = static_cast<uint32_t>(std::min(
            signal.data.blendshapes.size(), size_t(SIGHT_SNAPSHOT_MAX_BLENDSHAPES)));
        for (uint32_t i = 0; i < out->blendshape_count; ++i)
            out->blendshapes[i] = signal.data.blendshapes[i];
        if (signal.data.has_head_transform) {
            out->has_head_transform = 1;
            std::copy(signal.data.head_transform.begin(),
                      signal.data.head_transform.end(), out->head_transform);
        }
    }
    return true;
}
} // namespace sight
