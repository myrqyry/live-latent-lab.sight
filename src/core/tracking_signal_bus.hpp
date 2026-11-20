#pragma once
#include "tasks_tracker.hpp"
#include <array>
#include <chrono>
#include <cstdint>
#include <string>

namespace sight {

// One-way, per-OBS-process tracking signal channel. Copying a snapshot
// never calls the model. All coordinates are normalized in the producer's
// native OBS video source, not the downscaled model input, OBS scene
// canvas, or display coordinates. source_width/height are the *native*
// producer-video dimensions even when inference runs at 256/512/640 px.
struct TrackingSignal {
    TrackingKind kind = TrackingKind::Face;
    TrackingData data;
    uint32_t source_width = 0;
    uint32_t source_height = 0;
    uint64_t sequence = 0;
    uint64_t publisher = 0;
    std::chrono::steady_clock::time_point updated{};
};

class TrackingSignalBus {
public:
    static uint64_t new_publisher_id();
    static bool publish(const std::string& channel, uint64_t publisher,
                        TrackingKind kind, const TrackingData& points,
                        uint32_t width, uint32_t height);
    // Atomic preflight + publication of three synchronized Holistic output
    // modalities. Readers see each publication with the same updated time,
    // and a collision on ANY channel rejects the whole group.
    static bool publish_group(const std::array<std::string,3>& channels,
                              uint64_t publisher,
                              const std::array<TrackingData,3>& data,
                              uint32_t width, uint32_t height);
    // Copies an immutable snapshot. A newly published empty detection still
    // reads as a valid snapshot, with data.landmarks.empty() == true.
    static bool read(const std::string& channel, std::chrono::milliseconds max_age,
                     TrackingSignal& out);
    static void withdraw(const std::string& channel, uint64_t publisher);
    static void clear_for_tests();
};
}
