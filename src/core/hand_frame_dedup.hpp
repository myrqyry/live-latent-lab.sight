#pragma once
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <vector>

namespace sight {

// Exact-byte rendered-frame suppression for tracking inference workers. The camera may generate 30 fps while OBS redraws at 60.
// Rendered image equality does not establish native camera timestamps;
// filtering and overlays can also change pixels between camera frames.
// A 250ms heartbeat maintains the shared VRM signal's freshness on a
// motionless scene and advances the MediaPipe VIDEO timestamp.
// Keep all comparisons on the inference worker, never graphics thread.
class TrackingFrameDedup {
public:
  using Clock = std::chrono::steady_clock;
  static constexpr auto kRefresh = std::chrono::milliseconds(250);
  explicit TrackingFrameDedup(unsigned warmup=1)
      : warmup_(std::max(1u,warmup)) {}

  bool should_infer(const std::vector<uint8_t> &pixels, uint32_t width,
                    uint32_t height, Clock::time_point now) const {
    const auto expected = uint64_t(width) * height * 4;
    if (!width || !height || pixels.size() != expected)
      return false;
    if (accepted_ < warmup_ || previous_.empty() || width != width_ || height != height_ ||
        now < last_inference_ || now - last_inference_ >= kRefresh)
      return true;
    return !std::equal(pixels.begin(), pixels.end(), previous_.begin());
  }

  // Call only after a successful inference, and only with the same
  // generation/model so an old image cannot poison a newly loaded task.
  void accept(std::vector<uint8_t> &&pixels, uint32_t width, uint32_t height,
              Clock::time_point now) {
    previous_ = std::move(pixels);
    width_ = width;
    height_ = height;
    last_inference_ = now;
    if (accepted_ < warmup_) ++accepted_;
  }

  void reset() {
    previous_.clear();
    width_ = height_ = 0;
    last_inference_ = {};
    accepted_ = 0;
  }

private:
  unsigned warmup_=1, accepted_=0;
  std::vector<uint8_t> previous_;
  uint32_t width_ = 0, height_ = 0;
  Clock::time_point last_inference_{};
};
// Keep the existing hand policy and its tests source-compatible.
using HandFrameDedup = TrackingFrameDedup;
} // namespace sight
