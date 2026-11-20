#pragma once

#include "mediapipe/tasks/c/components/containers/landmark.h"
#include "tasks_tracker.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace sight {

// The pinned native C result structures expose optional world-landmark
// arrays separately from normalized image landmarks. Treat each list as a
// complete same-index 3D-meter observation or skip it; never allow a
// partial/NaN hand/pose to masquerade as a usable calibrated skeleton.
inline std::vector<TrackedPoint> copy_world_landmarks(const MpLandmarks *group,
                                                      size_t expected) {
  if (!group || !group->landmarks || group->landmarks_count != expected ||
      (expected != 21 && expected != 33))
    return {};
  std::vector<TrackedPoint> out;
  out.reserve(expected);
  for (size_t i = 0; i < expected; ++i) {
    const auto &p = group->landmarks[i];
    if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z) ||
        (p.has_visibility && !std::isfinite(p.visibility)) ||
        (p.has_presence && !std::isfinite(p.presence)))
      return {};
    float confidence = 1.f;
    if (p.has_visibility)
      confidence = std::min(confidence, std::clamp(p.visibility, 0.f, 1.f));
    if (p.has_presence)
      confidence = std::min(confidence, std::clamp(p.presence, 0.f, 1.f));
    out.push_back({p.x, p.y, p.z, confidence});
  }
  return out;
}

} // namespace sight
