#pragma once

#include "hand_signals.hpp"
#include "tasks_tracker.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace sight {

// MediaPipe's result index is not hand identity. Smooth using the
// handedness category, not hand[0]/hand[1], and forget a missing side.
// Leave unlabelled/ambiguous hands raw rather than borrowing another
// hand's previous position (which causes visible jumps on reacquisition).
struct HandSmoothingState {
  std::array<std::vector<TrackedPoint>, 2> previous{};
};

inline void smooth_hands_by_handedness(std::vector<HandData> &hands,
                                       HandSmoothingState &state,
                                       float previous_weight) {
  previous_weight = std::clamp(previous_weight, 0.f, .95f);
  std::array<unsigned, 2> counts{};
  const auto side_index = [](const std::string &label) {
    return label == "Left" ? 0 : label == "Right" ? 1 : -1;
  };
  for (const auto &hand : hands) {
    const int side = side_index(hand.handedness);
    if (side >= 0 && hand.landmarks.size() == 21)
      ++counts[side];
  }

  std::array<bool, 2> updated{};
  for (auto &hand : hands) {
    const int side = side_index(hand.handedness);
    // Never store a non-finite coordinate: blending it into the next
    // valid sample would otherwise keep that joint invalid indefinitely.
    const bool finite = std::all_of(
        hand.landmarks.begin(), hand.landmarks.end(), [](const TrackedPoint &p) {
          return std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z);
        });
    // A duplicate side label is ambiguous. Neither duplicate is
    // allowed to consume or overwrite the same history.
    if (side >= 0 && counts[side] == 1 && hand.landmarks.size() == 21 && finite) {
      auto &prior = state.previous[side];
      if (prior.size() == hand.landmarks.size() && previous_weight > 0.f) {
        for (size_t i = 0; i < hand.landmarks.size(); ++i) {
          auto &point = hand.landmarks[i];
          point.x =
              point.x * (1.f - previous_weight) + prior[i].x * previous_weight;
          point.y =
              point.y * (1.f - previous_weight) + prior[i].y * previous_weight;
          point.z =
              point.z * (1.f - previous_weight) + prior[i].z * previous_weight;
        }
      }
      prior = hand.landmarks;
      updated[side] = true;
    }
    hand.pinch = normalized_pinch(hand.landmarks);
  }
  for (int side = 0; side < 2; ++side)
    if (!updated[side])
      state.previous[side].clear();
}
} // namespace sight
