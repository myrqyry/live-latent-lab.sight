#include "core/hand_temporal_smoothing.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>

namespace {
sight::HandData at(const char *name, float x) {
  sight::HandData h;
  h.handedness = name;
  h.landmarks.resize(21);
  h.world_landmarks.resize(21);
  for (auto &point : h.landmarks) {
    point.x = x;
    point.y = .4f;
    point.z = .01f;
  }
  // Set 0/9 and 4/8 to produce nonconstant pinch in real code.
  h.landmarks[9].y = .65f;
  h.landmarks[8].x = x + .10f;
  h.landmarks[4].x = x + .12f;
  h.world_landmarks[0].x = x * 10.f;
  return h;
}
bool close(float a, float b) { return std::fabs(a - b) < .00001f; }
} // namespace
int main() {
  sight::HandSmoothingState state;
  auto first =
      std::vector<sight::HandData>{at("Left", .20f), at("Right", .80f)};
  sight::smooth_hands_by_handedness(first, state, .5f);
  assert(close(first[0].landmarks[0].x, .20f));
  assert(close(first[1].landmarks[0].x, .80f));

  // RED: MediaPipe may return the same hands in the opposite array order.
  // Index-keyed smoothing incorrectly blends the left hand toward right.
  auto swapped =
      std::vector<sight::HandData>{at("Right", .84f), at("Left", .24f)};
  sight::smooth_hands_by_handedness(swapped, state, .5f);
  assert(close(swapped[0].landmarks[0].x, .82f));
  assert(close(swapped[1].landmarks[0].x, .22f));
  assert(close(swapped[0].world_landmarks[0].x, 8.4f));
  assert(close(swapped[1].world_landmarks[0].x, 2.4f));
  // Primary landmark alias is produced AFTER smoothing, by caller.
  // When one side disappears, its history must expire, rather than
  // being silently applied to a different single hand next frame.
  auto only_left = std::vector<sight::HandData>{at("Left", .30f)};
  sight::smooth_hands_by_handedness(only_left, state, .5f);
  assert(close(only_left[0].landmarks[0].x, .26f));
  auto only_right = std::vector<sight::HandData>{at("Right", .78f)};
  sight::smooth_hands_by_handedness(only_right, state, .5f);
  assert(close(only_right[0].landmarks[0].x, .78f));
  // Unknown/duplicate labels must never mix person/hand tracks or
  // contaminate later valid left/right observations.
  auto unknown = std::vector<sight::HandData>{at("", .42f)};
  sight::smooth_hands_by_handedness(unknown, state, .5f);
  assert(close(unknown[0].landmarks[0].x, .42f));
  auto duplicate =
      std::vector<sight::HandData>{at("Left", .12f), at("Left", .82f)};
  sight::smooth_hands_by_handedness(duplicate, state, .5f);
  assert(close(duplicate[0].landmarks[0].x, .12f));
  assert(close(duplicate[1].landmarks[0].x, .82f));
  auto back = std::vector<sight::HandData>{at("Left", .33f)};
  sight::smooth_hands_by_handedness(back, state, .5f);
  assert(close(back[0].landmarks[0].x, .33f));
  auto none = std::vector<sight::HandData>{};
  sight::smooth_hands_by_handedness(none, state, .5f);
  auto next = std::vector<sight::HandData>{at("Left", .58f)};
  sight::smooth_hands_by_handedness(next, state, .5f);
  assert(close(next[0].landmarks[0].x, .58f));
  // Invalid coordinates must not poison future valid samples indefinitely.
  auto invalid = std::vector<sight::HandData>{at("Left", .6f), at("Right", .8f)};
  invalid[0].landmarks[8].x = std::numeric_limits<float>::quiet_NaN();
  sight::smooth_hands_by_handedness(invalid, state, .5f);
  auto recovered = std::vector<sight::HandData>{at("Left", .7f), at("Right", .9f)};
  sight::smooth_hands_by_handedness(recovered, state, .5f);
  if (!close(recovered[0].landmarks[8].x, .8f) ||
      !close(recovered[1].landmarks[0].x, .85f)) {
    std::cerr << "invalid hand poisoned recovery or reset the other hand\n";
    return 1;
  }
  std::cout
      << "PASS: handedness-stable hand smoothing under detection reordering, "
         "dropout and duplicate labels\n";
}
