#pragma once
#include "tasks_tracker.hpp"
#include <string_view>

namespace sight {
// MediaPipe's 21-point hand graph. Coordinates are normalized to source image.
float normalized_pinch(const std::vector<TrackedPoint>& points);
enum class HandGesture : int {
    None=0, ClosedFist, OpenPalm, PointingUp, ThumbDown, ThumbUp,
    Victory, ILoveYou
};
HandGesture parse_hand_gesture(std::string_view name);
float gesture_match(const HandData& hand, HandGesture target);
}
