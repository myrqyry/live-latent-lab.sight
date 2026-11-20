#pragma once
#include "vrm_tracking.hpp"
#include <chrono>
#include <string>
namespace sight::vrm {

enum class JointStatus {
  Unmapped,
  NotDriven,
  Disabled,
  Waiting,
  Live,
  Held,
  Returning,
  Resting
};
struct CoverageInput {
  const Asset &asset;
  const FacePoseState &face;
  const BodyPoseState &body;
  const HandPoseState &hand;
  float lost_return_ms = 2000.f;
  float lost_hold_ms = 700.f;
  bool rotate_wrists = true;
  bool animate_fingers = true;
  bool hand_drive_arms = false;
  const TrackingSignal *face_signal = nullptr;
  const TrackingSignal *body_signal = nullptr;
  const TrackingSignal *hand_signal = nullptr;
};

JointStatus joint_status(const CoverageInput &in, const std::string &name,
                         std::chrono::steady_clock::time_point now);
std::string format_coverage(const CoverageInput &in,
                            std::chrono::steady_clock::time_point now);
} // namespace sight::vrm
