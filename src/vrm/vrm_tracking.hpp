#pragma once
#include "core/tracking_signal_bus.hpp"
#include "vrm_asset.hpp"
#include <chrono>
#include <map>
#include <set>
#include <string>
namespace sight::vrm {
// Stateless mapper for a single Face/Body signal pair. The native OBS
// source uses ease_face_pose below to retain pose through missing frames.
Pose from_tracking(const Asset &asset, const TrackingSignal *face,
                   const TrackingSignal *body = nullptr,
                   float torso_strength = 1.f);
// Render-thread-only face continuity. Head and expression measurements can
// disappear separately. Lost return 0 means hold until reacquired.
struct FacePoseState {
  Pose current;
  std::chrono::steady_clock::time_point last_frame{};
  std::chrono::steady_clock::time_point head_seen{};
  std::chrono::steady_clock::time_point expression_seen{};
  bool initialized = false;
  bool has_head = false;
  bool has_expressions = false;
};
Pose ease_face_pose(const Asset &asset, const TrackingSignal *face,
                    FacePoseState &state,
                    std::chrono::steady_clock::time_point now,
                    float easing_ms = 150.f, float lost_return_ms = 0.f,
                    float lost_hold_ms = 700.f);

// Render-thread-only persistent upper-body posture. An absent, stale, or
// low-confidence body signal holds the last visible joint orientation.
// Reloading the avatar resets this state to a relaxed initial arm pose.
struct BodyPoseState {
  Pose current;
  std::chrono::steady_clock::time_point last_frame{};
  bool initialized = false;
  float resting_degrees = -1.f;
  std::map<std::string, std::chrono::steady_clock::time_point> observed_at;
  // A relaxed initial pose is NOT an actual camera observation. This set
  // records only joints for which the retargeter accepted landmark input.
  std::set<std::string> observed_real;
  // Camera-relative resting shoulder center for upper-body-only framing.
  // A later valid hip observation still takes precedence when available.
  bool shoulder_reference_valid = false;
  float shoulder_reference_x = 0.f;
  float shoulder_reference_width = .3f;
};
Pose ease_body_pose(
    const Asset &asset, const TrackingSignal *body, BodyPoseState &state,
    std::chrono::steady_clock::time_point now, float easing_ms = 220.f,
    float resting_degrees = 65.f, const TrackingSignal *hands = nullptr,
    bool hands_drive_arms = false, bool mirror_hand_labels = false,
    const TrackingSignal *face = nullptr, float lost_return_ms = 0.f,
    float lost_hold_ms = 700.f, float torso_strength = 1.f);

// For a humanoid finger joint, return +1 or -1 for a local-X flexion
// that bends its child/tip toward the authored palm. Missing/ambiguous
// joints return 0 rather than blindly bending backward.
float finger_flexion_sign(const Asset &asset, const std::string &bone);

// Each hand is matched by its MediaPipe Left/Right label, never its unstable
// detection index. Loss can hold indefinitely or ease open after a grace time.
struct HandPoseState {
  Pose current;
  std::chrono::steady_clock::time_point last_frame{};
  std::map<std::string, std::chrono::steady_clock::time_point> observed_at;
  // Rig-specific hinge direction is stable across frames and cached once
  // per finger joint until model/toggle state is reset.
  std::map<std::string, float> finger_flex_signs;
  bool initialized = false;
  bool mirror_labels = false;
  bool wrists_enabled = true;
  bool fingers_enabled = true;
};
Pose ease_hand_pose(const Asset &asset, const TrackingSignal *hands,
                    HandPoseState &state,
                    std::chrono::steady_clock::time_point now,
                    float easing_ms = 120.f, bool mirror_labels = false,
                    float lost_return_ms = 0.f, float lost_hold_ms = 700.f,
                    bool rotate_wrists = true, const Pose *arm_pose = nullptr,
                    bool animate_fingers = true);
} // namespace sight::vrm
