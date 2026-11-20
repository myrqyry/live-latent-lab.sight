#include "vrm_coverage.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <map>
#include <sstream>
#include <string_view>

namespace sight::vrm {
namespace {
using Clock = std::chrono::steady_clock;
bool is_body(std::string_view name) {
  for (const auto candidate : {"spine", "chest", "leftUpperArm", "leftLowerArm",
                               "rightUpperArm", "rightLowerArm"})
    if (name == candidate)
      return true;
  return false;
}
bool is_finger(std::string_view name) {
  std::string_view rest;
  if (name.starts_with("left"))
    rest = name.substr(4);
  else if (name.starts_with("right"))
    rest = name.substr(5);
  else
    return false;
  for (const auto root : {"Index", "Middle", "Ring", "Little", "Thumb"})
    if (rest.starts_with(root)) {
      const auto suffix = rest.substr(std::string_view(root).size());
      return suffix == "Metacarpal" || suffix == "Proximal" ||
             suffix == "Intermediate" || suffix == "Distal";
    }
  return false;
}
JointStatus temporal(Clock::time_point seen, Clock::time_point now, float hold,
                     float duration) {
  if (seen == Clock::time_point{})
    return JointStatus::Waiting;
  const float elapsed = std::max(
      0.f, std::chrono::duration<float, std::milli>(now - seen).count());
  if (elapsed <= 50.f)
    return JointStatus::Live;
  if (duration <= 0.f || elapsed < std::max(0.f, hold))
    return JointStatus::Held;
  // Exponential ease is >95% settled after three time constants.
  // Diagnostic label is approximate: the animation isn't hard-clamped.
  if (elapsed >= std::max(0.f, hold) + 3.f * duration)
    return JointStatus::Resting;
  return JointStatus::Returning;
}
const char *label(JointStatus s) {
  switch (s) {
  case JointStatus::Unmapped:
    return "not mapped";
  case JointStatus::NotDriven:
    return "never driven";
  case JointStatus::Disabled:
    return "disabled";
  case JointStatus::Waiting:
    return "waiting";
  case JointStatus::Live:
    return "tracked";
  case JointStatus::Held:
    return "held";
  case JointStatus::Returning:
    return "returning";
  case JointStatus::Resting:
    return "near rest";
  }
  return "?";
}
std::string signal(const TrackingSignal *v, std::string_view kind,
                   Clock::time_point now) {
  if (!v)
    return "absent/expired";
  std::ostringstream s;
  if (v->data.landmarks.empty()) {
    s << "no subject";
  } else {
    s << v->data.landmarks.size() << " image";
    if (!v->data.world_landmarks.empty())
      s << ", world=" << v->data.world_landmarks.size();
    if (kind == "hand") {
      s << ", detected=" << v->data.hands.size();
      for (const auto &hand : v->data.hands) {
        s << " [" << (hand.handedness.empty() ? "?" : hand.handedness);
        if (!hand.world_landmarks.empty())
          s << " world=" << hand.world_landmarks.size();
        s << "]";
      }
    }
  }
  s << ", age="
    << std::max(0L, std::chrono::duration_cast<std::chrono::milliseconds>(
                        now - v->updated)
                        .count())
    << "ms";
  return s.str();
}
} // namespace

JointStatus joint_status(const CoverageInput &in, const std::string &name,
                         Clock::time_point now) {
  if (!in.asset.bones.contains(name))
    return JointStatus::Unmapped;
  if (name == "head")
    return in.face.has_head ? temporal(in.face.head_seen, now, in.lost_hold_ms,
                                       in.lost_return_ms)
                            : JointStatus::Waiting;
  if (is_body(name)) {
    if (!in.body.observed_real.contains(name))
      return JointStatus::Waiting;
    const auto found = in.body.observed_at.find(name);
    return found == in.body.observed_at.end()
               ? JointStatus::Waiting
               : temporal(found->second, now, in.lost_hold_ms,
                          in.lost_return_ms);
  }
  const bool wrist = name == "leftHand" || name == "rightHand";
  if (wrist || is_finger(name)) {
    if ((wrist && !in.rotate_wrists) || (!wrist && !in.animate_fingers))
      return JointStatus::Disabled;
    const auto found = in.hand.observed_at.find(name);
    return found == in.hand.observed_at.end()
               ? JointStatus::Waiting
               : temporal(found->second, now, in.lost_hold_ms,
                          in.lost_return_ms);
  }
  return JointStatus::NotDriven;
}

std::string format_coverage(const CoverageInput &in, Clock::time_point now) {
  std::ostringstream out;
  size_t driven = 0, not_driven = 0, disabled = 0;
  std::string omitted;
  for (const auto &[bone, _] : in.asset.bones) {
    switch (joint_status(in, bone, now)) {
    case JointStatus::NotDriven:
      ++not_driven;
      if (!omitted.empty())
        omitted += ", ";
      omitted += bone;
      break;
    case JointStatus::Disabled:
      ++disabled;
      break;
    default:
      ++driven;
      break;
    }
  }
  out << "Rig coverage (snapshot, not a live poll)\n";
  out << "Mapped=" << in.asset.bones.size() << "; targets=" << driven
      << "; disabled=" << disabled << "; never driven=" << not_driven << "\n";
  out << "Signals: Face " << signal(in.face_signal, "face", now) << "; Body "
      << signal(in.body_signal, "body", now) << "; Hands "
      << signal(in.hand_signal, "hand", now) << "\n";
  out << "Head=" << label(joint_status(in, "head", now));
  if (in.asset.expressions.empty()) {
    out << "; expressions=not mapped\n";
  } else {
    out << "; expressions="
        << label(in.face.has_expressions
                     ? temporal(in.face.expression_seen, now, in.lost_hold_ms,
                                in.lost_return_ms)
                     : JointStatus::Waiting)
        << "\n";
  }
  out << "Torso: spine=" << label(joint_status(in, "spine", now))
      << ", chest=" << label(joint_status(in, "chest", now))
      << ", neck=" << label(joint_status(in, "neck", now))
      << ", upperChest=" << label(joint_status(in, "upperChest", now)) << "\n";
  for (const auto *side : {"left", "right"}) {
    out << side << " arm: upper="
        << label(joint_status(in, std::string(side) + "UpperArm", now))
        << ", lower="
        << label(joint_status(in, std::string(side) + "LowerArm", now))
        << ", wrist="
        << label(joint_status(in, std::string(side) + "Hand", now)) << "\n";
  }
  std::map<JointStatus, int> finger_status;
  for (const auto &[name, _] : in.asset.bones)
    if (is_finger(name))
      ++finger_status[joint_status(in, name, now)];
  out << "Fingers:";
  if (finger_status.empty())
    out << " no humanoid finger bones (wrist tracking still available)";
  for (const auto status :
       {JointStatus::Live, JointStatus::Held, JointStatus::Returning,
        JointStatus::Resting, JointStatus::Waiting, JointStatus::Disabled})
    if (finger_status[status])
      out << " " << label(status) << "=" << finger_status[status];
  out << "\nHands Drive Arms: " << (in.hand_drive_arms ? "on" : "off") << "\n";
  out << "Mapped, never driven: " << (omitted.empty() ? "none" : omitted)
      << "\n";
  out << "'tracked' = recent accepted input, not necessarily visible motion.\n";
  out << "'near rest' is estimated after 3 easing intervals; no snap "
         "assumed.\n";
  out << "Reopen Properties to refresh this ~1-second cached snapshot.";
  return out.str();
}
} // namespace sight::vrm
