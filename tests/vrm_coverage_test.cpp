#include "vrm/vrm_coverage.hpp"
#include <cassert>
#include <chrono>
#include <iostream>
using namespace sight::vrm;
int main() {
  using clock = std::chrono::steady_clock;
  const auto t = clock::time_point(std::chrono::seconds(50));
  Asset a;
  int node = 0;
  for (const auto *name : {"hips",
                           "spine",
                           "chest",
                           "neck",
                           "upperChest",
                           "head",
                           "leftShoulder",
                           "rightShoulder",
                           "leftUpperArm",
                           "leftLowerArm",
                           "rightUpperArm",
                           "rightLowerArm",
                           "leftHand",
                           "rightHand",
                           "leftIndexProximal",
                           "leftIndexIntermediate",
                           "leftIndexDistal",
                           "rightThumbMetacarpal",
                           "leftEye",
                           "rightFoot"}) {
    a.bones[name] = node++;
  }
  FacePoseState face;
  BodyPoseState body;
  HandPoseState hand;
  CoverageInput input{a, face, body, hand};
  assert(joint_status(input, "leftUpperArm", t) == JointStatus::Waiting);
  assert(joint_status(input, "spine", t) == JointStatus::Waiting);
  assert(joint_status(input, "head", t) == JointStatus::Waiting);
  assert(joint_status(input, "leftHand", t) == JointStatus::Waiting);
  assert(joint_status(input, "rightThumbMetacarpal", t) ==
         JointStatus::Waiting);
  assert(joint_status(input, "hips", t) == JointStatus::NotDriven);
  assert(joint_status(input, "neck", t) == JointStatus::NotDriven);
  assert(joint_status(input, "upperChest", t) == JointStatus::NotDriven);
  assert(joint_status(input, "leftEye", t) == JointStatus::NotDriven);
  assert(joint_status(input, "leftUpperLeg", t) == JointStatus::Unmapped);
  assert(joint_status(input, "leftIndexIntermediate", t) ==
         JointStatus::Waiting);
  // The relaxed rest pose seeds the old Body observed_at map at init.
  // That must NOT count as real tracking or show a fake held arm.
  body.initialized = true;
  body.observed_at["leftUpperArm"] = t;
  body.current.bone_rotation["leftUpperArm"] = from_euler(.4f, 0, 0);
  assert(joint_status(input, "leftUpperArm", t) == JointStatus::Waiting);
  body.observed_real.insert("leftUpperArm");
  assert(joint_status(input, "leftUpperArm", t) == JointStatus::Live);
  assert(joint_status(input, "leftUpperArm",
                      t + std::chrono::milliseconds(350)) == JointStatus::Held);
  assert(
      joint_status(input, "leftUpperArm", t + std::chrono::milliseconds(900)) ==
      JointStatus::Returning);
  assert(joint_status(input, "leftUpperArm", t + std::chrono::seconds(8)) ==
         JointStatus::Resting);
  input.lost_return_ms = 0;
  assert(joint_status(input, "leftUpperArm", t + std::chrono::seconds(8)) ==
         JointStatus::Held);
  input.lost_return_ms = 2000;
  face.has_head = true;
  face.head_seen = t;
  assert(joint_status(input, "head", t) == JointStatus::Live);
  assert(joint_status(input, "head", t + std::chrono::milliseconds(800)) ==
         JointStatus::Returning);
  hand.observed_at["leftHand"] = t;
  assert(joint_status(input, "leftHand", t) == JointStatus::Live);
  input.rotate_wrists = false;
  assert(joint_status(input, "leftHand", t) == JointStatus::Disabled);
  input.rotate_wrists = true;
  hand.observed_at["leftIndexProximal"] = t;
  input.animate_fingers = false;
  assert(joint_status(input, "leftIndexProximal", t) == JointStatus::Disabled);
  input.animate_fingers = true;
  assert(joint_status(input, "leftIndexProximal", t) == JointStatus::Live);
  sight::TrackingSignal face_signal, body_signal, hand_signal;
  face_signal.kind = sight::TrackingKind::Face;
  face_signal.data.landmarks.resize(478);
  face_signal.updated = t;
  body_signal.kind = sight::TrackingKind::Body;
  body_signal.data.landmarks.resize(33);
  body_signal.data.world_landmarks.resize(33);
  body_signal.updated = t;
  hand_signal.kind = sight::TrackingKind::Hand;
  hand_signal.data.landmarks.resize(21);
  hand_signal.data.hands.resize(1);
  hand_signal.data.hands[0].handedness = "Left";
  hand_signal.data.hands[0].landmarks.resize(21);
  hand_signal.data.hands[0].world_landmarks.resize(21);
  hand_signal.updated = t;
  input.face_signal = &face_signal;
  input.body_signal = &body_signal;
  input.hand_signal = &hand_signal;
  input.hand_drive_arms = false;
  auto report = format_coverage(input, t);
  assert(report.find("world=33") != std::string::npos);
  assert(report.find("world=21") != std::string::npos);
  assert(report.find("neck") != std::string::npos);
  assert(report.find("hips") != std::string::npos);
  assert(report.find("tracked") != std::string::npos);
  assert(report.find("never driven") != std::string::npos);
  assert(report.find("Hands Drive Arms: off") != std::string::npos);
  assert(report.find("Reopen Properties") != std::string::npos);
  // A legitimate ViiRM-style humanoid without finger chains can still
  // move both hands; flag the capability honestly rather than implying
  // "Animate Fingers" works on this rig.
  for (const char *finger : {"leftIndexProximal", "leftIndexIntermediate",
                             "leftIndexDistal", "rightThumbMetacarpal"})
    a.bones.erase(finger);
  const auto fingerless_report = format_coverage(input, t);
  assert(fingerless_report.find("no humanoid finger bones") !=
         std::string::npos);
  assert(fingerless_report.find("wrist tracking still available") !=
         std::string::npos);
  a.bones.erase("leftHand");
  assert(joint_status(input, "leftHand", t) == JointStatus::Unmapped);
  std::cout << "PASS: no false rest observations, live/held/returning,"
               " unmapped, disabled, 3D sources and exclusions\n";
}
