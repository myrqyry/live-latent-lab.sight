#include "vrm/vrm_tracking.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
int main() {
  sight::vrm::Asset avatar;
  avatar.vrm_version = 1;
  for (const auto &name : {"aa", "blinkLeft", "blinkRight", "happy"})
    avatar.expressions[name].push_back({0, 0, 0, 1.f});
  avatar.bones["head"] = 0;
  sight::TrackingSignal sample;
  sample.kind = sight::TrackingKind::Face;
  sample.data.landmarks.resize(478);
  sample.data.blendshape_names = {"jawOpen", "eyeBlinkLeft", "eyeBlinkRight",
                                  "mouthSmileLeft", "mouthSmileRight"};
  sample.data.blendshapes = {.8f, .6f, .2f, .7f, .5f};
  const auto mapped = sight::vrm::from_tracking(avatar, &sample);
  assert(std::fabs(mapped.expressions.at("aa") - .8f) < .0001);
  assert(std::fabs(mapped.expressions.at("blinkLeft") - .6f) < .0001);
  assert(std::fabs(mapped.expressions.at("blinkRight") - .2f) < .0001);
  assert(std::fabs(mapped.expressions.at("happy") - .6f) < .0001);
  assert(!mapped.bone_rotation.contains("head"));
  // Missing detection must not retain previous mouth or eyes.
  sample.data.landmarks.clear();
  assert(sight::vrm::from_tracking(avatar, &sample).expressions.empty());
  sample.data.landmarks.resize(478);
  sample.data.blendshapes[0] = NAN;
  const auto invalid = sight::vrm::from_tracking(avatar, &sample);
  assert(!invalid.expressions.contains("aa"));
  avatar.expressions.clear();
  avatar.vrm_version = 0;
  avatar.expressions["A"].push_back({0, 0, 0, 1});
  avatar.expressions["Blink_L"].push_back({0, 0, 0, 1});
  sample.data.blendshapes[0] = .7f;
  const auto old = sight::vrm::from_tracking(avatar, &sample);
  assert(std::fabs(old.expressions.at("A") - .7f) < .001f);
  assert(std::fabs(old.expressions.at("Blink_L") - .6f) < .001f);
  // MediaPipe matrix rotation is finite and bounded; translation is ignored.
  sample.data.has_head_transform = true;
  sample.data.head_transform = {1,         0, 0,  0,        0,        0.87758f,
                                -0.47943f, 0, 0,  0.47943f, 0.87758f, 0,
                                0,         0, 10, 1};
  const auto head = sight::vrm::from_tracking(avatar, &sample);
  assert(head.bone_rotation.contains("head"));
  // User-facing pitch contract: when MediaPipe reports a positive X
  // rotation, the displayed glTF/VRM head pitch must be the opposite sign.
  // Otherwise looking upward/downward moves Peepo in the wrong direction.
  assert(head.bone_rotation.at("head").x < -0.1f);
  sample.data.head_transform[6] = 0.47943f;
  sample.data.head_transform[9] = -0.47943f;
  const auto mirrored = sight::vrm::from_tracking(avatar, &sample);
  assert(mirrored.bone_rotation.at("head").x > 0.1f);
  // Camera-space yaw must agree with the VRM's *visible* arm/hand side.
  // This is independent from the already-corrected vertical pitch.
  // MediaPipe's +Y rotation matrix previously made Peepo look AWAY
  // from the same-side tracked fist, opposite his arms/hands.
  sample.data.head_transform = {0.98007f,  0, 0.19867f, 0, 0, 1, 0, 0,
                                -0.19867f, 0, 0.98007f, 0, 0, 0, 0, 1};
  const auto yaw_only = sight::vrm::from_tracking(avatar, &sample);
  assert(yaw_only.bone_rotation.at("head").y < -0.05f);
  // The opposite turn must reverse the mapped yaw too.
  sample.data.head_transform[2] = -0.19867f;
  sample.data.head_transform[8] = 0.19867f;
  const auto opposite_yaw = sight::vrm::from_tracking(avatar, &sample);
  assert(opposite_yaw.bone_rotation.at("head").y > 0.05f);
  // Keep head pitch and roll independent of this horizontal correction.
  sample.data.head_transform = {0.95534f, -0.29552f, 0, 0, 0.29552f, 0.95534f,
                                0,        0,         0, 0, 1,        0,
                                0,        0,         0, 1};
  const auto roll_only = sight::vrm::from_tracking(avatar, &sample);
  assert(roll_only.bone_rotation.at("head").z > 0.05f);

  // A face source currently snaps from the tracked head/mouth pose to
  // neutral when tracking expires. The native source needs persistent
  // face state, like the independent body/hand animation paths.
  avatar.vrm_version = 1;
  avatar.expressions.clear();
  avatar.expressions["aa"].push_back({0, 0, 0, 1.f});
  avatar.expressions["blink"].push_back({0, 0, 0, 1.f});
  sample.data.landmarks.resize(478);
  sample.data.blendshape_names = {"jawOpen", "eyeBlinkLeft", "eyeBlinkRight"};
  sample.data.blendshapes = {1.f, 1.f, 1.f};
  sample.data.has_head_transform = true;
  sample.data.head_transform = {0.98007f,  0, 0.19867f, 0, 0, 1, 0, 0,
                                -0.19867f, 0, 0.98007f, 0, 0, 0, 0, 1};
  using namespace std::chrono;
  const auto t = steady_clock::time_point(seconds(10));
  sight::vrm::FacePoseState state;
  const auto first = sight::vrm::ease_face_pose(avatar, &sample, state, t,
                                                150.f, 1000.f, 500.f);
  assert(first.expressions.empty()); // start neutral, never jump on start
  const auto active = sight::vrm::ease_face_pose(
      avatar, &sample, state, t + milliseconds(100), 150.f, 1000.f, 500.f);
  assert(active.bone_rotation.contains("head"));
  assert(active.bone_rotation.at("head").y < -.01f);
  assert(active.expressions.at("aa") > .1f &&
         active.expressions.at("aa") < 1.f);
  assert(active.expressions.at("blink") > .1f);
  // A stale face signal holds both head and mouth for the hold delay.
  const auto held = sight::vrm::ease_face_pose(
      avatar, nullptr, state, t + milliseconds(300), 150.f, 1000.f, 500.f);
  assert(std::fabs(held.expressions.at("aa") - active.expressions.at("aa")) <
         .0001f);
  assert(std::fabs(held.bone_rotation.at("head").y -
                   active.bone_rotation.at("head").y) < .0001f);
  // After the hold delay, return gradually, not all at once.
  const auto easing = sight::vrm::ease_face_pose(
      avatar, nullptr, state, t + milliseconds(800), 150.f, 1000.f, 500.f);
  assert(easing.expressions.at("aa") < held.expressions.at("aa"));
  assert(easing.expressions.at("aa") > .01f);
  assert(std::fabs(easing.bone_rotation.at("head").y) <
         std::fabs(held.bone_rotation.at("head").y));
  assert(std::fabs(easing.bone_rotation.at("head").y) > .001f);
  // Reacquiring face landmarks eases toward fresh, opposing yaw and
  // reduced jaw instead of snapping or reusing the prior target.
  auto changed = sample;
  changed.data.blendshapes = {.1f, .1f, .1f};
  changed.data.head_transform[2] = -.19867f;
  changed.data.head_transform[8] = .19867f;
  const auto reacquired = sight::vrm::ease_face_pose(
      avatar, &changed, state, t + milliseconds(816), 150.f, 1000.f, 500.f);
  assert(reacquired.bone_rotation.at("head").y >
         easing.bone_rotation.at("head").y);
  assert(reacquired.bone_rotation.at("head").y < .05f);
  // Face remains held indefinitely when return speed is configured zero.
  sight::vrm::FacePoseState forever;
  (void)sight::vrm::ease_face_pose(avatar, &sample, forever, t, 150.f, 0.f,
                                   500.f);
  const auto lasting = sight::vrm::ease_face_pose(
      avatar, &sample, forever, t + milliseconds(80), 150.f, 0.f, 500.f);
  const auto still = sight::vrm::ease_face_pose(
      avatar, nullptr, forever, t + seconds(50), 150.f, 0.f, 500.f);
  assert(std::fabs(still.expressions.at("aa") - lasting.expressions.at("aa")) <
         .0001f);
  // Valid face can update mouth even if its head transform is invalid:
  // do not silently reset the existing head turn.
  changed.data.has_head_transform = false;
  const auto only_mouth = sight::vrm::ease_face_pose(
      avatar, &changed, state, t + milliseconds(832), 150.f, 1000.f, 500.f);
  assert(only_mouth.bone_rotation.contains("head"));
  assert(std::fabs(only_mouth.bone_rotation.at("head").y -
                   reacquired.bone_rotation.at("head").y) < .0001f);
  // An explicitly observed neutral jaw is not a tracking dropout; it
  // should smooth toward zero, rather than sticking open.
  changed.data.blendshapes = {0.f, 0.f, 0.f};
  const auto observed_neutral = sight::vrm::ease_face_pose(
      avatar, &changed, state, t + milliseconds(848), 150.f, 1000.f, 500.f);
  assert(observed_neutral.expressions.at("aa") <
         only_mouth.expressions.at("aa"));
  std::cout << "PASS: named Face MediaPipe scores drive VRM 0.x / 1.0 "
               "expressions; stale and bad data ignored\n";
}
