#include "vrm/vrm_tracking.hpp"
#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>

using sight::TrackedPoint;
using sight::TrackingKind;
using sight::TrackingSignal;
using sight::vrm::Asset;
using sight::vrm::Pose;
using sight::vrm::Vec3;

namespace {
constexpr float Epsilon = 1e-4f;

Asset rig() {
  Asset avatar;
  avatar.vrm_version = 1;
  avatar.nodes.resize(9);
  avatar.nodes[0].children = {8};
  avatar.nodes[8].parent = 0;
  avatar.nodes[8].children = {1};
  avatar.nodes[8].translation = {0, .4f, 0};
  avatar.nodes[1].parent = 8;
  avatar.nodes[1].children = {2, 5};
  avatar.nodes[1].translation = {0, 1, 0};
  avatar.nodes[2].parent = 1;
  avatar.nodes[2].children = {3};
  avatar.nodes[2].translation = {0.5f, 0.25f, 0};
  avatar.nodes[3].parent = 2;
  avatar.nodes[3].children = {4};
  avatar.nodes[3].translation = {0.5f, 0, 0};
  avatar.nodes[4].parent = 3;
  avatar.nodes[4].translation = {0.5f, 0, 0};
  avatar.nodes[5].parent = 1;
  avatar.nodes[5].children = {6};
  avatar.nodes[5].translation = {-0.5f, 0.25f, 0};
  avatar.nodes[6].parent = 5;
  avatar.nodes[6].children = {7};
  avatar.nodes[6].translation = {-0.5f, 0, 0};
  avatar.nodes[7].parent = 6;
  avatar.nodes[7].translation = {-0.5f, 0, 0};
  avatar.bones = {{"spine", 8},         {"chest", 1},     {"leftUpperArm", 2},
                  {"leftLowerArm", 3},  {"leftHand", 4},  {"rightUpperArm", 5},
                  {"rightLowerArm", 6}, {"rightHand", 7}, {"head", 0}};
  avatar.expressions["aa"].push_back({0, 0, 0, 1.f});
  return avatar;
}
TrackingSignal body() {
  TrackingSignal signal;
  signal.kind = TrackingKind::Body;
  signal.source_width = 640;
  signal.source_height = 480;
  signal.data.landmarks.resize(33, {0.5f, 0.5f, 0.f, 0.f});
  auto point = [&](int i, float x, float y, float z = 0.f) {
    signal.data.landmarks[i] = {x, y, z, 1.f};
  };
  // Viewed from the front, the subject's left arm is on image-right.
  point(11, .65f, .35f);
  point(12, .35f, .35f);
  point(13, .85f, .35f);
  point(14, .15f, .35f);
  point(15, .95f, .35f);
  point(16, .05f, .35f);
  point(23, .6f, .75f);
  point(24, .4f, .75f);
  return signal;
}
float length(sight::vrm::Quat q) {
  return std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
}
} // namespace
int main() {
  const Asset avatar = rig();
  const TrackingSignal neutral = body();

  // Unseen body has no effect on the face-only avatar.
  const Pose missing = sight::vrm::from_tracking(avatar, nullptr, nullptr);
  assert(missing.bone_rotation.empty());
  const Pose symmetric = sight::vrm::from_tracking(avatar, nullptr, &neutral);
  assert(symmetric.bone_rotation.empty()); // No motion from T pose.

  // Lifting only the left elbow raises the corresponding upper arm,
  // not its mirrored partner.
  TrackingSignal raised = neutral;
  raised.data.landmarks[13] = {.65f, .15f, 0.f, 1.f};
  raised.data.landmarks[15] = {.65f, .03f, 0.f, 1.f};
  const Pose up = sight::vrm::from_tracking(avatar, nullptr, &raised);
  assert(up.bone_rotation.contains("leftUpperArm"));
  assert(up.bone_rotation.at("leftUpperArm").z > .15f);
  assert(!up.bone_rotation.contains("rightUpperArm"));
  assert(std::fabs(length(up.bone_rotation.at("leftUpperArm")) - 1.f) <
         Epsilon);
  assert(up.bone_rotation.at("leftUpperArm").x == 0.f);

  // Raising the right arm rotates its own upper arm the opposite way.
  TrackingSignal right_up = neutral;
  right_up.data.landmarks[14] = {.35f, .15f, 0.f, 1.f};
  right_up.data.landmarks[16] = {.35f, .03f, 0.f, 1.f};
  const Pose up_right = sight::vrm::from_tracking(avatar, nullptr, &right_up);
  assert(up_right.bone_rotation.at("rightUpperArm").z < -.15f);
  assert(!up_right.bone_rotation.contains("leftUpperArm"));

  // Wrist movement after the elbow bends the lower arm, separately from
  // the already-aimed upper arm.
  TrackingSignal bent = neutral;
  bent.data.landmarks[15] = {.85f, .18f, -.08f, 1.f};
  const Pose elbow = sight::vrm::from_tracking(avatar, nullptr, &bent);
  assert(elbow.bone_rotation.contains("leftLowerArm"));
  assert(!elbow.bone_rotation.contains("rightLowerArm"));

  // Tilt the shoulder line: chest roll moves, but no camera-induced head
  // pose is written. A separate face signal still owns head pitch.
  TrackingSignal lean = neutral;
  lean.data.landmarks[11].y = .25f;
  const Pose chest = sight::vrm::from_tracking(avatar, nullptr, &lean);
  assert(chest.bone_rotation.contains("chest"));
  assert(chest.bone_rotation.at("chest").z > .01f);
  assert(!chest.bone_rotation.contains("head"));

  // A whole-body lean with the shoulders level should move Peepo's
  // spine. Previously only shoulder-line tilt animated the chest.
  TrackingSignal torso_side = neutral;
  for (int id : {11, 12})
    torso_side.data.landmarks[id].x += .14f;
  const Pose side_lean =
      sight::vrm::from_tracking(avatar, nullptr, &torso_side);
  assert(side_lean.bone_rotation.contains("spine"));
  assert(side_lean.bone_rotation.at("spine").z < -.07f);
  assert(!side_lean.bone_rotation.contains("head"));
  assert(!side_lean.bone_rotation.contains("hips"));
  assert(!symmetric.bone_rotation.contains("spine"));
  TrackingSignal torso_forward = neutral;
  torso_forward.data.landmarks[11].z = -.24f;
  torso_forward.data.landmarks[12].z = -.24f;
  const Pose forward_lean =
      sight::vrm::from_tracking(avatar, nullptr, &torso_forward);
  assert(forward_lean.bone_rotation.contains("spine"));
  assert(forward_lean.bone_rotation.at("spine").x > .04f);
  TrackingSignal shifted = neutral;
  for (int id : {11, 12, 23, 24})
    shifted.data.landmarks[id].x += .06f;
  assert(!sight::vrm::from_tracking(avatar, nullptr, &shifted)
              .bone_rotation.contains("spine"));
  TrackingSignal invalid_hips = torso_side;
  invalid_hips.data.landmarks[23].confidence = .1f;
  invalid_hips.data.landmarks[24].confidence = .1f;
  assert(!sight::vrm::from_tracking(avatar, nullptr, &invalid_hips)
              .bone_rotation.contains("spine"));
  // The torso intensity control should allow suppressing spine motion,
  // independently from preserving body and face tracking.
  assert(!sight::vrm::from_tracking(avatar, nullptr, &torso_side, 0.f)
              .bone_rotation.contains("spine"));
  TrackingSignal face;
  face.kind = TrackingKind::Face;
  face.data.landmarks.resize(478);
  face.data.blendshape_names = {"jawOpen"};
  face.data.blendshapes = {.8f};
  face.data.has_head_transform = true;
  face.data.head_transform = {1, 0,       0,       0, 0, .87758f, -.47943f, 0,
                              0, .47943f, .87758f, 0, 0, 0,       0,        1};
  const Pose combined = sight::vrm::from_tracking(avatar, &face, &raised);
  const Pose face_only = sight::vrm::from_tracking(avatar, &face);
  assert(std::fabs(combined.expressions.at("aa") - .8f) < Epsilon);
  assert(std::fabs(combined.bone_rotation.at("head").x -
                   face_only.bone_rotation.at("head").x) < Epsilon);
  assert(combined.bone_rotation.contains("leftUpperArm"));
  assert(!face_only.bone_rotation.contains("leftUpperArm"));

  // A dropped/low-confidence wrist or elbow never reuses a previous pose,
  // and missing/bad source data may not animate a limb.
  TrackingSignal low = raised;
  low.data.landmarks[13].confidence = .1f;
  low.data.landmarks[15].confidence = .1f;
  assert(!sight::vrm::from_tracking(avatar, nullptr, &low)
              .bone_rotation.contains("leftUpperArm"));
  low = raised;
  low.data.landmarks[13].x = std::numeric_limits<float>::quiet_NaN();
  assert(!sight::vrm::from_tracking(avatar, nullptr, &low)
              .bone_rotation.contains("leftUpperArm"));
  low = raised;
  low.data.landmarks.resize(20);
  assert(
      sight::vrm::from_tracking(avatar, nullptr, &low).bone_rotation.empty());
  low = raised;
  low.kind = TrackingKind::Hand;
  assert(
      sight::vrm::from_tracking(avatar, nullptr, &low).bone_rotation.empty());

  // An avatar without a detected body should start from relaxed arms,
  // not a T pose. The default is symmetric and lowers both upper arms.
  using namespace std::chrono;
  sight::vrm::BodyPoseState continuity;
  const auto t0 = steady_clock::time_point(seconds(1));
  const auto rest =
      sight::vrm::ease_body_pose(avatar, nullptr, continuity, t0, 220.f, 65.f);
  assert(rest.bone_rotation.contains("leftUpperArm"));
  assert(rest.bone_rotation.contains("rightUpperArm"));
  assert(rest.bone_rotation.at("leftUpperArm").z < -.2f);
  assert(rest.bone_rotation.at("rightUpperArm").z > .2f);
  assert(!rest.bone_rotation.contains("head"));
  // Holding is per-joint and indefinite on missing/empty/stale-body signal.
  const auto hold_initial = sight::vrm::ease_body_pose(
      avatar, nullptr, continuity, t0 + seconds(5), 220.f, 65.f);
  assert(std::fabs(hold_initial.bone_rotation.at("leftUpperArm").z -
                   rest.bone_rotation.at("leftUpperArm").z) < Epsilon);

  // A newly reacquired body must approach the measured pose over time,
  // rather than instantly replacing the last pose.
  const auto first = sight::vrm::ease_body_pose(
      avatar, &raised, continuity, t0 + seconds(5) + milliseconds(16), 220.f,
      65.f);
  const auto goal = sight::vrm::from_tracking(avatar, nullptr, &raised);
  auto distance = [](sight::vrm::Quat a, sight::vrm::Quat b) {
    return 1.f - std::fabs(a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w);
  };
  const auto start_distance = distance(rest.bone_rotation.at("leftUpperArm"),
                                       goal.bone_rotation.at("leftUpperArm"));
  const auto first_distance = distance(first.bone_rotation.at("leftUpperArm"),
                                       goal.bone_rotation.at("leftUpperArm"));
  assert(first_distance < start_distance && first_distance > .05f);
  const auto moving = sight::vrm::ease_body_pose(
      avatar, &raised, continuity, t0 + seconds(5) + milliseconds(48), 220.f,
      65.f);
  assert(distance(moving.bone_rotation.at("leftUpperArm"),
                  goal.bone_rotation.at("leftUpperArm")) < first_distance);
  const auto frozen = sight::vrm::ease_body_pose(avatar, nullptr, continuity,
                                                 t0 + seconds(20), 220.f, 65.f);
  assert(distance(frozen.bone_rotation.at("leftUpperArm"),
                  moving.bone_rotation.at("leftUpperArm")) < Epsilon);
  TrackingSignal empty_body = raised;
  empty_body.data.landmarks.clear();
  const auto frozen_empty = sight::vrm::ease_body_pose(
      avatar, &empty_body, continuity, t0 + seconds(21), 220.f, 65.f);
  assert(distance(frozen_empty.bone_rotation.at("leftUpperArm"),
                  moving.bone_rotation.at("leftUpperArm")) < Epsilon);

  // When a valid body returns in a different pose, easing resumes from
  // the held orientation, with no instantaneous snap or full-delay leap.
  const auto resumed = sight::vrm::ease_body_pose(
      avatar, &neutral, continuity, t0 + seconds(21) + milliseconds(16), 220.f,
      65.f);
  assert(distance(resumed.bone_rotation.at("leftUpperArm"),
                  moving.bone_rotation.at("leftUpperArm")) < .01f);
  assert(distance(resumed.bone_rotation.at("leftUpperArm"),
                  sight::vrm::Quat{}) > .01f);

  // A desk camera may see the shoulders but not the hips. Calibrate
  // upper-body screen position on the first reliable shoulder frame;
  // moving the shoulder center sideways should still sway the spine.
  TrackingSignal upper_only = neutral;
  upper_only.data.landmarks[23].confidence = .1f;
  upper_only.data.landmarks[24].confidence = .1f;
  sight::vrm::BodyPoseState desk_state;
  const auto desk_t = t0 + seconds(100);
  (void)sight::vrm::ease_body_pose(avatar, &upper_only, desk_state, desk_t,
                                   220.f, 65.f);
  TrackingSignal desktop_lean = upper_only;
  desktop_lean.data.landmarks[11].x += .12f;
  desktop_lean.data.landmarks[12].x += .12f;
  const auto desktop_sway =
      sight::vrm::ease_body_pose(avatar, &desktop_lean, desk_state,
                                 desk_t + milliseconds(32), 220.f, 65.f);
  assert(desktop_sway.bone_rotation.contains("spine"));
  assert(desktop_sway.bone_rotation.at("spine").z < -.004f);
  const auto desktop_held = sight::vrm::ease_body_pose(
      avatar, nullptr, desk_state, desk_t + seconds(30), 220.f, 65.f);
  assert(distance(desktop_held.bone_rotation.at("spine"),
                  desktop_sway.bone_rotation.at("spine")) < Epsilon);
  // Turning the torso strength down to zero should ease spine toward
  // neutral even while the shoulders remain valid.
  const auto disabled_sway = sight::vrm::ease_body_pose(
      avatar, &desktop_lean, desk_state,
      desk_t + seconds(30) + milliseconds(32), 220.f, 65.f, nullptr, false,
      false, nullptr, 0.f, 700.f, 0.f);
  assert(distance(disabled_sway.bone_rotation.at("spine"), sight::vrm::Quat{}) <
         distance(desktop_held.bone_rotation.at("spine"), sight::vrm::Quat{}));
  // Full torso coordinates can't be inferred if shoulders are also lost.
  sight::vrm::BodyPoseState no_shoulders_state;
  (void)sight::vrm::ease_body_pose(avatar, &upper_only, no_shoulders_state,
                                   desk_t, 220.f, 65.f);
  auto invisible = desktop_lean;
  invisible.data.landmarks[11].confidence = .0f;
  invisible.data.landmarks[12].confidence = .0f;
  const auto missing_shoulders =
      sight::vrm::ease_body_pose(avatar, &invisible, no_shoulders_state,
                                 desk_t + milliseconds(32), 220.f, 65.f);
  assert(!missing_shoulders.bone_rotation.contains("spine"));
  // Torso receives the same hold/reacquisition easing as arms.
  sight::vrm::BodyPoseState torso_state;
  const auto torso_t = t0 + seconds(40);
  (void)sight::vrm::ease_body_pose(avatar, nullptr, torso_state, torso_t, 220.f,
                                   65.f);
  const auto torso_start =
      sight::vrm::ease_body_pose(avatar, &torso_side, torso_state,
                                 torso_t + milliseconds(16), 220.f, 65.f);
  assert(torso_start.bone_rotation.contains("spine"));
  assert(torso_start.bone_rotation.at("spine").z < -.002f);
  const auto torso_frozen = sight::vrm::ease_body_pose(
      avatar, nullptr, torso_state, torso_t + seconds(20), 220.f, 65.f);
  assert(distance(torso_frozen.bone_rotation.at("spine"),
                  torso_start.bone_rotation.at("spine")) < Epsilon);
  const auto torso_recover = sight::vrm::ease_body_pose(
      avatar, &neutral, torso_state, torso_t + seconds(20) + milliseconds(16),
      220.f, 65.f);
  assert(distance(torso_recover.bone_rotation.at("spine"), sight::vrm::Quat{}) <
         distance(torso_frozen.bone_rotation.at("spine"), sight::vrm::Quat{}));
  // A low-confidence elbow holds the prior upper-arm pose, while the
  // unaffected side can still follow reliable new landmarks.
  TrackingSignal mixed = right_up;
  mixed.data.landmarks[13].confidence = .1f;
  const auto only_right = sight::vrm::ease_body_pose(
      avatar, &mixed, continuity, t0 + seconds(21) + milliseconds(48), 220.f,
      65.f);
  assert(distance(only_right.bone_rotation.at("leftUpperArm"),
                  resumed.bone_rotation.at("leftUpperArm")) < Epsilon);
  assert(distance(only_right.bone_rotation.at("rightUpperArm"),
                  resumed.bone_rotation.at("rightUpperArm")) > Epsilon);
  // Explicit reset on model change restores the calm initial posture.
  continuity = {};
  const auto reset = sight::vrm::ease_body_pose(avatar, nullptr, continuity,
                                                t0 + seconds(30), 220.f, 65.f);
  assert(distance(reset.bone_rotation.at("leftUpperArm"),
                  rest.bone_rotation.at("leftUpperArm")) < Epsilon);

  // Desktop hand-only setup: a tracked hand wrist drives both shoulder
  // and elbow even when the Body model cannot see the user's torso.
  TrackingSignal hand_frame;
  hand_frame.kind = TrackingKind::Hand;
  sight::HandData raised_hand;
  raised_hand.handedness = "Left";
  raised_hand.landmarks.resize(21, {.5f, .5f, 0.f, 1.f});
  raised_hand.landmarks[0] = {.73f, .10f, 0.f, 1.f};
  hand_frame.data.hands = {raised_hand};
  hand_frame.data.landmarks = raised_hand.landmarks;
  hand_frame.source_width = 640;
  hand_frame.source_height = 480;
  sight::vrm::BodyPoseState hand_only;
  const auto h0 = t0 + std::chrono::seconds(60);
  auto hand_rest =
      sight::vrm::ease_body_pose(avatar, nullptr, hand_only, h0, 220.f, 65.f);
  assert(hand_rest.bone_rotation.contains("leftLowerArm"));
  assert(hand_rest.bone_rotation.contains("rightLowerArm"));
  // Off by default: publishing hands alone cannot silently move arms.
  const auto disabled = sight::vrm::ease_body_pose(
      avatar, nullptr, hand_only, h0 + std::chrono::milliseconds(16), 220.f,
      65.f, &hand_frame, false, false, nullptr);
  assert(distance(disabled.bone_rotation.at("leftUpperArm"),
                  hand_rest.bone_rotation.at("leftUpperArm")) < Epsilon);
  // Enabled: wrist position must animate both the matching upper arm and
  // forearm while the opposite arm stays in its current relaxed pose.
  const auto enabled = sight::vrm::ease_body_pose(
      avatar, nullptr, hand_only, h0 + std::chrono::milliseconds(48), 220.f,
      65.f, &hand_frame, true, false, nullptr);
  assert(distance(enabled.bone_rotation.at("leftUpperArm"),
                  disabled.bone_rotation.at("leftUpperArm")) > .0001f);
  assert(distance(enabled.bone_rotation.at("leftLowerArm"),
                  disabled.bone_rotation.at("leftLowerArm")) > .00001f);
  assert(distance(enabled.bone_rotation.at("rightUpperArm"),
                  disabled.bone_rotation.at("rightUpperArm")) < Epsilon);
  // Disappearance holds its last pose indefinitely with return speed 0.
  const auto vanished = sight::vrm::ease_body_pose(
      avatar, nullptr, hand_only, h0 + std::chrono::seconds(90), 220.f, 65.f,
      nullptr, true, false, nullptr, 0.f, 300.f);
  assert(distance(vanished.bone_rotation.at("leftUpperArm"),
                  enabled.bone_rotation.at("leftUpperArm")) < Epsilon);
  // An optional, nonzero lost-tracking return speed means: wait for the
  // grace period, then ease towards relaxed rather than snapping.
  const auto easing_home = sight::vrm::ease_body_pose(
      avatar, nullptr, hand_only,
      h0 + std::chrono::seconds(90) + std::chrono::milliseconds(16), 220.f,
      65.f, nullptr, true, false, nullptr, 1000.f, 300.f);
  assert(distance(easing_home.bone_rotation.at("leftUpperArm"),
                  enabled.bone_rotation.at("leftUpperArm")) > .0000001f);
  assert(distance(easing_home.bone_rotation.at("leftUpperArm"),
                  hand_rest.bone_rotation.at("leftUpperArm")) <
         distance(enabled.bone_rotation.at("leftUpperArm"),
                  hand_rest.bone_rotation.at("leftUpperArm")));
  assert(distance(easing_home.bone_rotation.at("leftUpperArm"),
                  hand_rest.bone_rotation.at("leftUpperArm")) > .0001f);
  // A different hand label drives the different arm, with optional swap.
  sight::vrm::BodyPoseState opposite;
  (void)sight::vrm::ease_body_pose(avatar, nullptr, opposite, h0, 220.f, 65.f);
  const auto right = sight::vrm::ease_body_pose(
      avatar, nullptr, opposite, h0 + std::chrono::milliseconds(48), 220.f,
      65.f, &hand_frame, true, true, nullptr);
  const auto mirror_rest = hand_rest.bone_rotation.at("rightUpperArm");
  assert(distance(right.bone_rotation.at("rightUpperArm"), mirror_rest) >
         .0001f);
  assert(distance(right.bone_rotation.at("leftUpperArm"),
                  hand_rest.bone_rotation.at("leftUpperArm")) < Epsilon);
  std::cout << "PASS: native VRM upper-body pose mapping, face+body fusion, "
               "rest fallback, shoulder tilt and confidence gates\n";
}
