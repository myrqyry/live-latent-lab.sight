#include "vrm/vrm_tracking.hpp"
#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>
#include <string>

using namespace sight;
using namespace sight::vrm;
namespace {
Asset tpose() {
  Asset a;
  a.vrm_version = 1;
  a.nodes.resize(9);
  auto add = [&](int parent, int idx, Vec3 translation, Quat rotation = {}) {
    a.nodes[idx].parent = parent;
    if (parent >= 0)
      a.nodes[parent].children.push_back(idx);
    a.nodes[idx].translation = translation;
    a.nodes[idx].rotation = rotation;
  };
  add(-1, 0, {});
  add(0, 1, {0, 1.0f, 0});
  add(1, 2, {-.18f, .07f, 0});
  add(2, 3, {-.04f, -.02f, 0}, from_euler(0, 0, -1.5707963f));
  add(3, 4, {-.015f, -.17f, 0});
  add(4, 5, {-.005f, -.17f, 0});
  add(1, 6, {.18f, .07f, 0});
  add(6, 7, {.04f, -.02f, 0}, from_euler(0, 0, 1.5707963f));
  // Put right lower-arm and hand in a second branch with distinct nodes.
  a.nodes.resize(11);
  add(7, 9, {.015f, -.17f, 0});
  add(9, 10, {.005f, -.17f, 0});
  a.bones = {{"leftShoulder", 2},  {"leftUpperArm", 3},  {"leftLowerArm", 4},
             {"leftHand", 5},      {"rightShoulder", 6}, {"rightUpperArm", 7},
             {"rightLowerArm", 9}, {"rightHand", 10}};
  return a;
}
TrackingSignal body() {
  TrackingSignal s;
  s.kind = TrackingKind::Body;
  s.source_width = 640;
  s.source_height = 480;
  s.data.landmarks.resize(33, {.5f, .5f, 0.f, 0.f});
  for (auto [i, x, y] : {std::tuple{11, .65f, .35f},
                         {12, .35f, .35f},
                         {13, .71f, .60f},
                         {14, .29f, .60f},
                         {15, .78f, .78f},
                         {16, .22f, .78f}})
    s.data.landmarks[i] = {x, y, 0.f, 1.f};
  return s;
}
TrackingSignal hands() {
  TrackingSignal s;
  s.kind = TrackingKind::Hand;
  for (const auto [side, x] : {std::pair{"Left", .80f}, {"Right", .20f}}) {
    HandData h;
    h.handedness = side;
    h.landmarks.resize(21, {x, .60f, 0.f, 1.f});
    h.landmarks[0] = {x, .60f, 0.f, 1.f};
    s.data.hands.push_back(h);
  }
  return s;
}
bool equivalent(Quat a, Quat b) {
  // q and -q encode the same orientation.
  const float sum = a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
  return std::fabs(std::fabs(sum) - 1.f) < .003f;
}
void assert_same(const Pose &a, const Pose &b, const char *label) {
  assert(a.bone_rotation.size() == b.bone_rotation.size());
  for (const auto &[name, q] : a.bone_rotation) {
    const auto it = b.bone_rotation.find(name);
    assert(it != b.bone_rotation.end());
    if (!equivalent(q, it->second))
      std::cerr << label << " differs on " << name << "\n";
    assert(equivalent(q, it->second));
  }
}
Pose initial_rest(const Asset &asset) {
  BodyPoseState state;
  return ease_body_pose(
      asset, nullptr, state,
      std::chrono::steady_clock::time_point(std::chrono::seconds(1)), 120.f,
      40.f);
}
Pose driven_by_hand(const Asset &asset) {
  BodyPoseState state;
  const auto t = std::chrono::steady_clock::time_point(std::chrono::seconds(1));
  (void)ease_body_pose(asset, nullptr, state, t, 120.f, 40.f);
  const auto h = hands();
  return ease_body_pose(asset, nullptr, state,
                        t + std::chrono::milliseconds(40), 120.f, 40.f, &h,
                        true, false);
}
void check(const Asset &asset) {
  assert(asset.bones.contains("leftUpperArm"));
  assert(asset.bones.contains("leftLowerArm"));
  const auto relaxed = initial_rest(asset);

  assert(relaxed.bone_rotation.contains("leftUpperArm"));
  assert(relaxed.bone_rotation.contains("rightUpperArm"));
  assert(relaxed.bone_rotation.contains("leftLowerArm"));
  assert(relaxed.bone_rotation.contains("rightLowerArm"));
  const auto b = body();
  const auto tracked = from_tracking(asset, nullptr, &b);
  assert(tracked.bone_rotation.contains("leftUpperArm"));
  assert(tracked.bone_rotation.contains("rightUpperArm"));
  assert(tracked.bone_rotation.contains("leftLowerArm"));
  assert(tracked.bone_rotation.contains("rightLowerArm"));
  BodyPoseState state;
  const auto now = std::chrono::steady_clock::now();
  (void)ease_body_pose(asset, nullptr, state, now, 120.f, 40.f);
  const auto h = hands();
  const auto driven =
      ease_body_pose(asset, nullptr, state, now + std::chrono::milliseconds(40),
                     120.f, 40.f, &h, true, false);
  assert(driven.bone_rotation.contains("leftUpperArm"));
  assert(driven.bone_rotation.contains("rightUpperArm"));
  // This tests IK actually overrides the relaxed pose on both arms.
  assert(!equivalent(driven.bone_rotation.at("leftUpperArm"),
                     relaxed.bone_rotation.at("leftUpperArm")));
  assert(!equivalent(driven.bone_rotation.at("rightUpperArm"),
                     relaxed.bone_rotation.at("rightUpperArm")));
}
} // namespace
int main(int argc, char **argv) {
  const Asset trs_asset = tpose();
  Asset matrices = trs_asset;
  for (auto &n : matrices.nodes) {
    n.matrix = trs(n.translation, n.rotation, n.scale);
    n.use_matrix = true;
    n.translation = {};
    n.rotation = {};
    n.scale = {1, 1, 1};
  }
  // The same authored rest skeleton, represented by two legitimate
  // glTF node formats, must have the same relaxed/body/hand motion.
  check(trs_asset);
  check(matrices);
  assert_same(initial_rest(trs_asset), initial_rest(matrices), "rest");
  const auto b = body();
  assert_same(from_tracking(trs_asset, nullptr, &b),
              from_tracking(matrices, nullptr, &b), "body");
  assert_same(driven_by_hand(trs_asset), driven_by_hand(matrices), "wrist IK");
  if (argc > 1) {
    Asset actual;
    std::string err;
    assert(load(argv[1], actual, err) && err.empty());
    assert(actual.vrm_version == 1);
    assert(actual.bones.size() >= 17);
    assert(actual.nodes[actual.bones.at("leftUpperArm")].use_matrix);
    check(actual);
    std::cout << "ACTUAL_VIIRM_MATRIX_RIG_OK bones=" << actual.bones.size()
              << " surfaces=" << actual.surfaces.size() << "\n";
  }
  std::cout << "PASS TRS/matrix uniform VRM arm rest, body retarget, hand IK\n";
}
