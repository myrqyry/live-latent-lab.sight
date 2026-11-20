#include "core/tasks_world_landmarks.hpp"
#include "core/tracking_signal_bus.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

int main() {
  using sight::TrackedPoint;
  std::vector<MpLandmark> world(33);
  for (int i = 0; i < 33; ++i) {
    world[i].x = .012f * i; // meters: not normalized image X
    world[i].y = -1.1f + .019f * i;
    world[i].z = -.37f + .007f * i;
    world[i].has_visibility = true;
    world[i].visibility = .95f;
    world[i].has_presence = true;
    world[i].presence = .83f;
  }
  MpLandmarks pose_raw{world.data(), uint32_t(world.size())};
  auto pose = sight::copy_world_landmarks(&pose_raw, 33);
  assert(pose.size() == 33);
  assert(std::fabs(pose[0].y + 1.1f) < .0001f);
  assert(std::fabs(pose[32].x - .384f) < .0001f);
  assert(std::fabs(pose[2].confidence - .83f) < .0001f);
  // Bad / incomplete results must not poison the valid image landmarks.
  assert(sight::copy_world_landmarks(&pose_raw, 21).empty());
  assert(sight::copy_world_landmarks(nullptr, 33).empty());
  world[10].z = std::numeric_limits<float>::quiet_NaN();
  assert(sight::copy_world_landmarks(&pose_raw, 33).empty());
  world[10].z = -.2f;
  world[12].visibility = std::numeric_limits<float>::quiet_NaN();
  assert(sight::copy_world_landmarks(&pose_raw, 33).empty());
  world[12].has_visibility = false; // optional value is not valid or required
  pose = sight::copy_world_landmarks(&pose_raw, 33);
  assert(pose.size() == 33 && pose[12].confidence == .83f);

  // Independent world data for two hands must match the same per-hand
  // category/normalized indices and survive a bus copy unmodified.
  std::vector<MpLandmark> left(21), right(21);
  for (int i = 0; i < 21; ++i) {
    left[i].x = .01f * i;
    left[i].y = .02f * i;
    left[i].z = -.03f * i;
    right[i].x = -.015f * i;
    right[i].y = .006f * i;
    right[i].z = .04f * i;
  }
  MpLandmarks left_raw{left.data(), 21}, right_raw{right.data(), 21};
  sight::TrackingSignalBus::clear_for_tests();
  constexpr auto body_channel = "test/world_body";
  constexpr auto hand_channel = "test/world_hand";
  constexpr uint64_t publisher = 501;
  sight::TrackingData body;
  body.world_landmarks = pose;
  body.landmarks.resize(33, {.5f, .5f, 0.f, 1.f});
  assert(sight::TrackingSignalBus::publish(
      body_channel, publisher, sight::TrackingKind::Body, body, 640, 480));
  sight::TrackingSignal signal;
  assert(sight::TrackingSignalBus::read(
      body_channel, std::chrono::milliseconds(1000), signal));
  assert(signal.kind == sight::TrackingKind::Body);
  assert(signal.data.world_landmarks.size() == 33);
  assert(signal.data.landmarks.size() == 33);
  assert(signal.data.hands.empty());
  signal.data.world_landmarks[0].x = 999.f;
  assert(sight::TrackingSignalBus::read(
      body_channel, std::chrono::milliseconds(1000), signal));
  assert(signal.data.world_landmarks[0].x != 999.f);

  sight::TrackingData hands;
  hands.hands.resize(2);
  hands.hands[0].handedness = "Right";
  hands.hands[0].landmarks.resize(21);
  hands.hands[0].world_landmarks = sight::copy_world_landmarks(&right_raw, 21);
  hands.hands[1].handedness = "Left";
  hands.hands[1].landmarks.resize(21);
  hands.hands[1].world_landmarks = sight::copy_world_landmarks(&left_raw, 21);
  hands.landmarks = hands.hands[0].landmarks;
  assert(hands.hands[0].world_landmarks.size() == 21);
  assert(hands.hands[1].world_landmarks.size() == 21);
  assert(hands.hands[0].world_landmarks[20].x < 0);
  assert(hands.hands[1].world_landmarks[20].x > 0);
  assert(sight::TrackingSignalBus::publish(
      hand_channel, publisher, sight::TrackingKind::Hand, hands, 640, 480));
  assert(sight::TrackingSignalBus::read(
      hand_channel, std::chrono::milliseconds(1000), signal));
  assert(signal.kind == sight::TrackingKind::Hand);
  assert(signal.data.world_landmarks.empty());
  assert(signal.data.hands.size() == 2);
  assert(signal.data.hands[0].handedness == "Right");
  assert(signal.data.hands[0].world_landmarks[5].x < 0);
  assert(signal.data.hands[1].world_landmarks[5].x > 0);
  signal.data.hands[1].world_landmarks[5].x = 999.f;
  assert(sight::TrackingSignalBus::read(
      hand_channel, std::chrono::milliseconds(1000), signal));
  assert(signal.data.hands[1].world_landmarks[5].x != 999.f);
  sight::TrackingSignalBus::clear_for_tests();
  std::cout << "PASS: real-meter world landmarks, confidence, invalid input, "
               "hand identity, bus deep copy\n";
}
