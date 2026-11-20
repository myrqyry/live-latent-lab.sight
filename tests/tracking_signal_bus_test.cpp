#include "core/tracking_signal_bus.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <thread>
namespace {
// Registered BEFORE first bus use. C++ static destructors registered later
// would otherwise destroy the bus before this late exit-time consumer.
void verify_exit_time_bus() {
  sight::TrackingData points;
  points.landmarks.resize(1);
  sight::TrackingSignal sample;
  if (!sight::TrackingSignalBus::publish(
          "exit/face", 77, sight::TrackingKind::Face, points, 1, 1) ||
      !sight::TrackingSignalBus::read("exit/face",
                                      std::chrono::milliseconds(100), sample) ||
      sample.data.landmarks.size() != 1)
    std::_Exit(90);
  std::fputs("SIGHT_TRACKING_BUS_EXIT_SAFE\n", stdout);
  std::fflush(stdout);
}
} // namespace
int main() {
  if (std::atexit(verify_exit_time_bus) != 0)
    return 40;
  using namespace std::chrono_literals;
  sight::TrackingSignalBus::clear_for_tests();
  const auto a = sight::TrackingSignalBus::new_publisher_id();
  const auto b = sight::TrackingSignalBus::new_publisher_id();
  if (!a || a == b)
    return 1;
  sight::TrackingData face;
  face.landmarks.resize(478);
  face.landmarks[4].x = 0.23f;
  face.landmarks[4].y = 0.76f;
  face.blendshapes = {.12f, .43f};
  face.has_head_transform = true;
  face.head_transform[12] = .6f;
  if (!sight::TrackingSignalBus::publish(
          "camera/face", a, sight::TrackingKind::Face, face, 640, 360))
    return 2;
  sight::TrackingSignal result;
  if (!sight::TrackingSignalBus::read("camera/face", 300ms, result) ||
      result.publisher != a || result.data.landmarks.size() != 478 ||
      result.data.landmarks[4].x != .23f ||
      result.data.blendshapes[1] != .43f ||
      result.data.head_transform[12] != .6f || result.source_width != 640 ||
      result.source_height != 360)
    return 3;
  const auto sequence = result.sequence;
  // Immutable copies: local consumer changes do not affect publisher.
  result.data.landmarks[4].x = 1.0f;
  if (!sight::TrackingSignalBus::read("camera/face", 300ms, result) ||
      result.data.landmarks[4].x != .23f)
    return 4;
  // Duplicate publishers cannot silently hijack existing channels.
  if (sight::TrackingSignalBus::publish(
          "camera/face", b, sight::TrackingKind::Body, {}, 640, 360))
    return 5;
  sight::TrackingSignalBus::withdraw("camera/face", b);
  if (!sight::TrackingSignalBus::read("camera/face", 300ms, result))
    return 6;
  if (sight::TrackingSignalBus::read("camera/face", 0ms, result))
    return 7;
  std::this_thread::sleep_for(4ms);
  if (sight::TrackingSignalBus::read("camera/face", 1ms, result))
    return 8;
  // A no-face result is explicitly published, not replaced with stale eyes.
  if (!sight::TrackingSignalBus::publish(
          "camera/face", a, sight::TrackingKind::Face, {}, 640, 360))
    return 9;
  if (!sight::TrackingSignalBus::read("camera/face", 300ms, result) ||
      !result.data.landmarks.empty() || result.sequence <= sequence)
    return 10;
  sight::TrackingSignalBus::withdraw("camera/face", a);
  if (sight::TrackingSignalBus::read("camera/face", 300ms, result))
    return 11;
  if (!sight::TrackingSignalBus::publish(
          "camera/face", b, sight::TrackingKind::Body, {}, 640, 360))
    return 12;
  sight::TrackingSignalBus::clear_for_tests();
  if (sight::TrackingSignalBus::read("camera/face", 300ms, result))
    return 13;
  sight::TrackingData hands;
  hands.hands.resize(2);
  hands.hands[0].landmarks.resize(21);
  hands.hands[0].gesture = "Victory";
  hands.hands[0].gesture_score = .92f;
  hands.hands[0].pinch = .7f;
  hands.hands[1].landmarks.resize(21);
  hands.landmarks = hands.hands[0].landmarks;
  if (!sight::TrackingSignalBus::publish(
          "camera/hand", b, sight::TrackingKind::Hand, hands, 512, 512))
    return 14;
  if (!sight::TrackingSignalBus::read("camera/hand", 300ms, result) ||
      result.data.hands.size() != 2 ||
      result.data.hands[0].gesture != "Victory" ||
      result.data.hands[0].pinch != .7f)
    return 15;
  std::cout << "PASS: timestamped face/body signal bus has independent copies, "
               "expiry,"
               " explicit no-subject, and single-owner channels\n";
}
