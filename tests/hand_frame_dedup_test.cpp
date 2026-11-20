#include "core/hand_frame_dedup.hpp"
#include <cassert>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <vector>

using Clock = std::chrono::steady_clock;
int main() {
  sight::HandFrameDedup gate;
  const auto start = Clock::time_point(std::chrono::seconds(5));
  auto a = std::vector<uint8_t>(640 * 360 * 4, 42);
  auto b = a;
  b[b.size() - 1] ^= 0xff;
  // First valid frame must be delivered immediately.
  assert(gate.should_infer(a, 640, 360, start));
  gate.accept(std::move(a), 640, 360, start);
  auto identical = std::vector<uint8_t>(640 * 360 * 4, 42);
  assert(!gate.should_infer(identical, 640, 360,
                            start + std::chrono::milliseconds(16)));
  assert(!gate.should_infer(identical, 640, 360,
                            start + std::chrono::milliseconds(233)));
  // Periodic heartbeat prevents a perfectly motionless frame from aging out
  // of the VRM receiver's one-second signal freshness window.
  assert(gate.should_infer(identical, 640, 360,
                           start + std::chrono::milliseconds(250)));
  gate.accept(std::move(identical), 640, 360,
              start + std::chrono::milliseconds(250));
  // Even one changed byte is a new rendered image, not a duplicate.
  assert(
      gate.should_infer(b, 640, 360, start + std::chrono::milliseconds(251)));
  gate.accept(std::move(b), 640, 360, start + std::chrono::milliseconds(251));
  auto changed = std::vector<uint8_t>(640 * 360 * 4, 42);
  assert(gate.should_infer(changed, 640, 360,
                           start + std::chrono::milliseconds(266)));
  gate.accept(std::move(changed), 640, 360,
              start + std::chrono::milliseconds(266));
  // A different resolution is never compared to the previous buffer.
  auto small = std::vector<uint8_t>(320 * 180 * 4, 42);
  assert(gate.should_infer(small, 320, 180,
                           start + std::chrono::milliseconds(267)));
  gate.accept(std::move(small), 320, 180,
              start + std::chrono::milliseconds(267));
  assert(!gate.should_infer(std::vector<uint8_t>(320 * 180 * 4, 42), 320, 180,
                            start + std::chrono::milliseconds(270)));
  gate.reset();
  assert(gate.should_infer(std::vector<uint8_t>(320 * 180 * 4, 42), 320, 180,
                           start + std::chrono::milliseconds(270)));
  // 60 OBS renders/sec with a new camera image every two renders:
  // accept the first of each pair, not every OBS render.
  sight::HandFrameDedup camera30;
  unsigned accepted = 0, skipped = 0;
  for (int render = 0; render < 60; ++render) {
    auto pixels = std::vector<uint8_t>(32 * 32 * 4, uint8_t(render / 2));
    const auto at = start + std::chrono::milliseconds(render * 16);
    if (camera30.should_infer(pixels, 32, 32, at)) {
      ++accepted;
      camera30.accept(std::move(pixels), 32, 32, at);
    } else {
      ++skipped;
    }
  }
  assert(accepted == 30 && skipped == 30);
  // A stationary capture remains alive at a bounded heartbeat, rather
  // than continually performing expensive model inference or expiring.
  sight::HandFrameDedup static_camera;
  unsigned static_accepted = 0;
  for (int render = 0; render < 60; ++render) {
    auto pixels = std::vector<uint8_t>(32 * 32 * 4, 17);
    const auto at = start + std::chrono::milliseconds(render * 16);
    if (static_camera.should_infer(pixels, 32, 32, at)) {
      ++static_accepted;
      static_camera.accept(std::move(pixels), 32, 32, at);
    }
  }
  assert(static_accepted >= 3 && static_accepted <= 5);
  // Holistic needs three initial observations for its stability preset.
  sight::TrackingFrameDedup holistic(3);
  for (int i=0;i<4;++i) {
    auto pixels=std::vector<uint8_t>(32*32*4,17);
    const auto at=start+std::chrono::milliseconds(16*i);
    assert(holistic.should_infer(pixels,32,32,at)==(i<3));
    if(i<3)holistic.accept(std::move(pixels),32,32,at);
  }
  holistic.reset();
  assert(holistic.should_infer(std::vector<uint8_t>(32*32*4,17),32,32,start));
  // Zero-size or malformed buffers cannot silently count as a valid frame.
  assert(!gate.should_infer({}, 0, 0, start));
  assert(!gate.should_infer({1, 2, 3}, 100, 100, start));
  std::cout << "PASS exact render-byte dedup, changed frame, heartbeat, "
               "dimensions, reset\n";
}
