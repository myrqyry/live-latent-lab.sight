#include "vrm/vrm_tracking.hpp"
#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
using namespace sight;
using namespace sight::vrm;

namespace {
Asset rig() {
  Asset a;
  // Minimal authored humanoid with independent hands and finger chains.
  a.nodes.resize(32);
  a.bones["leftHand"] = 0;
  a.bones["rightHand"] = 1;
  int index = 2;
  for (auto side : {"left", "right"}) {
    const int wrist = side[0] == 'l' ? 0 : 1;
    for (auto finger : {"Index", "Middle", "Ring", "Little", "Thumb"}) {
      const char *segments[] = {"Proximal", "Intermediate", "Distal"};
      if (std::string(finger) == "Thumb") {
        segments[0] = "Metacarpal";
        segments[1] = "Proximal";
      }
      int parent = wrist;
      for (int k = 0; k < 3; ++k) {
        const auto name = std::string(side) + finger + segments[k];
        a.bones[name] = index;
        a.nodes[index].parent = parent;
        a.nodes[parent].children.push_back(index);
        // Distinct palm axes are required for genuine wrist orientation:
        // middle points forward, index/pinky spread across the palm.
        const float spread =
            std::string(finger) == "Index"
                ? -.09f
                : (std::string(finger) == "Little" ? .09f : 0.f);
        a.nodes[index].translation =
            k == 0 ? Vec3{spread, .12f, 0.f} : Vec3{0.f, .12f, 0.f};
        parent = index++;
      }
    }
  }
  return a;
}
HandData open_hand(std::string label) {
  HandData h;
  h.handedness = std::move(label);
  h.landmarks.resize(21);
  auto put = [&](int i, float x, float y, float z = 0.f) {
    h.landmarks[i] = {x, y, z, 1.f};
  };
  put(0, .5f, .8f);
  for (const auto &[base, x, y] :
       {std::tuple{5, .42f, .55f}, std::tuple{9, .5f, .53f},
        std::tuple{13, .58f, .55f}, std::tuple{17, .64f, .59f}}) {
    put(base, x, y);
    put(base + 1, x, y - .10f);
    put(base + 2, x, y - .20f);
    put(base + 3, x, y - .30f);
  }
  put(1, .40f, .70f);
  put(2, .36f, .63f);
  put(3, .32f, .56f);
  put(4, .28f, .49f);
  return h;
}
TrackingSignal signal(std::vector<HandData> hands) {
  TrackingSignal s;
  s.kind = TrackingKind::Hand;
  s.data.hands = std::move(hands);
  if (!s.data.hands.empty())
    s.data.landmarks = s.data.hands[0].landmarks;
  return s;
}
void curl_index(HandData &h) {
  h.landmarks[7] = {.47f, .52f, 0, 1};
  h.landmarks[8] = {.52f, .61f, 0, 1};
}
float difference(Quat a, Quat b) {
  return 1.f - std::fabs(a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w);
}
} // namespace
int main(int argc, char **argv) {
  const Asset a = rig();
  // RED regression: a positive local X curl is NOT a positive inward
  // finger bend on every authored humanoid. Mirrored wrists and an
  // anomalous index MCP can have distinct bone-space hinge signs.
  assert(finger_flexion_sign(a, "leftIndexIntermediate") > 0);
  assert(finger_flexion_sign(a, "rightIndexIntermediate") < 0);
  Asset flipped_axis = a;
  flipped_axis.nodes[a.bones.at("leftIndexProximal")].rotation =
      from_euler(0.f, 3.14159265f, 0.f);
  assert(finger_flexion_sign(flipped_axis, "leftIndexProximal") < 0);
  assert(finger_flexion_sign(a, "leftUpperLeg") == 0);
  // Optional real private VRM fixtures are passed by the developer
  // during validation; CTest does NOT require or distribute the files.
  if (argc == 3) {
    for (int model = 1; model <= 2; ++model) {
      Asset actual;
      std::string error;
      assert(load(argv[model], actual, error));
      for (const char *side : {"left", "right"}) {
        const std::string hand(side);
        const float index_root =
            finger_flexion_sign(actual, hand + "IndexProximal");
        const float index_mid =
            finger_flexion_sign(actual, hand + "IndexIntermediate");
        const float middle =
            finger_flexion_sign(actual, hand + "MiddleProximal");
        const float thumb =
            finger_flexion_sign(actual, hand + "ThumbMetacarpal");
        assert(index_root == (model == 1 ? -1.f : 1.f));
        assert(index_mid == -1.f);
        assert(middle == -1.f);
        assert(thumb == -1.f);
        // Every authored finger joint on the real avatar must have a
        // nondegenerate palm-facing local hinge; otherwise that digit
        // would silently lose motion with the safety fallback.
        for (const char *finger :
             {"Index", "Middle", "Ring", "Little", "Thumb"}) {
          for (int segment = 0; segment < 3; ++segment) {
            const std::string part = std::string(finger) == "Thumb"
                                         ? (segment == 0   ? "Metacarpal"
                                            : segment == 1 ? "Proximal"
                                                           : "Distal")
                                         : (segment == 0   ? "Proximal"
                                            : segment == 1 ? "Intermediate"
                                                           : "Distal");
            const auto finger_joint = hand + finger + part;
            const float sign = finger_flexion_sign(actual, finger_joint);
            assert(sign == 1.f || sign == -1.f);
          }
        }
        // End-to-end mapping: a genuinely bent MediaPipe finger must
        // produce the authored sign, not merely a correct helper value.
        HandPoseState isolated;
        const auto start_at =
            std::chrono::steady_clock::time_point(std::chrono::seconds(8));
        (void)ease_hand_pose(actual, nullptr, isolated, start_at, 120.f, false,
                             0.f, 700.f, false);
        auto curling =
            open_hand(side == std::string("left") ? "Left" : "Right");
        curl_index(curling);
        // A sideways knuckle segment also produces a detectable MCP bend.
        curling.landmarks[6] = {.57f, .53f, 0.f, 1.f};
        const auto bent_signal = signal({curling});
        const auto bent_pose =
            ease_hand_pose(actual, &bent_signal, isolated,
                           start_at + std::chrono::milliseconds(45), 120.f,
                           false, 0.f, 700.f, false, nullptr, true);
        assert(bent_pose.bone_rotation.contains(hand + "IndexProximal"));
        assert(bent_pose.bone_rotation.contains(hand + "IndexIntermediate"));
        assert(bent_pose.bone_rotation.at(hand + "IndexProximal").x *
                   index_root >
               .01f);
        assert(bent_pose.bone_rotation.at(hand + "IndexIntermediate").x *
                   index_mid >
               .01f);
        std::cout << "REAL_VRM_INWARD_FINGER_FLEXION "
                  << (model == 1 ? "Peepo " : "Otaku ") << side
                  << " indexMCP=" << index_root << " indexPIP=" << index_mid
                  << " middle=" << middle << " thumb=" << thumb << "\n";
      }
    }
  }
  using Clock = std::chrono::steady_clock;
  const auto t = Clock::time_point(std::chrono::seconds(2));
  HandPoseState state;
  const auto resting = ease_hand_pose(a, nullptr, state, t, 120.f, false);
  assert(resting.bone_rotation.empty()); // open hands, no signal
  auto left = open_hand("Left");
  auto right = open_hand("Right");
  curl_index(left);
  const auto input = signal({left, right});
  const auto initial = ease_hand_pose(
      a, &input, state, t + std::chrono::milliseconds(16), 120.f, false);
  assert(initial.bone_rotation.contains("leftIndexIntermediate"));
  assert(std::fabs(initial.bone_rotation.at("leftIndexIntermediate").x) > .02f);
  assert(!initial.bone_rotation.contains("rightIndexIntermediate"));
  assert(!initial.bone_rotation.contains("leftMiddleIntermediate"));
  // Hand detection order is unstable, handedness must determine destination.
  const auto reversed = signal({right, left});
  const auto later = ease_hand_pose(
      a, &reversed, state, t + std::chrono::milliseconds(80), 120.f, false);
  assert(difference(initial.bone_rotation.at("leftIndexIntermediate"),
                    later.bone_rotation.at("leftIndexIntermediate")) > .00001f);
  assert(!later.bone_rotation.contains("rightIndexIntermediate"));
  // Missing detection holds pose indefinitely rather than opening the fist.
  auto held = ease_hand_pose(a, nullptr, state, t + std::chrono::seconds(20),
                             120.f, false);
  assert(difference(held.bone_rotation.at("leftIndexIntermediate"),
                    later.bone_rotation.at("leftIndexIntermediate")) < .00001f);
  auto empty = signal({});
  held = ease_hand_pose(a, &empty, state, t + std::chrono::seconds(21), 120.f,
                        false);
  assert(difference(held.bone_rotation.at("leftIndexIntermediate"),
                    later.bone_rotation.at("leftIndexIntermediate")) < .00001f);
  // Reacquisition interpolates toward the newly open hand.
  auto both_open = signal({open_hand("Left"), open_hand("Right")});
  const auto eased = ease_hand_pose(a, &both_open, state,
                                    t + std::chrono::seconds(21) +
                                        std::chrono::milliseconds(16),
                                    120.f, false);
  assert(difference(eased.bone_rotation.at("leftIndexIntermediate"), Quat{}) <
         difference(held.bone_rotation.at("leftIndexIntermediate"), Quat{}));
  assert(difference(eased.bone_rotation.at("leftIndexIntermediate"), Quat{}) >
         .001f);
  // Mirrored camera setting is an explicit choice; sides are not inferred
  // from screen X or from hand detection order.
  HandPoseState flipped;
  (void)ease_hand_pose(a, nullptr, flipped, t, 120.f, true);
  const auto swap = ease_hand_pose(
      a, &input, flipped, t + std::chrono::milliseconds(16), 120.f, true);
  assert(swap.bone_rotation.contains("rightIndexIntermediate"));
  assert(!swap.bone_rotation.contains("leftIndexIntermediate"));
  // Regression: both Peepo and Otaku have genuine humanoid wrist bones,
  // but Sight never applies rotation to leftHand/rightHand. A tilt of the
  // palm must change the wrist, even when no finger bend is requested and
  // hand-driven shoulder IK is disabled.
  auto rotate_palm = [](HandData hand, float radians) {
    const float cx = hand.landmarks[0].x, cy = hand.landmarks[0].y;
    const float c = std::cos(radians), s = std::sin(radians);
    for (auto &p : hand.landmarks) {
      const float x = p.x - cx, y = p.y - cy;
      p.x = cx + c * x - s * y;
      p.y = cy + s * x + c * y;
    }
    return hand;
  };
  HandPoseState wrist_state;
  (void)ease_hand_pose(a, nullptr, wrist_state, t, 100.f, false, 0.f, 700.f,
                       true, nullptr, false);
  const auto open_pair = signal({open_hand("Left"), open_hand("Right")});
  const auto palm_initial = ease_hand_pose(
      a, &open_pair, wrist_state, t + std::chrono::milliseconds(16), 100.f,
      false, 0.f, 700.f, true, nullptr, false);
  assert(palm_initial.bone_rotation.contains("leftHand"));
  assert(palm_initial.bone_rotation.contains("rightHand"));
  assert(!palm_initial.bone_rotation.contains("leftIndexIntermediate"));
  const auto rotated_palm =
      signal({rotate_palm(open_hand("Left"), .65f), open_hand("Right")});
  const auto palm_turned = ease_hand_pose(
      a, &rotated_palm, wrist_state, t + std::chrono::milliseconds(116), 100.f,
      false, 0.f, 700.f, true, nullptr, false);
  assert(difference(palm_initial.bone_rotation.at("leftHand"),
                    palm_turned.bone_rotation.at("leftHand")) > .002f);
  assert(difference(palm_initial.bone_rotation.at("rightHand"),
                    palm_turned.bone_rotation.at("rightHand")) < .00001f);
  const auto palm_held =
      ease_hand_pose(a, nullptr, wrist_state, t + std::chrono::seconds(12),
                     100.f, false, 0.f, 700.f, true, nullptr, false);
  assert(difference(palm_turned.bone_rotation.at("leftHand"),
                    palm_held.bone_rotation.at("leftHand")) < .00001f);
  const auto palm_return = ease_hand_pose(
      a, nullptr, wrist_state,
      t + std::chrono::seconds(12) + std::chrono::milliseconds(100), 100.f,
      false, 350.f, 0.f, true, nullptr, false);
  assert(difference(palm_return.bone_rotation.at("leftHand"), Quat{}) <
         difference(palm_held.bone_rotation.at("leftHand"), Quat{}));
  // Opting out must completely suppress wrist rotation, not just stop
  // consuming fresh landmarks. Finger tracking remains independent.
  HandPoseState no_wrist;
  (void)ease_hand_pose(a, nullptr, no_wrist, t, 100.f, false, 0.f, 700.f,
                       false);
  const auto disabled_wrist = ease_hand_pose(a, &rotated_palm, no_wrist,
                                             t + std::chrono::milliseconds(48),
                                             100.f, false, 0.f, 700.f, false);
  assert(!disabled_wrist.bone_rotation.contains("leftHand"));
  assert(!disabled_wrist.bone_rotation.contains("rightHand"));
  // An invalid/edge-on palm cannot rotate the wrist, while other detected
  // fingers remain independent; never extrapolate from NaNs or zero vectors.
  HandPoseState missing_palm;
  (void)ease_hand_pose(a, nullptr, missing_palm, t, 100.f, false, 0.f, 700.f,
                       true);
  auto bad_palm = rotated_palm;
  bad_palm.data.hands[0].landmarks[9].confidence = .01f;
  bad_palm.data.hands[1].landmarks[9].confidence = .01f;
  const auto invalid_wrist = ease_hand_pose(a, &bad_palm, missing_palm,
                                            t + std::chrono::milliseconds(48),
                                            100.f, false, 0.f, 700.f, true);
  assert(!invalid_wrist.bone_rotation.contains("leftHand"));
  assert(!invalid_wrist.bone_rotation.contains("rightHand"));
  // Unknown handedness or nonfinite geometry never moves a finger.
  HandPoseState unknown;
  (void)ease_hand_pose(a, nullptr, unknown, t, 120.f, false);
  auto no_label = signal({open_hand("")});
  curl_index(no_label.data.hands[0]);
  assert(ease_hand_pose(a, &no_label, unknown,
                        t + std::chrono::milliseconds(16), 120.f, false)
             .bone_rotation.empty());
  HandPoseState invalid;
  (void)ease_hand_pose(a, nullptr, invalid, t, 120.f, false);
  auto bad = input;
  bad.data.hands[0].landmarks[7].x = std::numeric_limits<float>::quiet_NaN();
  assert(!ease_hand_pose(a, &bad, invalid, t + std::chrono::milliseconds(16),
                         120.f, false)
              .bone_rotation.contains("leftIndexIntermediate"));
  auto no_rig = a;
  no_rig.bones.erase("leftIndexIntermediate");
  HandPoseState missing;
  (void)ease_hand_pose(no_rig, nullptr, missing, t, 120.f, false);
  assert(!ease_hand_pose(no_rig, &input, missing,
                         t + std::chrono::milliseconds(16), 120.f, false)
              .bone_rotation.contains("leftIndexIntermediate"));
  std::cout
      << "PASS: two-hand named finger articulation with dropout holding, eased "
         "reopening, left/right independence and mirrored label option\n";
}
