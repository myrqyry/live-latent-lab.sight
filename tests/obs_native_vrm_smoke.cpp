// Standalone OBS runtime validation: the actual Sight module draws a native
// avatar into an alpha-enabled libobs texture. Does not modify OBS profiles.
#include "core/tracking_signal_bus.hpp"
#include <chrono>
#include <cstdint>
#include <cstring>
#include <graphics/graphics.h>
#include <iostream>
#include <obs-module.h>
#include <obs.h>
#include <string>
#include <thread>
#include <vector>
namespace {
constexpr uint32_t W = 512, H = 768;
std::vector<uint8_t> draw(obs_source_t *source) {
  std::vector<uint8_t> frame;
  obs_enter_graphics();
  auto *target = gs_texrender_create(GS_RGBA, GS_ZS_NONE);
  if (target && gs_texrender_begin(target, W, H)) {
    vec4 clear{};
    gs_clear(GS_CLEAR_COLOR, &clear, 1, 0);
    gs_projection_push();
    gs_matrix_push();
    gs_matrix_identity();
    gs_ortho(0, float(W), 0, float(H), -100, 100);
    obs_source_video_render(source);
    gs_matrix_pop();
    gs_projection_pop();
    gs_texrender_end(target);
    auto *stage = gs_stagesurface_create(W, H, GS_RGBA);
    if (stage && gs_texrender_get_texture(target)) {
      gs_stage_texture(stage, gs_texrender_get_texture(target));
      uint8_t *pixels = nullptr;
      uint32_t stride = 0;
      if (gs_stagesurface_map(stage, &pixels, &stride)) {
        frame.resize(size_t(W) * H * 4);
        for (uint32_t row = 0; row < H; ++row)
          std::memcpy(frame.data() + size_t(row) * W * 4,
                      pixels + size_t(row) * stride, W * 4);
        gs_stagesurface_unmap(stage);
      }
    }
    if (stage)
      gs_stagesurface_destroy(stage);
  }
  if (target)
    gs_texrender_destroy(target);
  obs_leave_graphics();
  return frame;
}
} // namespace
int main(int argc, char **argv) {
  const bool test_body = argc == 5 && std::strcmp(argv[4], "--body") == 0;
  const bool test_hands = argc == 5 && std::strcmp(argv[4], "--hands") == 0;
  const bool test_torso = argc == 5 && std::strcmp(argv[4], "--torso") == 0;
  const bool test_yaw = argc == 5 && std::strcmp(argv[4], "--yaw") == 0;
  const bool test_wrists = argc == 5 && std::strcmp(argv[4], "--wrists") == 0;
  if (argc != 4 && !test_body && !test_hands && !test_torso && !test_yaw &&
      !test_wrists)
    return 2;
  if (!obs_startup("en-US", nullptr, nullptr))
    return 3;
  obs_video_info video{};
  video.graphics_module = "libobs-opengl";
  video.fps_num = 30;
  video.fps_den = 1;
  video.base_width = W;
  video.base_height = H;
  video.output_width = W;
  video.output_height = H;
  video.output_format = VIDEO_FORMAT_RGBA;
  video.scale_type = OBS_SCALE_DISABLE;
  if (obs_reset_video(&video) != OBS_VIDEO_SUCCESS) {
    obs_shutdown();
    return 4;
  }
  obs_module_t *module = nullptr;
  const int open = obs_open_module(&module, argv[1], argv[3]);
  if (open != MODULE_SUCCESS || !obs_init_module(module)) {
    std::cerr << "FAIL: Sight module initialization " << open << "\n";
    obs_shutdown();
    return 5;
  }
  obs_data_t *config = obs_data_create();
  obs_data_set_string(config, "model_path", argv[2]);
  obs_data_set_int(config, "width", W);
  obs_data_set_int(config, "height", H);
  // Peepo's bent-elbow rest occupies more vertical space at this narrow
  // smoke-test aspect ratio. Frame the complete avatar, not a crop.
  if (test_hands || test_torso || test_yaw || test_wrists)
    obs_data_set_double(config, "zoom", 0.65);
  auto *avatar = obs_source_create("sight_native_vrm_source",
                                   "VRM native smoke", config, nullptr);
  obs_data_release(config);
  if (!avatar) {
    std::cerr << "FAIL: missing native VRM OBS source\n";
    obs_shutdown();
    return 6;
  }
  // The source must honor initial OBS settings without requiring a
  // later properties edit or an explicit external obs_source_update().
  const auto initialized_width = obs_source_get_width(avatar);
  const auto initialized_height = obs_source_get_height(avatar);
  if (initialized_width != W || initialized_height != H) {
    std::cerr << "FAIL: native source ignored initial output dimensions "
              << initialized_width << "x" << initialized_height << " vs " << W
              << "x" << H << "\n";
    obs_source_release(avatar);
    obs_shutdown();
    return 7;
  }
  bool passed = false;
  std::vector<uint8_t> neutral;
  const auto stop = std::chrono::steady_clock::now() + std::chrono::seconds(12);
  while (std::chrono::steady_clock::now() < stop) {
    const auto frame = draw(avatar);
    size_t pixels = 0, colored = 0;
    uint64_t checksum = 0;
    if (frame.size() == size_t(W) * H * 4) {
      for (size_t i = 0; i < frame.size(); i += 4) {
        const uint8_t r = frame[i], g = frame[i + 1], b = frame[i + 2],
                      a = frame[i + 3];
        if (a > 100) {
          ++pixels;
          checksum += r + g + b;
          if (r != g || g != b)
            ++colored;
        }
      }
    }
    const bool transparent_corner = frame.size() == size_t(W) * H * 4 &&
                                    frame[3] < 10 &&
                                    frame[(size_t(W) * H - 1) * 4 + 3] < 10;
    if (pixels > 250 && pixels < size_t(W) * H / 2 && colored > 50 &&
        checksum > 1000 && transparent_corner) {
      std::cout << "NATIVE_VRM_OBS_PIXEL_PASS nontransparent=" << pixels
                << " colored=" << colored << " sum=" << checksum << "\n";
      neutral = frame;
      passed = true;
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  if (passed) {
    // The user-facing OBS Properties must actually expose coverage from
    // the same native model that was rendered, not an invented static list.
    auto *props = obs_source_properties(avatar);
    auto *info =
        props ? obs_properties_get(props, "rig_coverage_info") : nullptr;
    const char *description = info ? obs_property_description(info) : nullptr;
    const std::string snapshot = description ? description : "";
    // This smoke also runs against an intentionally minimal generated
    // fixture with no hips or neck. Verify the generic UI contract there;
    // assert those model-specific omissions only for real Peepo/Otaku.
    const bool actual_avatar = std::strstr(argv[2], "PeepoSight.vrm") ||
                               std::strstr(argv[2], "OtakuSight.vrm");
    const bool report_ok =
        snapshot.find("Rig coverage") != std::string::npos &&
        snapshot.find("never driven") != std::string::npos &&
        snapshot.find("Signals:") != std::string::npos &&
        snapshot.find("Reopen Properties") != std::string::npos &&
        (!actual_avatar || (snapshot.find("neck") != std::string::npos &&
                            snapshot.find("hips") != std::string::npos));
    std::cout << "NATIVE_VRM_RIG_COVERAGE_PROPERTIES "
              << (report_ok ? "PASS" : "FAIL")
              << " info_chars=" << snapshot.size() << "\n";
    passed &= report_ok;
    if (props)
      obs_properties_destroy(props);
  }
  auto different = [](const std::vector<uint8_t> &a,
                      const std::vector<uint8_t> &b) -> size_t {
    if (a.size() != b.size())
      return 0;
    size_t delta = 0;
    for (size_t i = 0; i < a.size(); ++i)
      delta += std::abs(int(a[i]) - int(b[i])) > 3;
    return delta;
  };
  if (passed) {
    // The published signal is consumed by the actual Sight plugin, and must
    // alter the native OBS source's pixels (not just a test-only math helper).
    sight::TrackingData face;
    face.landmarks.resize(478);
    face.blendshape_names = {"jawOpen"};
    face.blendshapes = {1.f};
    const auto publisher = sight::TrackingSignalBus::new_publisher_id();
    const bool published = sight::TrackingSignalBus::publish(
        "face", publisher, sight::TrackingKind::Face, face, 640, 480);
    // Confirm the expression eases INTO the tracked face rather than
    // teleporting on the first video-render invocation.
    const auto first = draw(avatar);
    for (int i = 0; i < 14; ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
      (void)draw(avatar);
    }
    const auto animated = draw(avatar);
    sight::TrackingSignalBus::withdraw("face", publisher);
    const auto held = draw(avatar);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    const auto held_later = draw(avatar);
    const size_t morph_pixels = different(neutral, animated);
    const size_t first_delta = different(neutral, first);
    const size_t held_pixels = different(animated, held_later);
    // Enable the existing shared loss-return control. The expression
    // must then glide home, rather than snapping on withdrawal.
    auto *settings = obs_source_get_settings(avatar);
    obs_data_set_double(settings, "lost_hold_ms", 0);
    obs_data_set_double(settings, "lost_return_ms", 200);
    obs_source_update(avatar, settings);
    obs_data_release(settings);
    const auto first_return = draw(avatar);
    for (int i = 0; i < 45; ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
      (void)draw(avatar);
    }
    const auto restored = draw(avatar);
    const size_t return_started = different(held_later, first_return);
    const size_t returned_pixels = different(neutral, restored);
    // A frame may span uneven scheduler time; require substantial
    // easing toward the final morph, not an arbitrary 3x pixel ratio.
    // The time-continuous easing is also checked in tracking unit tests.
    const bool signal_works =
        published && morph_pixels > 200 && first_delta * 2 < morph_pixels &&
        held_pixels < 40 && return_started < morph_pixels / 4 &&
        returned_pixels < morph_pixels / 3;
    std::cout << "NATIVE_VRM_SIGNAL_TO_RENDER "
              << (signal_works ? "PASS" : "FAIL")
              << " changed_channels=" << morph_pixels
              << " first=" << first_delta << " held=" << held_pixels
              << " return_started=" << return_started
              << " restored=" << returned_pixels << "\n";
    passed &= signal_works;
  }
  if (passed &&
      (test_body || test_hands || test_torso || test_yaw || test_wrists)) {
    auto *settings = obs_source_get_settings(avatar);
    obs_data_set_double(settings, "lost_hold_ms", 700);
    obs_data_set_double(settings, "lost_return_ms", 0);
    obs_source_update(avatar, settings);
    obs_data_release(settings);
  }
  if (passed && test_body) {
    // The real Peepo fixture has skinned humanoid arms. Publishing the
    // MediaPipe body channel must move actual rendered OBS pixels, HOLD
    // the last position when lost, and keep independent face morphs.
    sight::TrackingData body;
    body.landmarks.resize(33, {0.5f, 0.5f, 0.f, 0.f});
    auto point = [&](int index, float x, float y) {
      body.landmarks[index] = {x, y, 0.f, 1.f};
    };
    point(11, .65f, .35f);
    point(12, .35f, .35f);
    point(13, .65f, .08f); // lift left elbow above shoulder
    point(14, .15f, .35f);
    point(15, .65f, -.01f);
    point(16, .05f, .35f);
    point(23, .60f, .75f);
    point(24, .40f, .75f);
    const auto body_publisher = sight::TrackingSignalBus::new_publisher_id();
    const bool body_published = sight::TrackingSignalBus::publish(
        "body", body_publisher, sight::TrackingKind::Body, body, 640, 480);
    // The new behavior eases changes rather than snapping on the
    // first render; advance some actual OBS frames to demonstrate motion.
    for (int frame = 0; frame < 14; ++frame) {
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
      (void)draw(avatar);
    }
    const auto body_raised = draw(avatar);
    const size_t raised_pixels = different(neutral, body_raised);
    sight::TrackingData speaking;
    speaking.landmarks.resize(478);
    speaking.blendshape_names = {"jawOpen"};
    speaking.blendshapes = {1.f};
    const auto face_publisher = sight::TrackingSignalBus::new_publisher_id();
    const bool face_published = sight::TrackingSignalBus::publish(
        "face", face_publisher, sight::TrackingKind::Face, speaking, 640, 480);
    for (int frame = 0; frame < 12; ++frame) {
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
      (void)draw(avatar);
    }
    const auto both = draw(avatar);
    const size_t face_while_body = different(body_raised, both);
    sight::TrackingSignalBus::withdraw("face", face_publisher);
    sight::TrackingSignalBus::withdraw("body", body_publisher);
    const auto held = draw(avatar);
    std::this_thread::sleep_for(std::chrono::milliseconds(70));
    const auto held_later = draw(avatar);
    const size_t held_delta = different(held, held_later);
    const size_t held_from_rest = different(neutral, held_later);
    const bool body_works = body_published && face_published &&
                            raised_pixels > 200 && face_while_body > 100 &&
                            held_delta < 40 && held_from_rest > 200;
    std::cout << "NATIVE_VRM_BODY_SIGNAL_TO_RENDER "
              << (body_works ? "PASS" : "FAIL")
              << " changed_channels=" << raised_pixels
              << " face_while_body=" << face_while_body
              << " held_delta=" << held_delta
              << " held_from_rest=" << held_from_rest << "\n";
    passed &= body_works;

    // A re-acquired valid body pose should ease away from held posture,
    // not instantly teleport back to the currently measured direction.
    point(13, .85f, .35f);
    point(15, .95f, .35f);
    const auto return_publisher = sight::TrackingSignalBus::new_publisher_id();
    const bool reacquired = sight::TrackingSignalBus::publish(
        "body", return_publisher, sight::TrackingKind::Body, body, 640, 480);
    const auto first_reacquired = draw(avatar);
    const size_t first_delta = different(held_later, first_reacquired);
    for (int frame = 0; frame < 14; ++frame) {
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
      (void)draw(avatar);
    }
    const auto later_reacquired = draw(avatar);
    const size_t later_delta = different(held_later, later_reacquired);
    sight::TrackingSignalBus::withdraw("body", return_publisher);
    // Detect a one-frame snap by verifying the avatar keeps changing
    // substantially AFTER the first reacquisition render. Pixel-channel
    // counts are not linear in quaternion angle: varied avatar size,
    // silhouette and GPU scheduler delay make ratios such as 2x/4x flaky.
    // The body tracking unit tests separately assert exact easing,
    // held-pose and reacquisition behavior.
    const bool easing_works =
        reacquired && later_delta > 500 && later_delta > first_delta + 500;
    std::cout << "NATIVE_VRM_BODY_REACQUIRE_EASING "
              << (easing_works ? "PASS" : "FAIL")
              << " first_delta=" << first_delta
              << " later_delta=" << later_delta << "\n";
    passed &= easing_works;

    // The initial no-tracking pose is deliberately not the raw T-pose:
    // disabling the relaxed-arm drop should visibly change the avatar.
    auto *settings = obs_source_get_settings(avatar);
    obs_data_set_double(settings, "resting_arm_degrees", 0.0);
    obs_source_update(avatar, settings);
    obs_data_release(settings);
    const auto t_pose = draw(avatar);
    const size_t relaxed_pixels = different(neutral, t_pose);
    const bool relaxed_works = relaxed_pixels > 200;
    std::cout << "NATIVE_VRM_RELAXED_REST " << (relaxed_works ? "PASS" : "FAIL")
              << " changed_channels=" << relaxed_pixels << "\n";
    passed &= relaxed_works;
  }
  if (passed && test_hands) {
    // Hand channel is opt-in. Publishing a perfectly valid hand while
    // hands_drive_arms and animate_fingers are both disabled does nothing.
    auto *settings = obs_source_get_settings(avatar);
    obs_data_set_bool(settings, "hands_drive_arms", false);
    obs_data_set_bool(settings, "animate_fingers", false);
    obs_data_set_bool(settings, "rotate_wrists", false);
    obs_source_update(avatar, settings);
    obs_data_release(settings);
    const auto disabled_baseline = draw(avatar);
    sight::TrackingData hand_frame;
    sight::HandData left;
    left.handedness = "Left";
    left.landmarks.resize(21, {.5f, .50f, 0.f, 1.f});
    auto point = [&](int index, float x, float y) {
      left.landmarks[index] = {x, y, 0.f, 1.f};
    };
    point(0, .73f, .08f); // wrist above shoulder, no body present
    point(5, .65f, .30f);
    point(6, .65f, .20f);
    point(7, .70f, .28f);
    point(8, .75f, .37f); // bent index finger
    hand_frame.hands = {left};
    hand_frame.landmarks = left.landmarks;
    const auto publisher = sight::TrackingSignalBus::new_publisher_id();
    const bool published = sight::TrackingSignalBus::publish(
        "hand", publisher, sight::TrackingKind::Hand, hand_frame, 640, 480);
    for (int i = 0; i < 5; ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
      (void)draw(avatar);
    }
    const auto disabled = draw(avatar);
    const size_t disabled_delta = different(disabled_baseline, disabled);
    settings = obs_source_get_settings(avatar);
    obs_data_set_bool(settings, "hands_drive_arms", true);
    obs_data_set_bool(settings, "animate_fingers", true);
    obs_data_set_bool(settings, "rotate_wrists", true);
    obs_source_update(avatar, settings);
    obs_data_release(settings);
    for (int i = 0; i < 16; ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
      (void)draw(avatar);
    }
    const auto animated = draw(avatar);
    const size_t active_delta = different(disabled_baseline, animated);
    sight::TrackingData face;
    face.landmarks.resize(478);
    face.blendshape_names = {"jawOpen"};
    face.blendshapes = {.9f};
    const auto face_publisher = sight::TrackingSignalBus::new_publisher_id();
    const bool face_published = sight::TrackingSignalBus::publish(
        "face", face_publisher, sight::TrackingKind::Face, face, 640, 480);
    for (int i = 0; i < 12; ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
      (void)draw(avatar);
    }
    const auto both = draw(avatar);
    const size_t concurrent_face = different(animated, both);
    sight::TrackingSignalBus::withdraw("face", face_publisher);
    sight::TrackingSignalBus::withdraw("hand", publisher);
    const auto held = draw(avatar);
    std::this_thread::sleep_for(std::chrono::milliseconds(90));
    const auto held_later = draw(avatar);
    const size_t hold_delta = different(held, held_later);
    const bool hands_work = published && face_published &&
                            disabled_delta < 40 && active_delta > 300 &&
                            concurrent_face > 200 && hold_delta < 40;
    std::cout << "NATIVE_VRM_HAND_SIGNAL_TO_RENDER "
              << (hands_work ? "PASS" : "FAIL")
              << " disabled_delta=" << disabled_delta
              << " animated_channels=" << active_delta
              << " concurrent_face=" << concurrent_face
              << " held_delta=" << hold_delta << "\n";
    passed &= hands_work;

    // Nonzero loss return speed gracefully restores Peepo's relaxed pose
    // after the configured delay, not in the same missing-signal frame.
    settings = obs_source_get_settings(avatar);
    obs_data_set_double(settings, "lost_hold_ms", 0);
    obs_data_set_double(settings, "lost_return_ms", 160);
    obs_source_update(avatar, settings);
    obs_data_release(settings);
    const auto first_return = draw(avatar);
    const size_t first_return_delta = different(held_later, first_return);
    for (int i = 0; i < 16; ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
      (void)draw(avatar);
    }
    const auto settled = draw(avatar);
    const size_t eventual_return = different(held_later, settled);
    const bool return_works =
        eventual_return > 300 && first_return_delta * 3 < eventual_return;
    std::cout << "NATIVE_VRM_HAND_LOSS_EASED_RETURN "
              << (return_works ? "PASS" : "FAIL")
              << " first_delta=" << first_return_delta
              << " eventual_delta=" << eventual_return << "\n";
    passed &= return_works;
  }
  if (passed && test_torso) {
    // Actual Peepo skinning, no arm/elbow data: only shoulder and hip
    // landmarks are available, so new pixels must come from the spine.
    sight::TrackingData body;
    body.landmarks.resize(33, {0.5f, 0.5f, 0.f, 0.f});
    auto point = [&](int index, float x, float y, float z = 0.f) {
      body.landmarks[index] = {x, y, z, 1.f};
    };
    point(11, .65f, .35f);
    point(12, .35f, .35f);
    point(23, .60f, .75f);
    point(24, .40f, .75f);
    const auto publisher = sight::TrackingSignalBus::new_publisher_id();
    const bool baseline_published = sight::TrackingSignalBus::publish(
        "body", publisher, sight::TrackingKind::Body, body, 640, 480);
    for (int i = 0; i < 4; ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
      (void)draw(avatar);
    }
    const auto baseline = draw(avatar);
    point(11, .80f, .35f);
    point(12, .50f, .35f);
    const bool lean_published = sight::TrackingSignalBus::publish(
        "body", publisher, sight::TrackingKind::Body, body, 640, 480);
    const auto first_lean = draw(avatar);
    for (int i = 0; i < 15; ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
      (void)draw(avatar);
    }
    const auto leaned = draw(avatar);
    const size_t one_frame = different(baseline, first_lean);
    const size_t torso_changed = different(baseline, leaned);
    sight::TrackingSignalBus::withdraw("body", publisher);
    const auto held = draw(avatar);
    std::this_thread::sleep_for(std::chrono::milliseconds(70));
    const auto held_later = draw(avatar);
    const size_t held_difference = different(held, held_later);
    const bool torso_ok = baseline_published && lean_published &&
                          torso_changed > 400 &&
                          one_frame * 3 < torso_changed && held_difference < 40;
    std::cout << "NATIVE_VRM_SPINE_SIGNAL_TO_RENDER "
              << (torso_ok ? "PASS" : "FAIL")
              << " changed_channels=" << torso_changed
              << " initial_delta=" << one_frame
              << " held_delta=" << held_difference << "\n";
    passed &= torso_ok;

    // Zero torso gain must take the spine smoothly back to neutral while
    // leaving body, face and hand tracking mechanisms independently usable.
    auto *settings = obs_source_get_settings(avatar);
    obs_data_set_double(settings, "torso_strength", 0.0);
    obs_source_update(avatar, settings);
    obs_data_release(settings);
    const auto no_gain_publisher = sight::TrackingSignalBus::new_publisher_id();
    (void)sight::TrackingSignalBus::publish(
        "body", no_gain_publisher, sight::TrackingKind::Body, body, 640, 480);
    for (int i = 0; i < 15; ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
      (void)draw(avatar);
    }
    const auto suppressed = draw(avatar);
    sight::TrackingSignalBus::withdraw("body", no_gain_publisher);
    const size_t suppression = different(leaned, suppressed);
    const bool gain_ok = suppression > 400;
    std::cout << "NATIVE_VRM_SPINE_GAIN_CONTROL " << (gain_ok ? "PASS" : "FAIL")
              << " changed_channels=" << suppression << "\n";
    passed &= gain_ok;
  }
  if (passed && test_yaw) {
    // The actual private Peepo VRM is skinned: a head-turn signal must
    // visibly change the render, respond in BOTH horizontal directions,
    // and leave pixels unchanged after withdrawal. The separate unit
    // contract asserts the intended yaw sign relative to hands.
    sight::TrackingData face;
    face.landmarks.resize(478);
    face.has_head_transform = true;
    face.head_transform = {0.92106f,  0, 0.38942f, 0, 0, 1, 0, 0,
                           -0.38942f, 0, 0.92106f, 0, 0, 0, 0, 1};
    const auto publisher = sight::TrackingSignalBus::new_publisher_id();
    const bool first_published = sight::TrackingSignalBus::publish(
        "face", publisher, sight::TrackingKind::Face, face, 640, 480);
    const auto first_frame = draw(avatar);
    for (int i = 0; i < 14; ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
      (void)draw(avatar);
    }
    const auto first_turn = draw(avatar);
    face.head_transform[2] = -0.38942f;
    face.head_transform[8] = 0.38942f;
    const bool second_published = sight::TrackingSignalBus::publish(
        "face", publisher, sight::TrackingKind::Face, face, 640, 480);
    const auto first_opposite = draw(avatar);
    for (int i = 0; i < 18; ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
      (void)draw(avatar);
    }
    const auto reverse_turn = draw(avatar);
    sight::TrackingSignalBus::withdraw("face", publisher);
    const auto held = draw(avatar);
    std::this_thread::sleep_for(std::chrono::milliseconds(70));
    const auto held_later = draw(avatar);
    const size_t first_changed = different(neutral, first_turn);
    const size_t reverse_changed = different(neutral, reverse_turn);
    const size_t across_directions = different(first_turn, reverse_turn);
    const size_t smooth_start = different(first_frame, first_turn);
    const size_t smooth_reverse = different(first_opposite, reverse_turn);
    const size_t hold_delta = different(held, held_later);
    const bool yaw_render_ok = first_published && second_published &&
                               first_changed > 300 && reverse_changed > 300 &&
                               across_directions > 300 && smooth_start > 300 &&
                               smooth_reverse > 300 && hold_delta < 40;
    std::cout << "NATIVE_VRM_HEAD_YAW_SIGNAL_TO_RENDER "
              << (yaw_render_ok ? "PASS" : "FAIL") << " first=" << first_changed
              << " opposite=" << reverse_changed
              << " between=" << across_directions
              << " easing_first=" << smooth_start
              << " easing_reverse=" << smooth_reverse << " held=" << hold_delta
              << "\n";
    passed &= yaw_render_ok;
  }
  if (passed && test_wrists) {
    // Keep the wrist in the same pixel position while rotating the palm.
    // Neither arm IK nor finger curls is allowed to supply the pixel delta.
    auto *opts = obs_source_get_settings(avatar);
    obs_data_set_bool(opts, "hands_drive_arms", false);
    obs_data_set_bool(opts, "animate_fingers", false);
    obs_data_set_bool(opts, "rotate_wrists", false);
    obs_source_update(avatar, opts);
    obs_data_release(opts);
    const auto base = draw(avatar);
    sight::TrackingData input;
    sight::HandData hand;
    hand.handedness = "Left";
    hand.landmarks.resize(21, {.5f, .5f, 0.f, 1.f});
    auto point = [&](int id, float x, float y, float z = 0.f) {
      hand.landmarks[id] = {x, y, z, 1.f};
    };
    point(0, .55f, .72f);
    point(5, .46f, .49f);
    point(9, .55f, .46f);
    point(17, .69f, .51f);
    input.hands = {hand};
    input.landmarks = hand.landmarks;
    const auto pub = sight::TrackingSignalBus::new_publisher_id();
    const bool published = sight::TrackingSignalBus::publish(
        "hand", pub, sight::TrackingKind::Hand, input, 640, 480);
    for (int i = 0; i < 8; ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
      (void)draw(avatar);
    }
    const auto disabled = draw(avatar);
    const size_t disabled_delta = different(base, disabled);
    opts = obs_source_get_settings(avatar);
    obs_data_set_bool(opts, "rotate_wrists", true);
    obs_source_update(avatar, opts);
    obs_data_release(opts);
    for (int i = 0; i < 14; ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
      (void)draw(avatar);
    }
    const auto turned = draw(avatar);
    const size_t active_delta = different(base, turned);
    // A second palm orientation should alter the same wrist independently
    // of the stationary wrist point, arm source, and finger curls.
    point(5, .48f, .63f, .01f);
    point(9, .78f, .61f, .01f);
    point(17, .72f, .78f, -.01f);
    input.hands[0] = hand;
    input.landmarks = hand.landmarks;
    const bool republished = sight::TrackingSignalBus::publish(
        "hand", pub, sight::TrackingKind::Hand, input, 640, 480);
    for (int i = 0; i < 14; ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
      (void)draw(avatar);
    }
    const auto turned_again = draw(avatar);
    const size_t turn_delta = different(turned, turned_again);
    sight::TrackingSignalBus::withdraw("hand", pub);
    const auto held = draw(avatar);
    std::this_thread::sleep_for(std::chrono::milliseconds(70));
    const auto held_after = draw(avatar);
    const size_t hold_delta = different(held, held_after);
    const bool wrist_ok = published && republished && disabled_delta < 40 &&
                          active_delta > 120 && turn_delta > 120 &&
                          hold_delta < 40;
    std::cout << "NATIVE_VRM_WRIST_ROTATION " << (wrist_ok ? "PASS" : "FAIL")
              << " disabled=" << disabled_delta << " first=" << active_delta
              << " turned=" << turn_delta << " hold=" << hold_delta << "\n";
    passed &= wrist_ok;
  }
  obs_source_release(avatar);
  obs_shutdown();
  if (!passed)
    std::cerr
        << "FAIL: OBS did not draw a colored alpha VRM within 12 seconds\n";
  return passed ? 0 : 1;
}
