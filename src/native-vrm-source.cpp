// Native OBS avatar source: glTF/VRM geometry, morphs and skinning are
// drawn through libobs, not Chromium/OBS Browser Source.
#include "core/tracking_signal_bus.hpp"
#include "core/tracking_discovery.hpp"
#include "vrm/vrm_asset.hpp"
#include "vrm/vrm_coverage.hpp"
#include "vrm/vrm_tracking.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <filesystem>
#include <graphics/graphics.h>
#include <memory>
#include <mutex>
#include <obs-module.h>
#include <string>
#include <thread>
#include <util/platform.h>
#include <vector>

namespace {
constexpr const char *Shader = R"EFFECT(
uniform float4x4 ViewProj;
uniform texture2d avatar_image;
uniform float4 avatar_tint;
uniform float alpha_cutoff;
sampler_state linear_clamp {
    Filter=Linear;
    AddressU=Clamp;
    AddressV=Clamp;
};
struct VIn { float4 position:POSITION; float2 uv:TEXCOORD0; };
struct VOut { float4 position:POSITION; float2 uv:TEXCOORD0; };
VOut VSMain(VIn v_in) {
    VOut o;
    o.position=mul(float4(v_in.position.xyz,1.0),ViewProj);
    o.uv=v_in.uv;
    return o;
}
float4 PSMain(VOut v_in):TARGET {
    float4 color=avatar_image.Sample(linear_clamp,v_in.uv)*avatar_tint;
    if(color.a<alpha_cutoff)discard;
    return color;
}
technique Draw {
    pass {vertex_shader=VSMain(v_in);pixel_shader=PSMain(v_in);}
}
)EFFECT";
struct Settings {
  std::string model_path;
  std::string face_channel = "face";
  std::string body_channel = "body";
  std::string hand_channel = "hand";
  bool hands_drive_arms = false;
  bool animate_fingers = true;
  bool rotate_wrists = true;
  bool flip_hand_labels = false;
  float hand_easing_ms = 150.f;
  float face_easing_ms = 150.f;
  float lost_return_ms = 0.f;
  float lost_hold_ms = 700.f;
  uint32_t width = 720, height = 1280;
  bool portrait = false;
  float zoom = 1.f;
  unsigned tracking_ms = 750;
  float body_easing_ms = 220.f;
  float torso_strength = 1.f;
  float resting_arm_degrees = 65.f;
};
struct GPU {
  std::vector<gs_vertbuffer_t *> streams;
  std::vector<gs_texture_t *> images;
  gs_texture_t *white = nullptr;
  gs_effect_t *shader = nullptr;
  gs_texrender_t *render_target = nullptr;
  sight::vrm::Vec3 min{}, max{};
  bool has_bounds = false;
};
struct Source {
  obs_source_t *context = nullptr;
  std::mutex mutex;
  std::condition_variable cv;
  bool stop = false;
  uint64_t desired_generation = 0;
  Settings config{};
  std::unique_ptr<sight::vrm::Asset> staged;
  std::string staged_error;
  bool has_update = false;
  std::thread worker;
  sight::vrm::Asset model;
  GPU gpu;
  bool model_ready = false;
  bool warned = false;
  // Touched only on the OBS render thread. Reset whenever a new model loads.
  sight::vrm::FacePoseState face_state{};
  sight::vrm::BodyPoseState body_state{};
  sight::vrm::HandPoseState hand_state{};
  // Produced on the render thread at most once per second; read by the UI
  // Properties callback under the pre-existing mutex, never touching live
  // render-thread pose state from the OBS UI thread.
  std::chrono::steady_clock::time_point next_coverage_update{};
  std::string latest_coverage = "Waiting for first rendered avatar frame.";
};
void destroy_gpu(GPU &gpu) {
  for (auto *stream : gpu.streams)
    if (stream)
      gs_vertexbuffer_destroy(stream);
  for (auto *image : gpu.images)
    if (image)
      gs_texture_destroy(image);
  if (gpu.white)
    gs_texture_destroy(gpu.white);
  if (gpu.shader)
    gs_effect_destroy(gpu.shader);
  if (gpu.render_target)
    gs_texrender_destroy(gpu.render_target);
  gpu = {};
}
void worker(Source *s) {
  uint64_t handled = 0;
  while (true) {
    uint64_t generation;
    std::string path;
    {
      std::unique_lock lock(s->mutex);
      s->cv.wait(lock,
                 [&] { return s->stop || s->desired_generation != handled; });
      if (s->stop)
        return;
      generation = s->desired_generation;
      path = s->config.model_path;
    }
    std::unique_ptr<sight::vrm::Asset> model;
    std::string error;
    if (!path.empty()) {
      std::error_code ec;
      const auto size = std::filesystem::file_size(path, ec);
      if (ec || size > 128 * 1024 * 1024ull) {
        error = "VRM file unreadable or exceeds 128 MiB";
      } else {
        model = std::make_unique<sight::vrm::Asset>();
        if (!sight::vrm::load(path, *model, error))
          model.reset();
      }
    }
    {
      std::lock_guard lock(s->mutex);
      handled = generation;
      if (s->stop)
        return;
      if (generation != s->desired_generation)
        continue;
      s->staged = std::move(model);
      s->staged_error = std::move(error);
      s->has_update = true;
    }
  }
}
bool create_gpu(Source *s) {
  auto &g = s->gpu;
  const auto &model = s->model;
  char *diagnostic = nullptr;
  g.shader = gs_effect_create(Shader, "sight-vrm-avatar.effect", &diagnostic);
  if (diagnostic) {
    blog(LOG_WARNING, "[Sight/VRM] Shader: %s", diagnostic);
    bfree(diagnostic);
  }
  if (!g.shader)
    return false;
  const uint8_t white[4] = {255, 255, 255, 255};
  const uint8_t *white_pixels = white;
  g.white = gs_texture_create(1, 1, GS_RGBA, 1, &white_pixels, 0);
  if (!g.white)
    return false;
  for (const auto &image : model.images) {
    gs_texture_t *texture = nullptr;
    if (image.width > 0 && image.height > 0 && !image.rgba.empty()) {
      const uint8_t *pixels = image.rgba.data();
      texture =
          gs_texture_create(image.width, image.height, GS_RGBA, 1, &pixels, 0);
    }
    g.images.push_back(texture);
  }
  g.render_target = gs_texrender_create(GS_RGBA, GS_Z24_S8);
  if (!g.render_target)
    return false;
  const auto neutral = sight::vrm::deform(model, {});
  for (size_t index = 0; index < model.surfaces.size(); ++index) {
    const auto &surface = model.surfaces[index];
    const auto count = surface.triangles.size();
    auto *data = gs_vbdata_create();
    data->num = count;
    data->points = (vec3 *)bzalloc(count * sizeof(vec3));
    data->num_tex = 1;
    data->tvarray = (gs_tvertarray *)bzalloc(sizeof(gs_tvertarray));
    data->tvarray[0].width = 2;
    data->tvarray[0].array = bzalloc(count * sizeof(vec2));
    auto *uv = (vec2 *)data->tvarray[0].array;
    for (size_t j = 0; j < count; ++j) {
      const uint32_t v = surface.triangles[j];
      vec2_set(&uv[j], surface.vertices[v].u, surface.vertices[v].v);
    }
    auto *stream = gs_vertexbuffer_create(data, GS_DYNAMIC);
    if (!stream)
      return false;
    g.streams.push_back(stream);
    for (const auto pos : neutral[index].positions) {
      if (!g.has_bounds) {
        g.min = g.max = pos;
        g.has_bounds = true;
      } else {
        g.min.x = std::min(g.min.x, pos.x);
        g.max.x = std::max(g.max.x, pos.x);
        g.min.y = std::min(g.min.y, pos.y);
        g.max.y = std::max(g.max.y, pos.y);
        g.min.z = std::min(g.min.z, pos.z);
        g.max.z = std::max(g.max.z, pos.z);
      }
    }
  }
  return g.has_bounds;
}
void collect_loaded(Source *s) {
  std::unique_ptr<sight::vrm::Asset> next;
  std::string error;
  {
    std::lock_guard lock(s->mutex);
    if (!s->has_update)
      return;
    s->has_update = false;
    next = std::move(s->staged);
    error = std::move(s->staged_error);
  }
  destroy_gpu(s->gpu);
  s->model = {};
  s->model_ready = false;
  s->face_state = {};
  s->body_state = {};
  s->hand_state = {};
  s->next_coverage_update = {};
  {
    std::lock_guard lock(s->mutex);
    s->latest_coverage = "Avatar loading; waiting for first rendered frame.";
  }
  if (!next) {
    if (!error.empty())
      blog(LOG_ERROR, "[Sight/VRM] Model load failed: %s", error.c_str());
    return;
  }
  s->model = std::move(*next);
  if (!create_gpu(s)) {
    blog(LOG_ERROR, "[Sight/VRM] GPU avatar resources failed");
    destroy_gpu(s->gpu);
    s->model = {};
    return;
  }
  s->model_ready = true;
  // Installation probe: distinguish this corrected runtime from an older
  // Sight binary that might still be found on the OBS plugin search path.
  blog(LOG_INFO, "[Sight/VRM] PITCH_FIX_ACTIVE_20261104_B714");
  blog(LOG_INFO, "[Sight/VRM] HEAD_YAW_HAND_ALIGNMENT_20261104_K421");
  blog(LOG_INFO, "[Sight/VRM] BODY_FUSION_ACTIVE_20261104_R631");
  blog(LOG_INFO, "[Sight/VRM] RELAXED_HOLD_EASE_ACTIVE_20261104_N638");
  blog(LOG_INFO, "[Sight/VRM] HAND_WRIST_IK_ACTIVE_20261104_W913");
  blog(LOG_INFO, "[Sight/VRM] PALM_WRIST_ROTATION_ACTIVE_20261104_C519");
  blog(LOG_INFO, "[Sight/VRM] FINGER_BIND_PALM_FLEX_ACTIVE_20261104_S916");
  blog(LOG_INFO, "[Sight/VRM] FACE_CONTINUITY_ACTIVE_20261104_F284");
  blog(LOG_INFO, "[Sight/VRM] WORLD_LANDMARK_DATA_ACTIVE_20261104_M438");
  blog(LOG_INFO, "[Sight/VRM] RIG_COVERAGE_PROPERTIES_ACTIVE_20261104_D726");
  blog(LOG_INFO, "[Sight/VRM] MATRIX_TRS_RIG_INTEROP_ACTIVE_20261104_B903");
  blog(LOG_INFO, "[Sight/VRM] SPINE_RETARGET_ACTIVE_20261104_T762");
  blog(LOG_INFO,
       "[Sight/VRM] NATIVE_VRM_SOURCE_READY_20261104 (%zu surfaces, VRM %s)",
       s->model.surfaces.size(), s->model.vrm_version ? "1.0" : "0.x");
}
void draw(Source *s, const Settings &config) {
  const auto &a = s->model;
  auto &g = s->gpu;
  sight::TrackingSignal face;
  const auto age = std::chrono::milliseconds(config.tracking_ms);
  const auto *f = sight::TrackingSignalBus::read(config.face_channel, age, face)
                      ? &face
                      : nullptr;
  sight::TrackingSignal body;
  const auto *b = sight::TrackingSignalBus::read(config.body_channel, age, body)
                      ? &body
                      : nullptr;
  sight::TrackingSignal hands;
  const auto *h =
      (config.hands_drive_arms || config.animate_fingers ||
       config.rotate_wrists) &&
              sight::TrackingSignalBus::read(config.hand_channel, age, hands)
          ? &hands
          : nullptr;
  // The face owns head and expressions. Body owns chest and arm pose;
  // optional hands override arms on their own side when in frame.
  // Finger articulation is orthogonal and can be disabled independently.
  const auto now = std::chrono::steady_clock::now();
  auto pose = sight::vrm::ease_face_pose(
      a, f, s->face_state, now, config.face_easing_ms, config.lost_return_ms,
      config.lost_hold_ms);
  const auto body_pose = sight::vrm::ease_body_pose(
      a, b, s->body_state, now, config.body_easing_ms,
      config.resting_arm_degrees, h, config.hands_drive_arms,
      config.flip_hand_labels, f, config.lost_return_ms, config.lost_hold_ms,
      config.torso_strength);
  for (const auto &[name, rotation] : body_pose.bone_rotation)
    pose.bone_rotation[name] = rotation;
  if (config.animate_fingers || config.rotate_wrists) {
    const auto hand_pose = sight::vrm::ease_hand_pose(
        a, h, s->hand_state, now, config.hand_easing_ms,
        config.flip_hand_labels, config.lost_return_ms, config.lost_hold_ms,
        config.rotate_wrists, &body_pose, config.animate_fingers);
    for (const auto &[name, rotation] : hand_pose.bone_rotation)
      pose.bone_rotation[name] = rotation;
  } else {
    s->hand_state = {};
  }
  if (now >= s->next_coverage_update) {
    const sight::vrm::CoverageInput coverage{a,
                                             s->face_state,
                                             s->body_state,
                                             s->hand_state,
                                             config.lost_return_ms,
                                             config.lost_hold_ms,
                                             config.rotate_wrists,
                                             config.animate_fingers,
                                             config.hands_drive_arms,
                                             f,
                                             b,
                                             h};
    auto summary = sight::vrm::format_coverage(coverage, now);
    {
      std::lock_guard lock(s->mutex);
      s->latest_coverage = std::move(summary);
    }
    s->next_coverage_update = now + std::chrono::seconds(1);
  }
  const auto frame = sight::vrm::deform(a, pose);
  const float width = float(config.width), height = float(config.height);
  const float body_height = std::max(.1f, g.max.y - g.min.y);
  const float body_width = std::max(.1f, g.max.x - g.min.x);
  const float target_h = body_height * (config.portrait ? .68f : 1.22f);
  const float scale =
      std::min(height / target_h, width / (body_width * 1.14f)) * config.zoom;
  const float focus_y = g.min.y + body_height * (config.portrait ? .74f : .53f);
  const float focus_x =
      (g.min.x + g.max.x) * .5f * (a.vrm_version == 0 ? -1.f : 1.f);
  const float focus_z =
      (g.min.z + g.max.z) * .5f * (a.vrm_version == 0 ? -1.f : 1.f);
  gs_texrender_reset(g.render_target);
  if (!gs_texrender_begin(g.render_target, config.width, config.height))
    return;
  const vec4 clear = {};
  gs_clear(GS_CLEAR_COLOR | GS_CLEAR_DEPTH, &clear, 1, 0);
  gs_projection_push();
  gs_matrix_push();
  gs_matrix_identity();
  gs_ortho(0, width, 0, height, -100, 100);
  gs_blend_state_push();
  gs_enable_depth_test(true);
  gs_depth_function(GS_LESS);
  gs_enable_blending(true);
  gs_blend_function(GS_BLEND_SRCALPHA, GS_BLEND_INVSRCALPHA);
  const auto previous_cull = gs_get_cull_mode();
  gs_set_cull_mode(GS_NEITHER);
  for (size_t i = 0; i < a.surfaces.size(); ++i) {
    const auto &surface = a.surfaces[i];
    auto *vb = g.streams[i];
    auto *points = gs_vertexbuffer_get_data(vb)->points;
    for (size_t j = 0; j < surface.triangles.size(); ++j) {
      auto pos = frame[i].positions[surface.triangles[j]];
      // VRM 0.x uses the opposite model-facing convention from 1.0.
      if (a.vrm_version == 0) {
        pos.x = -pos.x;
        pos.z = -pos.z;
      }
      vec3_set(&points[j], width * .5f + (pos.x - focus_x) * scale,
               height * .5f - (pos.y - focus_y) * scale, (pos.z - focus_z));
    }
    gs_vertexbuffer_flush(vb);
    gs_load_vertexbuffer(vb);
    gs_load_indexbuffer(nullptr);
    const sight::vrm::Material *mat = nullptr;
    if (surface.material >= 0 && size_t(surface.material) < a.materials.size())
      mat = &a.materials[surface.material];
    auto *image = g.white;
    if (mat && mat->image >= 0 && size_t(mat->image) < g.images.size() &&
        g.images[mat->image])
      image = g.images[mat->image];
    auto *texture_param = gs_effect_get_param_by_name(g.shader, "avatar_image");
    if (texture_param)
      gs_effect_set_texture(texture_param, image);
    auto *color_param = gs_effect_get_param_by_name(g.shader, "avatar_tint");
    if (color_param) {
      vec4 color{};
      vec4_set(&color, mat ? mat->color[0] : 1.f, mat ? mat->color[1] : 1.f,
               mat ? mat->color[2] : 1.f, mat ? mat->color[3] : 1.f);
      gs_effect_set_vec4(color_param, &color);
    }
    auto *cutoff = gs_effect_get_param_by_name(g.shader, "alpha_cutoff");
    if (cutoff)
      gs_effect_set_float(cutoff, mat && mat->mask ? mat->alpha_cutoff : .005f);
    while (gs_effect_loop(g.shader, "Draw"))
      gs_draw(GS_TRIS, 0, uint32_t(surface.triangles.size()));
  }
  gs_load_vertexbuffer(nullptr);
  gs_set_cull_mode(previous_cull);
  gs_enable_depth_test(false);
  gs_blend_state_pop();
  gs_matrix_pop();
  gs_projection_pop();
  gs_texrender_end(g.render_target);
  auto *texture = gs_texrender_get_texture(g.render_target);
  if (!texture)
    return;
  auto *effect = obs_get_base_effect(OBS_EFFECT_DEFAULT);
  auto *param = gs_effect_get_param_by_name(effect, "image");
  if (param)
    gs_effect_set_texture(param, texture);
  while (gs_effect_loop(effect, "Draw"))
    gs_draw_sprite(texture, 0, config.width, config.height);
}
const char *name(void *) { return "Sight: VRM Avatar (Native)"; }
void update(void *data, obs_data_t *settings);
void *create(obs_data_t *settings, obs_source_t *context) {
  auto *s = new Source;
  s->context = context;
  s->worker = std::thread(worker, s);
  const auto *path = obs_data_get_string(settings, "model_path");
  {
    std::lock_guard lock(s->mutex);
    s->config.model_path = path ? path : "";
    ++s->desired_generation;
  }
  s->cv.notify_one();
  // OBS doesn't guarantee a later update callback for initial source
  // settings. Apply the entire saved configuration at construction time.
  update(s, settings);
  return s;
}
void destroy(void *data) {
  auto *s = static_cast<Source *>(data);
  if (!s)
    return;
  {
    std::lock_guard lock(s->mutex);
    s->stop = true;
  }
  s->cv.notify_one();
  if (s->worker.joinable())
    s->worker.join();
  obs_enter_graphics();
  destroy_gpu(s->gpu);
  obs_leave_graphics();
  delete s;
}
void update(void *data, obs_data_t *settings) {
  auto *s = static_cast<Source *>(data);
  Settings next;
  const auto *path = obs_data_get_string(settings, "model_path");
  const auto *face = obs_data_get_string(settings, "face_channel");
  const auto *body = obs_data_get_string(settings, "body_channel");
  const auto *hand = obs_data_get_string(settings, "hand_channel");
  next.model_path = path ? path : "";
  next.face_channel = face ? face : "face";
  next.body_channel = body ? body : "body";
  next.hand_channel = hand ? hand : "hand";
  next.hands_drive_arms = obs_data_get_bool(settings, "hands_drive_arms");
  next.animate_fingers = obs_data_get_bool(settings, "animate_fingers");
  next.rotate_wrists = obs_data_get_bool(settings, "rotate_wrists");
  next.flip_hand_labels = obs_data_get_bool(settings, "flip_hand_labels");
  next.hand_easing_ms = std::clamp(
      float(obs_data_get_double(settings, "hand_easing_ms")), 40.f, 700.f);
  next.face_easing_ms = std::clamp(
      float(obs_data_get_double(settings, "face_easing_ms")), 40.f, 700.f);
  next.lost_return_ms = std::clamp(
      float(obs_data_get_double(settings, "lost_return_ms")), 0.f, 4000.f);
  next.lost_hold_ms = std::clamp(
      float(obs_data_get_double(settings, "lost_hold_ms")), 0.f, 5000.f);
  next.width = std::clamp(obs_data_get_int(settings, "width"), 128LL, 3840LL);
  next.height = std::clamp(obs_data_get_int(settings, "height"), 128LL, 3840LL);
  next.portrait = obs_data_get_bool(settings, "portrait");
  next.zoom =
      std::clamp(float(obs_data_get_double(settings, "zoom")), .35f, 3.f);
  next.tracking_ms =
      std::clamp(obs_data_get_int(settings, "tracking_ms"), 100LL, 2000LL);
  next.body_easing_ms = std::clamp(
      float(obs_data_get_double(settings, "body_easing_ms")), 40.f, 700.f);
  next.torso_strength = std::clamp(
      float(obs_data_get_double(settings, "torso_strength")), 0.f, 2.f);
  next.resting_arm_degrees = std::clamp(
      float(obs_data_get_double(settings, "resting_arm_degrees")), 0.f, 85.f);
  bool changed;
  {
    std::lock_guard lock(s->mutex);
    changed = next.model_path != s->config.model_path;
    s->config = std::move(next);
    if (changed)
      ++s->desired_generation;
  }
  if (changed)
    s->cv.notify_one();
}
void defaults(obs_data_t *settings) {
  obs_data_set_default_string(settings, "model_path", "");
  obs_data_set_default_string(settings, "face_channel", "face");
  obs_data_set_default_string(settings, "body_channel", "body");
  obs_data_set_default_string(settings, "hand_channel", "hand");
  obs_data_set_default_bool(settings, "hands_drive_arms", false);
  obs_data_set_default_bool(settings, "animate_fingers", true);
  obs_data_set_default_bool(settings, "rotate_wrists", true);
  obs_data_set_default_bool(settings, "flip_hand_labels", false);
  obs_data_set_default_double(settings, "hand_easing_ms", 150);
  obs_data_set_default_double(settings, "face_easing_ms", 150);
  obs_data_set_default_double(settings, "lost_return_ms", 0);
  obs_data_set_default_double(settings, "lost_hold_ms", 700);
  obs_data_set_default_int(settings, "width", 720);
  obs_data_set_default_int(settings, "height", 1280);
  obs_data_set_default_bool(settings, "portrait", false);
  obs_data_set_default_double(settings, "zoom", 1);
  obs_data_set_default_int(settings, "tracking_ms", 750);
  obs_data_set_default_double(settings, "body_easing_ms", 220);
  obs_data_set_default_double(settings, "torso_strength", 1);
  obs_data_set_default_double(settings, "resting_arm_degrees", 65);
}
obs_properties_t *properties(void *data) {
  auto *s = static_cast<Source *>(data);
  std::string status = "Waiting for an avatar render.";
  if (s) {
    std::lock_guard lock(s->mutex);
    status = s->latest_coverage;
  }
  auto *props = obs_properties_create();
  obs_properties_add_path(props, "model_path", "Local VRM Model", OBS_PATH_FILE,
                          "VRM Avatar (*.vrm)", nullptr);
  sight::discovery::add_picker(props,s?s->context:nullptr,"face_channel",
      "Sight Face Motion Channel","face",sight::TrackingKind::Face);
  sight::discovery::add_picker(props,s?s->context:nullptr,"body_channel",
      "Sight Body Motion Channel","body",sight::TrackingKind::Body);
  sight::discovery::add_picker(props,s?s->context:nullptr,"hand_channel",
      "Sight Hand Motion Channel","hand",sight::TrackingKind::Hand);
  obs_properties_add_bool(props, "hands_drive_arms",
                          "Hands Drive Arms (wrist position IK)");
  obs_properties_add_bool(props, "animate_fingers",
                          "Animate Fingers from Hand Landmarks");
  obs_properties_add_bool(props, "rotate_wrists",
                          "Rotate Wrists from Palm Landmarks");
  obs_properties_add_bool(props, "flip_hand_labels",
                          "Swap Tracked Left/Right Hands");
  obs_properties_add_float_slider(props, "hand_easing_ms",
                                  "Hand/Finger Ease (ms)", 40, 700, 10);
  obs_properties_add_float_slider(props, "face_easing_ms",
                                  "Head/Expression Ease (ms)", 40, 700, 10);
  obs_properties_add_float_slider(
      props, "lost_hold_ms", "Tracking Loss Hold Delay (ms)", 0, 5000, 100);
  obs_properties_add_float_slider(props, "lost_return_ms",
                                  "Lost Tracking Return Speed (ms; 0=hold)", 0,
                                  4000, 100);
  obs_properties_add_int(props, "width", "Output Width", 128, 3840, 1);
  obs_properties_add_int(props, "height", "Output Height", 128, 3840, 1);
  obs_properties_add_bool(props, "portrait", "Portrait Framing");
  obs_properties_add_float_slider(props, "zoom", "Zoom", .35, 3, .05);
  obs_properties_add_int(props, "tracking_ms", "Tracking Timeout (ms)", 100,
                         2000, 50);
  obs_properties_add_float_slider(props, "resting_arm_degrees",
                                  "Relaxed Arm Drop (degrees)", 0, 85, 5);
  obs_properties_add_float_slider(props, "body_easing_ms",
                                  "Body Motion Ease (ms)", 40, 700, 20);
  obs_properties_add_float_slider(props, "torso_strength",
                                  "Torso Lean Strength", 0, 2, .1);
  // Keep the long read-only snapshot AFTER the actual animation controls.
  // The property label is sampled when Properties opens; there is no
  // re-entrant property refresh or render-thread Qt mutation.
  obs_properties_add_text(props, "rig_coverage_info", status.c_str(),
                          OBS_TEXT_INFO);
  return props;
}
uint32_t get_width(void *data) {
  auto *s = static_cast<Source *>(data);
  std::lock_guard lock(s->mutex);
  return s->config.width;
}
uint32_t get_height(void *data) {
  auto *s = static_cast<Source *>(data);
  std::lock_guard lock(s->mutex);
  return s->config.height;
}
void render(void *data, gs_effect_t *) {
  auto *s = static_cast<Source *>(data);
  Settings config;
  {
    std::lock_guard lock(s->mutex);
    config = s->config;
  }
  collect_loaded(s);
  if (!s->model_ready)
    return;
  draw(s, config);
}
} // namespace

struct obs_source_info sight_native_vrm_source_info = {
    .id = "sight_native_vrm_source",
    .type = OBS_SOURCE_TYPE_INPUT,
    .output_flags = OBS_SOURCE_VIDEO | OBS_SOURCE_CUSTOM_DRAW,
    .get_name = name,
    .create = create,
    .destroy = destroy,
    .get_width = get_width,
    .get_height = get_height,
    .get_defaults = defaults,
    .get_properties = properties,
    .update = update,
    .video_render = render,
};
