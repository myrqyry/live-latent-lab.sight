#include "vrm_tracking.hpp"
#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <optional>
#include <string>
#include <vector>
namespace sight::vrm {
namespace {
float score(const TrackingData &data, const char *name) {
  for (size_t i = 0;
       i < data.blendshape_names.size() && i < data.blendshapes.size(); ++i)
    if (data.blendshape_names[i] == name) {
      const float value = data.blendshapes[i];
      return std::isfinite(value) ? std::clamp(value, 0.f, 1.f) : 0.f;
    }
  return 0;
}
void expression(const Asset &asset, Pose &pose, const char *name,
                float weight) {
  if (weight > 0.001f && asset.expressions.contains(name))
    pose.expressions[name] = std::clamp(weight, 0.f, 1.f);
}
void first_expression(const Asset &asset, Pose &pose,
                      std::initializer_list<const char *> names, float weight) {
  for (const auto name : names)
    if (asset.expressions.contains(name)) {
      expression(asset, pose, name, weight);
      return;
    }
}

// Pose Landmarker yields 33 camera-normalized points. Use only reliable
// observed shoulder/elbow/wrist/hip landmarks; no invented joint positions.
bool reliable(const TrackingSignal &body, int idx) {
  if (idx < 0 || size_t(idx) >= body.data.landmarks.size())
    return false;
  const auto &p = body.data.landmarks[idx];
  return std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z) &&
         std::isfinite(p.confidence) && p.confidence >= .5f && p.x > -.25f &&
         p.x < 1.25f && p.y > -.25f && p.y < 1.25f && std::fabs(p.z) < 3.f;
}
Vec3 minus(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
float dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
Vec3 cross(Vec3 a, Vec3 b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
Vec3 multiply_scalar(Vec3 v, float scale) {
  return {v.x * scale, v.y * scale, v.z * scale};
}
float norm(Vec3 a) { return std::sqrt(dot(a, a)); }
Vec3 unit(Vec3 a) {
  const float length = norm(a);
  return length > 1e-7f ? multiply_scalar(a, 1.f / length) : Vec3{};
}
// glTF node transforms are either a matrix OR separate TRS fields. The
// native renderer already handles both, but the retargeter's old bone-length
// reads silently used zero translations on matrix-backed ViiRM exports.
// Interpret the authored local offset at this single shared boundary.
Vec3 local_offset(const Node &node) {
  if (node.use_matrix)
    return {node.matrix[12], node.matrix[13], node.matrix[14]};
  return node.translation;
}
Vec3 camera_segment(const TrackingSignal &body, int from, int to) {
  const auto &a = body.data.landmarks[from];
  const auto &b = body.data.landmarks[to];
  // Model +X is camera image-right for front-facing humanoids,
  // camera image-up is model +Y, and depth is intentionally conservative.
  const float aspect =
      body.source_width && body.source_height
          ? std::clamp(float(body.source_width) / body.source_height, .5f, 3.f)
          : 1.f;
  return {(b.x - a.x) * aspect, a.y - b.y, (a.z - b.z) * .5f};
}
Mat4 global_before_correction(const Asset &asset, int bone, const Pose &pose) {
  std::vector<int> path;
  for (int n = bone; n >= 0 && size_t(n) < asset.nodes.size() &&
                     path.size() < asset.nodes.size();
       n = asset.nodes[n].parent)
    path.push_back(n);
  Mat4 global = identity();
  // Include ancestor body rotations (e.g. chest) but NOT this joint's
  // yet-to-be-computed correction. The runtime applies rest*correction.
  for (auto it = path.rbegin(); it != path.rend(); ++it) {
    const int idx = *it;
    const auto &node = asset.nodes[idx];
    auto local = node.use_matrix
                     ? node.matrix
                     : trs(node.translation, node.rotation, node.scale);
    if (idx != bone) {
      for (const auto &[name, bone_index] : asset.bones)
        if (bone_index == idx) {
          if (const auto found = pose.bone_rotation.find(name);
              found != pose.bone_rotation.end())
            local = multiply(local, trs({}, found->second, {1, 1, 1}));
          break;
        }
    }
    global = multiply(global, local);
  }
  return global;
}
Quat between(Vec3 source, Vec3 target, float max_angle) {
  source = unit(source);
  target = unit(target);
  if (norm(source) < .5f || norm(target) < .5f)
    return {};
  const float cosine = std::clamp(dot(source, target), -1.f, 1.f);
  const float angle = std::min(std::acos(cosine), max_angle);
  if (angle < .012f)
    return {};
  auto axis = cross(source, target);
  if (norm(axis) < 1e-5f) {
    // True opposite directions: choose a stable perpendicular axis.
    axis = cross(source,
                 std::fabs(source.x) < .9f ? Vec3{1, 0, 0} : Vec3{0, 1, 0});
  }
  axis = unit(axis);
  const float s = std::sin(angle * .5f);
  return {axis.x * s, axis.y * s, axis.z * s, std::cos(angle * .5f)};
}
void aim_arm_bone(const Asset &asset, Pose &pose, const char *name,
                  const char *child_name, Vec3 camera_direction,
                  float max_angle) {
  const auto bone_it = asset.bones.find(name);
  const auto child_it = asset.bones.find(child_name);
  if (bone_it == asset.bones.end() || child_it == asset.bones.end())
    return;
  const int bone = bone_it->second, child = child_it->second;
  if (bone < 0 || child < 0 || size_t(bone) >= asset.nodes.size() ||
      size_t(child) >= asset.nodes.size() ||
      asset.nodes[child].parent != bone || norm(camera_direction) < .03f)
    return;
  const auto global = global_before_correction(asset, bone, pose);
  // Express the camera direction in this bone's current post-rest basis.
  // This accounts for authored FBX bone axes and any preceding chest/arm
  // correction, rather than assuming all rigs use X-aligned bones.
  const Vec3 x = unit({global[0], global[1], global[2]});
  const Vec3 y = unit({global[4], global[5], global[6]});
  const Vec3 z = unit({global[8], global[9], global[10]});
  const Vec3 desired_local = {dot(camera_direction, x),
                              dot(camera_direction, y),
                              dot(camera_direction, z)};
  const auto q =
      between(local_offset(asset.nodes[child]), desired_local, max_angle);
  // Do not insert identity rotations: neutral body frames must stay neutral.
  if (std::fabs(q.x) + std::fabs(q.y) + std::fabs(q.z) > .006f)
    pose.bone_rotation[name] = q;
}
void apply_body(const Asset &asset, const TrackingSignal *body, Pose &pose,
                float torso_strength) {
  if (!body || body->kind != TrackingKind::Body ||
      body->data.landmarks.size() != 33)
    return;
  constexpr int LS = 11, RS = 12, LE = 13, RE = 14, LW = 15, RW = 16, LH = 23,
                RH = 24;
  const bool shoulders = reliable(*body, LS) && reliable(*body, RS);
  const bool hips = reliable(*body, LH) && reliable(*body, RH);
  // Rotate the actual spine bone from shoulders relative to the pelvis,
  // rather than only rotating the chest when the shoulder LINE tilts.
  // The pelvis position itself is not inferred from a single 2D camera.
  if (shoulders && hips && torso_strength > .001f) {
    const auto shoulder = camera_segment(*body, RS, LS);
    const auto hip = camera_segment(*body, RH, LH);
    const auto &ls = body->data.landmarks[LS];
    const auto &rs = body->data.landmarks[RS];
    const auto &lh = body->data.landmarks[LH];
    const auto &rh = body->data.landmarks[RH];
    const float dx =
        ((ls.x + rs.x) - (lh.x + rh.x)) * .5f *
        (body->source_height ? std::clamp(float(body->source_width) /
                                              float(body->source_height),
                                          .5f, 3.f)
                             : 1.f);
    const float vertical = ((lh.y + rh.y) - (ls.y + rs.y)) * .5f;
    const float depth = ((lh.z + rh.z) - (ls.z + rs.z)) * .25f;
    // Require a plausible torso, not reversed or collapsed landmarks.
    if (shoulder.x > .07f && hip.x > .05f && vertical > .12f &&
        vertical < 1.f && std::fabs(dx) < 1.f && std::fabs(depth) < .7f) {
      if (asset.bones.contains("spine")) {
        // Positive screen-right shoulder drift above fixed hips = negative
        // Z rotation. Shoulder depth toward camera = positive X pitch.
        const float lean = std::clamp(
            -std::atan2(dx, vertical) * .9f * torso_strength, -.52f, .52f);
        const float pitch = std::clamp(
            std::atan2(depth, vertical) * .85f * torso_strength, -.40f, .40f);
        if (std::fabs(lean) > .012f || std::fabs(pitch) > .012f)
          pose.bone_rotation["spine"] = from_euler(pitch, 0, lean);
      }
      if (asset.bones.contains("chest")) {
        // Shoulder asymmetry adds modest chest rotation relative to spine.
        const float roll = std::clamp(std::atan2(shoulder.y, shoulder.x) * .8f *
                                          torso_strength,
                                      -.35f, .35f);
        if (std::fabs(roll) > .018f)
          pose.bone_rotation["chest"] = from_euler(0, 0, roll);
      }
    }
  }
  if (reliable(*body, LS) && reliable(*body, LE))
    aim_arm_bone(asset, pose, "leftUpperArm", "leftLowerArm",
                 camera_segment(*body, LS, LE), 1.95f);
  if (reliable(*body, RS) && reliable(*body, RE))
    aim_arm_bone(asset, pose, "rightUpperArm", "rightLowerArm",
                 camera_segment(*body, RS, RE), 1.95f);
  if (reliable(*body, LE) && reliable(*body, LW))
    aim_arm_bone(asset, pose, "leftLowerArm", "leftHand",
                 camera_segment(*body, LE, LW), 2.2f);
  if (reliable(*body, RE) && reliable(*body, RW))
    aim_arm_bone(asset, pose, "rightLowerArm", "rightHand",
                 camera_segment(*body, RE, RW), 2.2f);
}

// MediaPipe reports hand detection order opportunistically. Never infer
// persistent left/right identity from order or screen X.
const HandData *hand_for(const TrackingSignal *signal, const char *side,
                         bool mirror) {
  if (!signal || signal->kind != TrackingKind::Hand)
    return nullptr;
  const std::string wanted = side[0] == 'l' ? "Left" : "Right";
  for (const auto &hand : signal->data.hands) {
    if (hand.landmarks.size() != 21)
      continue;
    const std::string label = hand.handedness;
    if (label != "Left" && label != "Right")
      continue;
    if ((mirror ? (label == "Left" ? "Right" : "Left") : label) == wanted)
      return &hand;
  }
  return nullptr;
}
bool valid_hand_point(const HandData &hand, int index) {
  if (index < 0 || size_t(index) >= hand.landmarks.size())
    return false;
  const auto &p = hand.landmarks[index];
  return std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z) &&
         std::isfinite(p.confidence) && p.confidence >= .5f && p.x >= -.15f &&
         p.x <= 1.15f && p.y >= -.15f && p.y <= 1.15f && std::fabs(p.z) <= 3.f;
}
float finger_bend(const HandData &hand, int prev, int joint, int next,
                  float deadzone) {
  if (!valid_hand_point(hand, prev) || !valid_hand_point(hand, joint) ||
      !valid_hand_point(hand, next))
    return -1.f;
  const auto &a = hand.landmarks[prev], &b = hand.landmarks[joint],
             &c = hand.landmarks[next];
  const Vec3 before = {b.x - a.x, a.y - b.y, (b.z - a.z) * .5f};
  const Vec3 after = {c.x - b.x, b.y - c.y, (c.z - b.z) * .5f};
  if (norm(before) < .006f || norm(after) < .006f)
    return -1.f;
  return std::clamp(
      std::acos(std::clamp(dot(unit(before), unit(after)), -1.f, 1.f)) -
          deadzone,
      0.f, 2.25f);
}
float safe_alpha(float elapsed, float ms) {
  const float dt = std::clamp(elapsed, 0.f, .06f);
  return 1.f - std::exp(-dt / std::max(.025f, ms * .001f));
}
std::chrono::steady_clock::time_point time_or(
    const std::map<std::string, std::chrono::steady_clock::time_point> &times,
    const std::string &name, std::chrono::steady_clock::time_point fallback) {
  const auto it = times.find(name);
  return it == times.end() ? fallback : it->second;
}
// A planar two-bone inverse kinematics solution. Shoulder position comes
// from a valid Body observation if present; otherwise from face proportions,
// or lastly a conservative desktop camera anchor. Camera-space only:
// without calibrated depth, reaching toward/away from lens is approximate.
void aim_from_wrist(const Asset &asset, Pose &pose, const char *side,
                    const HandData &hand, const TrackingSignal *body,
                    const TrackingSignal *face) {
  if (!valid_hand_point(hand, 0))
    return;
  const bool left = std::string(side) == "left";
  const char *upper = left ? "leftUpperArm" : "rightUpperArm";
  const char *lower = left ? "leftLowerArm" : "rightLowerArm";
  const char *wrist_name = left ? "leftHand" : "rightHand";
  const auto iu = asset.bones.find(upper), il = asset.bones.find(lower),
             iw = asset.bones.find(wrist_name);
  if (iu == asset.bones.end() || il == asset.bones.end() ||
      iw == asset.bones.end())
    return;
  const int u = iu->second, l = il->second, w = iw->second;
  if (u < 0 || l < 0 || w < 0 || size_t(w) >= asset.nodes.size() ||
      size_t(l) >= asset.nodes.size() || size_t(u) >= asset.nodes.size() ||
      asset.nodes[l].parent != u || asset.nodes[w].parent != l)
    return;
  const float a = norm(local_offset(asset.nodes[l]));
  const float b = norm(local_offset(asset.nodes[w]));
  if (a < .03f || b < .03f)
    return;

  float sx = left ? .65f : .35f, sy = .37f;
  const int shoulder_index = left ? 11 : 12;
  if (body && body->kind == TrackingKind::Body &&
      body->data.landmarks.size() == 33 && reliable(*body, shoulder_index)) {
    sx = body->data.landmarks[shoulder_index].x;
    sy = body->data.landmarks[shoulder_index].y;
  } else if (face && face->kind == TrackingKind::Face &&
             face->data.landmarks.size() >= 468) {
    // The facial landmark bounding box gives a scale-dependent shoulder
    // anchor when body tracking is lost (not an inferred real shoulder).
    float minx = 2.f, maxx = -1.f, miny = 2.f, maxy = -1.f;
    for (const auto &p : face->data.landmarks) {
      if (!std::isfinite(p.x) || !std::isfinite(p.y) || p.x < 0 || p.x > 1 ||
          p.y < 0 || p.y > 1)
        continue;
      minx = std::min(minx, p.x);
      maxx = std::max(maxx, p.x);
      miny = std::min(miny, p.y);
      maxy = std::max(maxy, p.y);
    }
    if (maxx > minx + .025f && maxy > miny + .025f) {
      const float halfwidth = (maxx - minx) * .65f;
      sx = std::clamp((minx + maxx) * .5f + (left ? halfwidth : -halfwidth),
                      .05f, .95f);
      sy = std::clamp(maxy + .25f * (maxy - miny), .05f, .85f);
    }
  }
  const auto &pt = hand.landmarks[0];
  const float aspect =
      body && body->source_width && body->source_height
          ? std::clamp(float(body->source_width) / body->source_height, .7f,
                       2.4f)
          : (face && face->source_width && face->source_height
                 ? std::clamp(float(face->source_width) / face->source_height,
                              .7f, 2.4f)
                 : 1.3333f);
  const float factor = (a + b) / .75f;
  Vec3 target = {(pt.x - sx) * aspect * factor, (sy - pt.y) * factor, 0};
  float distance = norm(target);
  if (distance < .08f * a)
    return;
  // A 2D target cannot distinguish depth; keep it inside a stable reach.
  distance = std::clamp(distance, std::fabs(a - b) + .035f, (a + b) * .96f);
  const Vec3 dir = unit(target);
  const float along = (a * a - b * b + distance * distance) / (2.f * distance);
  const float elbow_offset = std::sqrt(std::max(0.f, a * a - along * along));
  const Vec3 perp = {-dir.y, dir.x, 0};
  const Vec3 elbow =
      minus(multiply_scalar(dir, along),
            multiply_scalar(perp, elbow_offset * (left ? -1.f : 1.f)));
  const Vec3 hand_target = multiply_scalar(dir, distance);
  aim_arm_bone(asset, pose, upper, lower, elbow, 2.2f);
  aim_arm_bone(asset, pose, lower, wrist_name, minus(hand_target, elbow),
               2.35f);
}
bool valid_segment(const TrackingSignal &body, int from, int to) {
  return reliable(body, from) && reliable(body, to) &&
         norm(camera_segment(body, from, to)) > .03f;
}
bool valid_chest(const TrackingSignal &body) {
  if (!reliable(body, 11) || !reliable(body, 12) || !reliable(body, 23) ||
      !reliable(body, 24))
    return false;
  return camera_segment(body, 12, 11).x > .07f &&
         camera_segment(body, 24, 23).x > .05f;
}
Quat normalized(Quat q) {
  const float n = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
  if (!std::isfinite(n) || n < 1.e-7f)
    return {};
  return {q.x / n, q.y / n, q.z / n, q.w / n};
}
Quat interpolate(Quat previous, Quat target, float blend) {
  previous = normalized(previous);
  target = normalized(target);
  float cosine = previous.x * target.x + previous.y * target.y +
                 previous.z * target.z + previous.w * target.w;
  if (cosine < 0.f) {
    target = {-target.x, -target.y, -target.z, -target.w};
    cosine = -cosine;
  }
  const float t = std::clamp(blend, 0.f, 1.f);
  float a = 1.f - t, b = t;
  if (cosine < .9995f) {
    const float angle = std::acos(std::clamp(cosine, 0.f, 1.f));
    const float denom = std::sin(angle);
    a = std::sin((1.f - t) * angle) / denom;
    b = std::sin(t * angle) / denom;
  }
  return normalized(
      {a * previous.x + b * target.x, a * previous.y + b * target.y,
       a * previous.z + b * target.z, a * previous.w + b * target.w});
}

// Reconstruct a conservative camera-space hand attitude from wrist, middle
// MCP and the across-palm index/pinky knuckles. A wrist position alone has
// no rotation: a real 2D hand plane + available relative Z is required.
// Return the correction in the wrist's authored LOCAL bind coordinate frame,
// after all already-tracked arm/torso ancestors have been applied.
std::optional<Quat> wrist_attitude(const Asset &asset, const char *side,
                                   const HandData &hand, float aspect,
                                   const Pose *arm_pose) {
  constexpr int wrist = 0, index = 5, middle = 9, little = 17;
  if (!valid_hand_point(hand, wrist) || !valid_hand_point(hand, index) ||
      !valid_hand_point(hand, middle) || !valid_hand_point(hand, little))
    return std::nullopt;
  const std::string side_name = side;
  const auto h = asset.bones.find(side_name + "Hand");
  const auto i = asset.bones.find(side_name + "IndexProximal");
  const auto m = asset.bones.find(side_name + "MiddleProximal");
  const auto l = asset.bones.find(side_name + "LittleProximal");
  if (h == asset.bones.end() || i == asset.bones.end() ||
      m == asset.bones.end() || l == asset.bones.end())
    return std::nullopt;
  const int wrist_id = h->second;
  if (wrist_id < 0 || size_t(wrist_id) >= asset.nodes.size())
    return std::nullopt;
  for (const auto bone : {i->second, m->second, l->second})
    if (bone < 0 || size_t(bone) >= asset.nodes.size() ||
        asset.nodes[bone].parent != wrist_id)
      return std::nullopt;
  const Vec3 local_length = unit(local_offset(asset.nodes[m->second]));
  const Vec3 local_width = unit(minus(local_offset(asset.nodes[i->second]),
                                      local_offset(asset.nodes[l->second])));
  if (norm(local_length) < .5f || norm(cross(local_length, local_width)) < .2f)
    return std::nullopt;
  const auto observed = [&](int from, int to) {
    const auto &a = hand.landmarks[from], &b = hand.landmarks[to];
    return Vec3{(b.x - a.x) * aspect, a.y - b.y, (a.z - b.z) * .65f};
  };
  const Vec3 palm_forward = observed(wrist, middle);
  const Vec3 palm_width = observed(little, index);
  if (norm(palm_forward) < .07f || norm(palm_width) < .04f ||
      norm(cross(unit(palm_forward), unit(palm_width))) < .15f)
    return std::nullopt; // edge-on/degenerate, keep previous wrist pose
  const Vec3 world_forward = unit(palm_forward),
             world_across = unit(palm_width);
  const auto world =
      global_before_correction(asset, wrist_id, arm_pose ? *arm_pose : Pose{});
  const Vec3 axis_x = unit({world[0], world[1], world[2]});
  const Vec3 axis_y = unit({world[4], world[5], world[6]});
  const Vec3 axis_z = unit({world[8], world[9], world[10]});
  if (norm(axis_x) < .5f || norm(axis_y) < .5f || norm(axis_z) < .5f)
    return std::nullopt;
  const Vec3 length = {dot(world_forward, axis_x), dot(world_forward, axis_y),
                       dot(world_forward, axis_z)};
  const Vec3 across = {dot(world_across, axis_x), dot(world_across, axis_y),
                       dot(world_across, axis_z)};
  const Quat aim = between(local_length, length, 1.05f);
  const auto rotate = [](Quat q, Vec3 v) -> Vec3 {
    const Vec3 u = {q.x, q.y, q.z};
    const Vec3 t = multiply_scalar(cross(u, v), 2.f);
    return Vec3{v.x + q.w * t.x + cross(u, t).x,
                v.y + q.w * t.y + cross(u, t).y,
                v.z + q.w * t.z + cross(u, t).z};
  };
  const auto perpendicular = [&](Vec3 v) {
    return unit(minus(v, multiply_scalar(length, dot(v, length))));
  };
  const Vec3 from = perpendicular(rotate(aim, local_width));
  const Vec3 to = perpendicular(across);
  if (norm(from) < .5f || norm(to) < .5f)
    return std::nullopt;
  const float twist = std::clamp(
      std::atan2(dot(cross(from, to), length), dot(from, to)), -1.00f, 1.00f);
  const float half = twist * .5f;
  const Quat spin = {length.x * std::sin(half), length.y * std::sin(half),
                     length.z * std::sin(half), std::cos(half)};
  // Local correction = palm-axis twist * primary hand-direction bend.
  const Quat combined = normalized(
      {spin.w * aim.x + aim.w * spin.x + spin.y * aim.z - spin.z * aim.y,
       spin.w * aim.y + aim.w * spin.y + spin.z * aim.x - spin.x * aim.z,
       spin.w * aim.z + aim.w * spin.z + spin.x * aim.y - spin.y * aim.x,
       spin.w * aim.w - spin.x * aim.x - spin.y * aim.y - spin.z * aim.z});
  // A real hand can turn further, but camera palm-depth gets noisy at
  // steep angles. Cap to a comfortable one-joint rotation.
  const float angle =
      2.f * std::acos(std::clamp(std::fabs(combined.w), 0.f, 1.f));
  return interpolate({}, combined, angle > 1.20f ? 1.20f / angle : 1.f);
}
Pose relaxed_body_pose(const Asset &asset, float degrees) {
  Pose relaxed;
  const float radians = std::clamp(degrees, 0.f, 85.f) * .01745329252f;
  if (radians < .01f)
    return relaxed;
  for (const char *upper : {"leftUpperArm", "rightUpperArm"}) {
    const char *child =
        std::string(upper) == "leftUpperArm" ? "leftLowerArm" : "rightLowerArm";
    auto arm = asset.bones.find(upper);
    auto elbow = asset.bones.find(child);
    if (arm == asset.bones.end() || elbow == asset.bones.end() ||
        arm->second < 0 || elbow->second < 0 ||
        size_t(arm->second) >= asset.nodes.size() ||
        size_t(elbow->second) >= asset.nodes.size() ||
        asset.nodes[elbow->second].parent != arm->second)
      continue;
    // Determine the arm's actual world rest direction from the rig,
    // instead of assuming a particular FBX/VRM local bone axis.
    const auto m = global_before_correction(asset, arm->second, {});
    const auto t = local_offset(asset.nodes[elbow->second]);
    const Vec3 rest = {m[0] * t.x + m[4] * t.y + m[8] * t.z,
                       m[1] * t.x + m[5] * t.y + m[9] * t.z,
                       m[2] * t.x + m[6] * t.y + m[10] * t.z};
    const float horizontal = std::hypot(rest.x, rest.z);
    if (horizontal < .03f)
      continue;
    const Vec3 desired = {rest.x / horizontal * std::cos(radians),
                          -std::sin(radians),
                          rest.z / horizontal * std::cos(radians)};
    aim_arm_bone(asset, relaxed, upper, child, desired, 1.6f);
    const bool left = std::string(upper) == "leftUpperArm";
    const char *hand = left ? "leftHand" : "rightHand";
    // Elbows remain gently bent inward, not vertically locked/straight.
    const Vec3 forearm = {left ? -.30f : .30f, -.94f, 0.f};
    aim_arm_bone(asset, relaxed, child, hand, forearm, .85f);
  }
  return relaxed;
}
} // namespace
float finger_flexion_sign(const Asset &asset, const std::string &bone) {
  const std::string side = bone.starts_with("left")    ? "left"
                           : bone.starts_with("right") ? "right"
                                                       : "";
  if (side.empty() || !asset.bones.contains(bone))
    return 0.f;
  const std::string_view stem = std::string_view(bone).substr(side.size());
  const bool thumb = stem.starts_with("Thumb");
  bool supported = false;
  std::string neighbor;
  bool use_parent = false;
  for (const char *digit : {"Index", "Middle", "Ring", "Little", "Thumb"}) {
    const std::string prefix = digit;
    if (!stem.starts_with(prefix))
      continue;
    const auto segment = stem.substr(prefix.size());
    if (thumb) {
      if (segment == "Metacarpal")
        neighbor = side + prefix + "Proximal";
      else if (segment == "Proximal")
        neighbor = side + prefix + "Distal";
      else if (segment == "Distal") {
        neighbor = side + prefix + "Proximal";
        use_parent = true;
      } else
        return 0.f;
    } else {
      if (segment == "Proximal")
        neighbor = side + prefix + "Intermediate";
      else if (segment == "Intermediate")
        neighbor = side + prefix + "Distal";
      else if (segment == "Distal") {
        neighbor = side + prefix + "Intermediate";
        use_parent = true;
      } else
        return 0.f;
    }
    supported = true;
    break;
  }
  if (!supported)
    return 0.f;

  const auto bone_index = [&](const std::string &name) -> int {
    const auto it = asset.bones.find(name);
    if (it == asset.bones.end() || it->second < 0 ||
        size_t(it->second) >= asset.nodes.size())
      return -1;
    return it->second;
  };
  const int hand = bone_index(side + "Hand");
  const int middle = bone_index(side + "MiddleProximal");
  const int index = bone_index(side + "IndexProximal");
  const int little = bone_index(side + "LittleProximal");
  const int joint = bone_index(bone);
  const int adjacent = bone_index(neighbor);
  if (hand < 0 || middle < 0 || index < 0 || little < 0 || joint < 0 ||
      adjacent < 0)
    return 0.f;
  if (asset.nodes[adjacent].parent != joint && !use_parent)
    return 0.f;
  if (use_parent && asset.nodes[joint].parent != adjacent)
    return 0.f;
  const auto position = [&](int node) {
    const auto world = global_before_correction(asset, node, {});
    return Vec3{world[12], world[13], world[14]};
  };
  // The humanoid hand plane is encoded by the wrist-to-middle direction
  // and index/pinky width in the model's ACTUAL authored rest skeleton.
  // Mirrored right hands reverse the cross-product winding. This yields
  // the anatomical palm-facing direction without assuming a particular
  // per-joint X-axis sign or relying on uncalibrated camera depth.
  const Vec3 forward = minus(position(middle), position(hand));
  const Vec3 across = minus(position(index), position(little));
  const Vec3 palm = unit(
      multiply_scalar(cross(forward, across), side == "right" ? -1.f : 1.f));
  if (norm(palm) < .5f || norm(forward) < 1.e-4f || norm(across) < 1.e-4f)
    return 0.f;
  const auto world = global_before_correction(asset, joint, {});
  const Vec3 axis = unit({world[0], world[1], world[2]});
  // Distal humanoid joints have no tip bone. The incoming segment is
  // their only observable authored tangent; other joints use the next
  // child segment. No fabricated fingertip position is needed.
  const Vec3 tangent =
      unit(use_parent ? minus(position(joint), position(adjacent))
                      : minus(position(adjacent), position(joint)));
  const float toward_palm = dot(cross(axis, tangent), palm);
  if (!std::isfinite(toward_palm) || std::fabs(toward_palm) < .15f)
    return 0.f; // Cannot establish a safe flex hinge from this rig
  return toward_palm > 0.f ? 1.f : -1.f;
}

Pose from_tracking(const Asset &asset, const TrackingSignal *face,
                   const TrackingSignal *body, float torso_strength) {
  Pose pose;
  apply_body(asset, body, pose, torso_strength);
  if (!face || face->kind != TrackingKind::Face ||
      face->data.landmarks.size() < 468)
    return pose;
  const auto &data = face->data;
  const auto blink_left = score(data, "eyeBlinkLeft");
  const auto blink_right = score(data, "eyeBlinkRight");
  const auto jaw = score(data, "jawOpen");
  const auto smile =
      (score(data, "mouthSmileLeft") + score(data, "mouthSmileRight")) * .5f;
  const auto pucker = score(data, "mouthPucker");
  const auto funnel = score(data, "mouthFunnel");
  // VRM expressions are authored per-model; don't assume all presets exist.
  first_expression(asset, pose, {"aa", "A", "a"}, jaw);
  first_expression(asset, pose, {"ou", "U", "u"}, std::max(pucker, funnel));
  first_expression(asset, pose, {"happy", "Joy", "joy"}, smile);
  const bool has_left = asset.expressions.contains("blinkLeft") ||
                        asset.expressions.contains("Blink_L");
  const bool has_right = asset.expressions.contains("blinkRight") ||
                         asset.expressions.contains("Blink_R");
  if (has_left)
    first_expression(asset, pose, {"blinkLeft", "Blink_L"}, blink_left);
  if (has_right)
    first_expression(asset, pose, {"blinkRight", "Blink_R"}, blink_right);
  if (!has_left || !has_right)
    first_expression(asset, pose, {"blink", "Blink"},
                     std::max(blink_left, blink_right));
  if (data.has_head_transform && asset.bones.contains("head")) {
    const auto &m = data.head_transform;
    // MediaPipe Tasks matrix is row-major: extract a bounded rotation.
    // Translation and scale from camera calibration are not VRM bone motion.
    bool finite = true;
    for (auto f : m)
      finite &= std::isfinite(f);
    if (finite) {
      // MediaPipe camera pitch and the glTF/VRM head bone use opposite
      // vertical rotation signs. Keep yaw and roll in their existing basis.
      const float pitch = -std::atan2(m[9], m[10]);
      // Camera/VRM horizontal handedness is reversed: keep the head
      // turning toward the same on-screen side as tracked arms/hands.
      // Do not change the separately verified pitch or roll signs.
      const float yaw = std::atan2(m[8], std::hypot(m[0], m[4]));
      const float roll = std::atan2(m[4], m[0]);
      pose.bone_rotation["head"] = from_euler(std::clamp(pitch, -.65f, .65f),
                                              std::clamp(yaw, -.85f, .85f),
                                              std::clamp(roll, -.55f, .55f));
    }
  }
  return pose;
}

Pose ease_face_pose(const Asset &asset, const TrackingSignal *face,
                    FacePoseState &state,
                    std::chrono::steady_clock::time_point now, float easing_ms,
                    float lost_return_ms, float lost_hold_ms) {
  if (!state.initialized) {
    state = {};
    state.initialized = true;
    state.last_frame = now;
    return state.current;
  }
  const float elapsed =
      std::chrono::duration<float>(now - state.last_frame).count();
  const float dt = std::clamp(elapsed, 0.f, .06f);
  state.last_frame = now;
  const float tracking_alpha = safe_alpha(dt, easing_ms);
  const float return_alpha =
      lost_return_ms > 0.f ? safe_alpha(dt, lost_return_ms) : 0.f;
  const auto should_return = [&](bool has_seen,
                                 std::chrono::steady_clock::time_point when) {
    return has_seen && lost_return_ms > 0.f &&
           std::chrono::duration<float, std::milli>(now - when).count() >=
               std::max(0.f, lost_hold_ms);
  };
  const bool tracked = face && face->kind == TrackingKind::Face &&
                       face->data.landmarks.size() >= 468;
  const Pose measured = tracked ? from_tracking(asset, face) : Pose{};

  if (const auto head = measured.bone_rotation.find("head");
      head != measured.bone_rotation.end()) {
    const auto prior = state.current.bone_rotation.find("head");
    const Quat old =
        prior != state.current.bone_rotation.end() ? prior->second : Quat{};
    state.current.bone_rotation["head"] =
        interpolate(old, head->second, tracking_alpha);
    state.has_head = true;
    state.head_seen = now;
  } else if (should_return(state.has_head, state.head_seen)) {
    const auto prior = state.current.bone_rotation.find("head");
    if (prior != state.current.bone_rotation.end())
      prior->second = interpolate(prior->second, {}, return_alpha);
  }

  // Distinguish a real observed zero score from a missing signal. Tracking
  // output with a valid blendshape list gently closes mouth/eyes, whereas
  // missing face data holds the last pose until the return delay expires.
  const bool expressions_tracked = tracked &&
                                   !face->data.blendshape_names.empty() &&
                                   !face->data.blendshapes.empty();
  if (expressions_tracked) {
    state.has_expressions = true;
    state.expression_seen = now;
    for (const auto &[name, weight] : measured.expressions) {
      const auto prior = state.current.expressions.find(name);
      const float old =
          prior != state.current.expressions.end() ? prior->second : 0.f;
      state.current.expressions[name] = old + (weight - old) * tracking_alpha;
    }
    // An absent target expression during valid tracking is a genuine
    // observed return toward zero, not an expired tracking frame.
    for (auto it = state.current.expressions.begin();
         it != state.current.expressions.end();) {
      if (!measured.expressions.contains(it->first))
        it->second *= 1.f - tracking_alpha;
      if (it->second < .0005f)
        it = state.current.expressions.erase(it);
      else
        ++it;
    }
  } else if (should_return(state.has_expressions, state.expression_seen)) {
    for (auto it = state.current.expressions.begin();
         it != state.current.expressions.end();) {
      it->second *= 1.f - return_alpha;
      if (it->second < .0005f)
        it = state.current.expressions.erase(it);
      else
        ++it;
    }
  }
  return state.current;
}

Pose ease_body_pose(const Asset &asset, const TrackingSignal *body,
                    BodyPoseState &state,
                    std::chrono::steady_clock::time_point now, float easing_ms,
                    float resting_degrees, const TrackingSignal *hands,
                    bool hands_drive_arms, bool mirror_hand_labels,
                    const TrackingSignal *face, float lost_return_ms,
                    float lost_hold_ms, float torso_strength) {
  resting_degrees = std::clamp(resting_degrees, 0.f, 85.f);
  if (!state.initialized || state.resting_degrees != resting_degrees) {
    state.current = relaxed_body_pose(asset, resting_degrees);
    state.last_frame = now;
    state.resting_degrees = resting_degrees;
    state.initialized = true;
    state.shoulder_reference_valid = false;
    // Preserve the initial image-center reference even if hips are invisible.
    if (body && body->kind == TrackingKind::Body &&
        body->data.landmarks.size() == 33 && reliable(*body, 11) &&
        reliable(*body, 12) &&
        body->data.landmarks[11].x > body->data.landmarks[12].x + .07f) {
      state.shoulder_reference_x =
          (body->data.landmarks[11].x + body->data.landmarks[12].x) * .5f;
      state.shoulder_reference_width = std::clamp(
          body->data.landmarks[11].x - body->data.landmarks[12].x, .10f, .85f);
      state.shoulder_reference_valid = true;
    }
    state.observed_at.clear();
    state.observed_real.clear();
    for (const auto &[name, _] : state.current.bone_rotation)
      state.observed_at[name] = now;
    return state.current;
  }
  const float elapsed =
      std::chrono::duration<float>(now - state.last_frame).count();
  // Limit one frame's easing step; resuming OBS after being suspended must
  // never teleport arms, even if the camera returns after many seconds.
  const float dt = std::clamp(elapsed, 0.f, .06f);
  state.last_frame = now;
  const bool valid_body = body && body->kind == TrackingKind::Body &&
                          body->data.landmarks.size() == 33;
  const float factor = safe_alpha(dt, easing_ms);
  const bool shoulders_seen =
      valid_body && reliable(*body, 11) && reliable(*body, 12) &&
      body->data.landmarks[11].x > body->data.landmarks[12].x + .07f;
  if (shoulders_seen && !state.shoulder_reference_valid) {
    state.shoulder_reference_x =
        (body->data.landmarks[11].x + body->data.landmarks[12].x) * .5f;
    state.shoulder_reference_width = std::clamp(
        body->data.landmarks[11].x - body->data.landmarks[12].x, .10f, .85f);
    state.shoulder_reference_valid = true;
  }
  Pose target =
      valid_body ? from_tracking(asset, nullptr, body, torso_strength) : Pose{};
  // When the hips are outside a desktop camera, calibrated lateral
  // shoulder displacement still drives gentle spine sway. This does not
  // pretend to measure pelvis position, root translation, or 3D depth.
  const bool hips_seen = valid_body && valid_chest(*body);
  if (shoulders_seen && !hips_seen && state.shoulder_reference_valid &&
      asset.bones.contains("spine") && torso_strength > .001f) {
    const float center =
        (body->data.landmarks[11].x + body->data.landmarks[12].x) * .5f;
    const float aspect =
        body->source_height
            ? std::clamp(float(body->source_width) / float(body->source_height),
                         .5f, 3.f)
            : 1.f;
    const float horizontal = (center - state.shoulder_reference_x) * aspect;
    const float shoulder_width =
        std::max(.14f, state.shoulder_reference_width * aspect);
    const float sway = std::clamp(
        -horizontal / shoulder_width * .55f * torso_strength, -.38f, .38f);
    if (std::fabs(sway) > .012f)
      target.bone_rotation["spine"] = from_euler(0, 0, sway);
  }
  const auto relaxed = relaxed_body_pose(asset, resting_degrees);
  bool left_wrist = false, right_wrist = false;
  if (hands_drive_arms) {
    for (const char *side : {"left", "right"}) {
      const auto *hand = hand_for(hands, side, mirror_hand_labels);
      if (!hand || !valid_hand_point(*hand, 0))
        continue;
      aim_from_wrist(asset, target, side, *hand, valid_body ? body : nullptr,
                     face);
      if (side[0] == 'l')
        left_wrist = true;
      else
        right_wrist = true;
    }
  }
  const auto advance = [&](const char *name, bool seen) {
    if (!asset.bones.contains(name))
      return;
    const auto found = target.bone_rotation.find(name);
    const auto old = state.current.bone_rotation.find(name);
    const Quat current =
        old == state.current.bone_rotation.end() ? Quat{} : old->second;
    if (seen) {
      state.observed_at[name] = now;
      state.observed_real.insert(name);
    }
    if (seen && factor > 0.f) {
      state.current.bone_rotation[name] = interpolate(
          current, found != target.bone_rotation.end() ? found->second : Quat{},
          factor);
    } else if (!seen && lost_return_ms > 0.f) {
      const auto last = time_or(state.observed_at, name, now);
      const float idle =
          std::chrono::duration<float, std::milli>(now - last).count();
      if (idle >= std::max(0.f, lost_hold_ms)) {
        const auto rest = relaxed.bone_rotation.find(name);
        state.current.bone_rotation[name] = interpolate(
            current,
            rest != relaxed.bone_rotation.end() ? rest->second : Quat{},
            safe_alpha(dt, lost_return_ms));
      }
    }
  };
  const bool shoulder = valid_body && valid_chest(*body);
  // Strength zero deliberately eases toward neutral if shoulders are
  // present; it must not freeze a previously tracked spine angle.
  advance("spine",
          shoulder || (shoulders_seen && state.shoulder_reference_valid));
  advance("chest", shoulder);
  for (const bool left : {true, false}) {
    const char *upper = left ? "leftUpperArm" : "rightUpperArm";
    const char *lower = left ? "leftLowerArm" : "rightLowerArm";
    const bool hand_override = left ? left_wrist : right_wrist;
    const bool upper_seen =
        hand_override ||
        (valid_body && valid_segment(*body, left ? 11 : 12, left ? 13 : 14));
    const bool lower_seen =
        hand_override ||
        (valid_body && valid_segment(*body, left ? 11 : 12, left ? 13 : 14) &&
         valid_segment(*body, left ? 13 : 14, left ? 15 : 16));
    // Even an open hand with no usable rig aim cannot force an arm to rest.
    advance(upper, upper_seen);
    advance(lower, lower_seen);
  }
  return state.current;
}

Pose ease_hand_pose(const Asset &asset, const TrackingSignal *hands,
                    HandPoseState &state,
                    std::chrono::steady_clock::time_point now, float easing_ms,
                    bool mirror_labels, float lost_return_ms,
                    float lost_hold_ms, bool rotate_wrists,
                    const Pose *arm_pose, bool animate_fingers) {
  if (!state.initialized || state.mirror_labels != mirror_labels ||
      state.wrists_enabled != rotate_wrists ||
      state.fingers_enabled != animate_fingers) {
    state = {};
    state.initialized = true;
    state.mirror_labels = mirror_labels;
    state.wrists_enabled = rotate_wrists;
    state.fingers_enabled = animate_fingers;
    state.last_frame = now;
    return state.current;
  }
  const float dt = std::clamp(
      std::chrono::duration<float>(now - state.last_frame).count(), 0.f, .06f);
  state.last_frame = now;
  const float alpha = safe_alpha(dt, easing_ms);
  const std::array<std::pair<const char *, int>, 5> fingers = {{{"Index", 5},
                                                                {"Middle", 9},
                                                                {"Ring", 13},
                                                                {"Little", 17},
                                                                {"Thumb", 1}}};
  for (const char *side : {"left", "right"}) {
    const HandData *hand = hand_for(hands, side, mirror_labels);
    if (rotate_wrists) {
      const std::string bone = std::string(side) + "Hand";
      const float aspect =
          hands && hands->source_width && hands->source_height
              ? std::clamp(float(hands->source_width) / hands->source_height,
                           .7f, 2.4f)
              : 1.3333f;
      const auto target =
          hand ? wrist_attitude(asset, side, *hand, aspect, arm_pose)
               : std::optional<Quat>{};
      const auto old = state.current.bone_rotation.find(bone);
      const Quat current =
          old == state.current.bone_rotation.end() ? Quat{} : old->second;
      if (target) {
        state.observed_at[bone] = now;
        state.current.bone_rotation[bone] =
            interpolate(current, *target, alpha);
      } else if (lost_return_ms > 0.f &&
                 old != state.current.bone_rotation.end()) {
        const auto seen = state.observed_at.find(bone);
        if (seen != state.observed_at.end() &&
            std::chrono::duration<float, std::milli>(now - seen->second)
                    .count() >= std::max(0.f, lost_hold_ms))
          state.current.bone_rotation[bone] =
              interpolate(current, {}, safe_alpha(dt, lost_return_ms));
      }
    }
    if (!animate_fingers)
      continue;
    for (const auto &[finger, base] : fingers) {
      const bool thumb = std::string(finger) == "Thumb";
      const std::array<std::string, 3> segments =
          thumb ? std::array<std::string, 3>{"Metacarpal", "Proximal", "Distal"}
                : std::array<std::string, 3>{"Proximal", "Intermediate",
                                             "Distal"};
      const std::array<int, 3> prev =
          thumb ? std::array<int, 3>{0, 1, 2}
                : std::array<int, 3>{0, base, base + 1};
      const std::array<int, 3> center =
          thumb ? std::array<int, 3>{1, 2, 3}
                : std::array<int, 3>{base, base + 1, base + 2};
      const std::array<int, 3> next =
          thumb ? std::array<int, 3>{2, 3, 4}
                : std::array<int, 3>{base + 1, base + 2, base + 3};
      for (int k = 0; k < 3; ++k) {
        const std::string bone = std::string(side) + finger + segments[k];
        if (!asset.bones.contains(bone))
          continue;
        const float bend = hand ? finger_bend(*hand, prev[k], center[k],
                                              next[k], k == 0 ? .23f : .10f)
                                : -1.f;
        const auto old = state.current.bone_rotation.find(bone);
        const Quat current =
            old == state.current.bone_rotation.end() ? Quat{} : old->second;
        if (bend >= 0.f) {
          // The authored local-X hinge changes sign between models and
          // sometimes even adjacent joints (Otaku's index MCP vs PIP).
          // Cache the sign so bone-space hierarchy isn't walked per frame.
          auto orientation = state.finger_flex_signs.find(bone);
          if (orientation == state.finger_flex_signs.end()) {
            const float sign = finger_flexion_sign(asset, bone);
            orientation = state.finger_flex_signs.emplace(bone, sign).first;
          }
          if (orientation->second == 0.f)
            continue;
          state.observed_at[bone] = now;
          const auto target = from_euler(bend * orientation->second, 0, 0);
          const auto next_rotation = interpolate(current, target, alpha);
          if (std::fabs(next_rotation.x) + std::fabs(next_rotation.y) +
                  std::fabs(next_rotation.z) >
              .005f)
            state.current.bone_rotation[bone] = next_rotation;
          else
            state.current.bone_rotation.erase(bone);
        } else if (lost_return_ms > 0.f &&
                   old != state.current.bone_rotation.end()) {
          const auto seen = state.observed_at.find(bone);
          if (seen != state.observed_at.end() &&
              std::chrono::duration<float, std::milli>(now - seen->second)
                      .count() >= std::max(0.f, lost_hold_ms)) {
            const auto next_rotation =
                interpolate(current, {}, safe_alpha(dt, lost_return_ms));
            if (std::fabs(next_rotation.x) + std::fabs(next_rotation.y) +
                    std::fabs(next_rotation.z) >
                .005f)
              state.current.bone_rotation[bone] = next_rotation;
            else
              state.current.bone_rotation.erase(bone);
          }
        }
      }
    }
  }
  return state.current;
}
} // namespace sight::vrm
