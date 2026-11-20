#include "vrm_asset.hpp"
#define TINYGLTF_NO_INCLUDE_JSON
#define TINYGLTF_NO_INCLUDE_STB_IMAGE
#define TINYGLTF_NO_STB_IMAGE_WRITE
#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_STDIO
#include "vendor/stb_image.h"
#include <nlohmann/json.hpp>
#define TINYGLTF_IMPLEMENTATION
#include "vendor/tiny_gltf.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace sight::vrm {
namespace {
using Value = tinygltf::Value;
const Value &field(const Value &v, const char *key) {
  static const Value missing;
  return v.IsObject() ? v.Get(key) : missing;
}
float number(const Value &v, float fallback = 0) {
  return v.IsNumber() ? float(v.GetNumberAsDouble()) : fallback;
}
int integer(const Value &v, int fallback = -1) {
  return v.IsNumber() ? v.GetNumberAsInt() : fallback;
}
std::string string(const Value &v, const std::string &fallback = "") {
  return v.IsString() ? v.Get<std::string>() : fallback;
}
bool valid(int i, size_t n) { return i >= 0 && size_t(i) < n; }
Vec3 vec(const std::vector<double> &x, Vec3 fallback = {}) {
  if (x.size() != 3)
    return fallback;
  return {float(x[0]), float(x[1]), float(x[2])};
}
Quat quat(const std::vector<double> &x) {
  if (x.size() != 4)
    return {};
  return {float(x[0]), float(x[1]), float(x[2]), float(x[3])};
}
float component(const uint8_t *ptr, int type, bool normalized) {
  switch (type) {
  case TINYGLTF_COMPONENT_TYPE_FLOAT: {
    float v;
    std::memcpy(&v, ptr, 4);
    return std::isfinite(v) ? v : 0;
  }
  case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
    return normalized ? float(*ptr) / 255.f : float(*ptr);
  case TINYGLTF_COMPONENT_TYPE_BYTE: {
    int8_t v;
    std::memcpy(&v, ptr, 1);
    return normalized ? std::max(-1.f, float(v) / 127.f) : float(v);
  }
  case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: {
    uint16_t v;
    std::memcpy(&v, ptr, 2);
    return normalized ? float(v) / 65535.f : float(v);
  }
  case TINYGLTF_COMPONENT_TYPE_SHORT: {
    int16_t v;
    std::memcpy(&v, ptr, 2);
    return normalized ? std::max(-1.f, float(v) / 32767.f) : float(v);
  }
  case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT: {
    uint32_t v;
    std::memcpy(&v, ptr, 4);
    return float(v);
  }
  default:
    throw std::runtime_error("Unsupported vertex accessor component type");
  }
}
struct Reader {
  const tinygltf::Accessor *a = nullptr;
  const uint8_t *ptr = nullptr;
  size_t stride = 0, components = 0, component_bytes = 0;
  Reader(const tinygltf::Model &model, int index) {
    if (!valid(index, model.accessors.size()))
      throw std::runtime_error("Bad glTF accessor index");
    a = &model.accessors[index];
    if (a->sparse.isSparse || !valid(a->bufferView, model.bufferViews.size()))
      throw std::runtime_error("Sparse or missing glTF accessor unsupported");
    const auto &view = model.bufferViews[a->bufferView];
    if (!valid(view.buffer, model.buffers.size()))
      throw std::runtime_error("Bad glTF buffer");
    const auto &bytes = model.buffers[view.buffer].data;
    components = tinygltf::GetNumComponentsInType(a->type);
    component_bytes = tinygltf::GetComponentSizeInBytes(a->componentType);
    if (!components || !component_bytes)
      throw std::runtime_error("Bad glTF accessor type");
    const size_t packed = components * component_bytes;
    stride = view.byteStride ? view.byteStride : packed;
    if (stride < packed || a->byteOffset > view.byteLength ||
        view.byteOffset > bytes.size() ||
        view.byteLength > bytes.size() - view.byteOffset)
      throw std::runtime_error("Invalid glTF buffer bounds");
    const size_t start = view.byteOffset + a->byteOffset;
    const size_t end = view.byteOffset + view.byteLength;
    if (a->count && (start > end || packed > end - start ||
                     (a->count - 1) > (end - start - packed) / stride))
      throw std::runtime_error("glTF accessor exceeds buffer");
    ptr = bytes.data() + start;
  }
  float get(size_t i, size_t c = 0) const {
    if (i >= a->count || c >= components)
      throw std::runtime_error("glTF vertex index out of range");
    return component(ptr + i * stride + c * component_bytes, a->componentType,
                     a->normalized);
  }
  Vec3 position(size_t i) const {
    if (components != 3)
      throw std::runtime_error("Expected vec3 glTF attribute");
    return {get(i), get(i, 1), get(i, 2)};
  }
};
} // namespace

Mat4 identity() { return {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}; }
Mat4 multiply(const Mat4 &a, const Mat4 &b) {
  Mat4 m{};
  for (int c = 0; c < 4; ++c)
    for (int r = 0; r < 4; ++r)
      for (int k = 0; k < 4; ++k)
        m[c * 4 + r] += a[k * 4 + r] * b[c * 4 + k];
  return m;
}
Vec3 transform_point(const Mat4 &m, Vec3 p) {
  return {m[0] * p.x + m[4] * p.y + m[8] * p.z + m[12],
          m[1] * p.x + m[5] * p.y + m[9] * p.z + m[13],
          m[2] * p.x + m[6] * p.y + m[10] * p.z + m[14]};
}
Quat from_euler(float x, float y, float z) {
  const auto cx = std::cos(x / 2), sx = std::sin(x / 2);
  const auto cy = std::cos(y / 2), sy = std::sin(y / 2);
  const auto cz = std::cos(z / 2), sz = std::sin(z / 2);
  return {sx * cy * cz - cx * sy * sz, cx * sy * cz + sx * cy * sz,
          cx * cy * sz - sx * sy * cz, cx * cy * cz + sx * sy * sz};
}
Mat4 trs(Vec3 t, Quat q, Vec3 s) {
  const float norm = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w;
  const float k = norm > 1e-9f ? 2.f / norm : 0.f;
  const float xx = q.x * q.x * k, yy = q.y * q.y * k, zz = q.z * q.z * k;
  const float xy = q.x * q.y * k, xz = q.x * q.z * k, yz = q.y * q.z * k;
  const float wx = q.w * q.x * k, wy = q.w * q.y * k, wz = q.w * q.z * k;
  return {(1 - yy - zz) * s.x,
          (xy + wz) * s.x,
          (xz - wy) * s.x,
          0,
          (xy - wz) * s.y,
          (1 - xx - zz) * s.y,
          (yz + wx) * s.y,
          0,
          (xz + wy) * s.z,
          (yz - wx) * s.z,
          (1 - xx - yy) * s.z,
          0,
          t.x,
          t.y,
          t.z,
          1};
}
bool load(const std::string &filename, Asset &out, std::string &error) {
  error.clear();
  out = {};
  try {
    tinygltf::TinyGLTF parser;
    tinygltf::Model model;
    std::string warning;
    if (!parser.LoadBinaryFromFile(&model, &error, &warning, filename))
      throw std::runtime_error(error.empty() ? "Failed parsing VRM GLB"
                                             : error);
    const auto v0 = model.extensions.find("VRM");
    const auto v1 = model.extensions.find("VRMC_vrm");
    if (v0 == model.extensions.end() && v1 == model.extensions.end())
      throw std::runtime_error("VRM 0.x / VRM 1.0 extension missing");
    out.vrm_version = v1 != model.extensions.end() ? 1 : 0;
    const auto &vrm = out.vrm_version ? v1->second : v0->second;
    const auto &bones = field(field(vrm, "humanoid"), "humanBones");
    if (out.vrm_version == 1 && bones.IsObject()) {
      for (const auto &[bone, info] : bones.Get<Value::Object>()) {
        const int index = integer(field(info, "node"));
        if (valid(index, model.nodes.size()))
          out.bones[bone] = index;
      }
    } else if (bones.IsArray()) {
      for (const auto &item : bones.Get<Value::Array>()) {
        const int index = integer(field(item, "node"));
        if (valid(index, model.nodes.size()))
          out.bones[string(field(item, "bone"))] = index;
      }
    }
    if (out.bones.empty())
      throw std::runtime_error("VRM humanoid mapping missing");
    out.nodes.resize(model.nodes.size());
    for (size_t i = 0; i < model.nodes.size(); ++i) {
      const auto &in = model.nodes[i];
      auto &node = out.nodes[i];
      node.children = in.children;
      node.translation = vec(in.translation);
      node.scale = vec(in.scale, {1, 1, 1});
      node.rotation = quat(in.rotation);
      if (in.matrix.size() == 16) {
        node.use_matrix = true;
        for (int k = 0; k < 16; ++k)
          node.matrix[k] = float(in.matrix[k]);
      }
      for (const int child : node.children) {
        if (!valid(child, out.nodes.size()) || child == int(i))
          throw std::runtime_error("Invalid VRM scene graph");
        if (out.nodes[child].parent >= 0 && out.nodes[child].parent != int(i))
          throw std::runtime_error("VRM node has multiple parents");
        out.nodes[child].parent = int(i);
      }
    }
    for (size_t i = 0; i < out.nodes.size(); ++i) {
      int node = int(i);
      for (size_t depth = 0; node >= 0 && depth < out.nodes.size(); ++depth)
        node = out.nodes[node].parent;
      if (node >= 0)
        throw std::runtime_error("Cyclic VRM hierarchy");
    }
    for (const auto &sk : model.skins) {
      Skin skin;
      skin.joints = sk.joints;
      skin.inverse_bind.resize(skin.joints.size(), identity());
      if (sk.inverseBindMatrices >= 0) {
        Reader ibm(model, sk.inverseBindMatrices);
        if (ibm.components != 16 || ibm.a->count < skin.joints.size())
          throw std::runtime_error("Invalid glTF inverse bind matrices");
        for (size_t j = 0; j < skin.joints.size(); ++j)
          for (int k = 0; k < 16; ++k)
            skin.inverse_bind[j][k] = ibm.get(j, k);
      }
      for (const int joint : skin.joints)
        if (!valid(joint, model.nodes.size()))
          throw std::runtime_error("Bad glTF skin joint");
      out.skins.push_back(std::move(skin));
    }
    for (const auto &raw : model.images) {
      Image img;
      img.width = raw.width;
      img.height = raw.height;
      if (raw.width > 0 && raw.height > 0 && raw.component >= 3 &&
          raw.bits == 8 &&
          uint64_t(raw.width) * uint64_t(raw.height) <= 16000000) {
        if (raw.image.size() < size_t(img.width) * img.height * raw.component)
          throw std::runtime_error("Truncated VRM image data");
        img.rgba.resize(size_t(img.width) * img.height * 4);
        for (size_t i = 0; i < size_t(img.width) * img.height; ++i) {
          img.rgba[i * 4 + 0] = raw.image[i * raw.component];
          img.rgba[i * 4 + 1] = raw.image[i * raw.component + 1];
          img.rgba[i * 4 + 2] = raw.image[i * raw.component + 2];
          img.rgba[i * 4 + 3] =
              raw.component >= 4 ? raw.image[i * raw.component + 3] : 255;
        }
      }
      out.images.push_back(std::move(img));
    }
    for (const auto &raw : model.materials) {
      Material mat;
      mat.blend = raw.alphaMode == "BLEND";
      mat.mask = raw.alphaMode == "MASK";
      mat.alpha_cutoff = float(raw.alphaCutoff);
      mat.double_sided = raw.doubleSided;
      for (int i = 0; i < 4; ++i)
        mat.color[i] = float(raw.pbrMetallicRoughness.baseColorFactor[i]);
      const int texture = raw.pbrMetallicRoughness.baseColorTexture.index;
      if (valid(texture, model.textures.size())) {
        const int image = model.textures[texture].source;
        if (valid(image, out.images.size()))
          mat.image = image;
      }
      out.materials.push_back(mat);
    }
    for (size_t node_id = 0; node_id < model.nodes.size(); ++node_id) {
      const auto &n = model.nodes[node_id];
      if (!valid(n.mesh, model.meshes.size()))
        continue;
      const auto &mesh = model.meshes[n.mesh];
      for (const auto &prim : mesh.primitives) {
        if (prim.mode != -1 && prim.mode != TINYGLTF_MODE_TRIANGLES)
          continue;
        const auto positions = prim.attributes.find("POSITION");
        if (positions == prim.attributes.end())
          continue;
        Reader pos(model, positions->second);
        if (pos.components != 3 || pos.a->count > 1000000)
          throw std::runtime_error("Invalid VRM position vertex count");
        Surface surf;
        surf.mesh = n.mesh;
        surf.node = int(node_id);
        surf.skin = n.skin;
        surf.material = prim.material;
        surf.vertices.resize(pos.a->count);
        for (size_t i = 0; i < pos.a->count; ++i)
          surf.vertices[i].position = pos.position(i);
        if (auto it = prim.attributes.find("NORMAL");
            it != prim.attributes.end()) {
          Reader normal(model, it->second);
          if (normal.a->count != pos.a->count)
            throw std::runtime_error("VRM normal count mismatch");
          for (size_t i = 0; i < pos.a->count; ++i)
            surf.vertices[i].normal = normal.position(i);
        }
        if (auto it = prim.attributes.find("TEXCOORD_0");
            it != prim.attributes.end()) {
          Reader uv(model, it->second);
          if (uv.a->count != pos.a->count || uv.components != 2)
            throw std::runtime_error("VRM UV count mismatch");
          for (size_t i = 0; i < pos.a->count; ++i) {
            surf.vertices[i].u = uv.get(i);
            surf.vertices[i].v = uv.get(i, 1);
          }
        }
        const auto joint = prim.attributes.find("JOINTS_0");
        const auto weight = prim.attributes.find("WEIGHTS_0");
        if (joint != prim.attributes.end() && weight != prim.attributes.end()) {
          Reader j(model, joint->second), w(model, weight->second);
          if (j.components != 4 || w.components != 4 ||
              j.a->count != pos.a->count || w.a->count != pos.a->count)
            throw std::runtime_error("VRM skin attribute mismatch");
          for (size_t i = 0; i < pos.a->count; ++i)
            for (int k = 0; k < 4; ++k) {
              surf.vertices[i].joints[k] = int(j.get(i, k));
              surf.vertices[i].weights[k] = w.get(i, k);
            }
        }
        if (prim.indices >= 0) {
          Reader ix(model, prim.indices);
          if (ix.components != 1 || ix.a->count % 3)
            throw std::runtime_error("Invalid VRM triangle index count");
          for (size_t i = 0; i < ix.a->count; ++i) {
            const auto index = uint32_t(ix.get(i));
            if (index >= surf.vertices.size())
              throw std::runtime_error("VRM index out of range");
            surf.triangles.push_back(index);
          }
        } else {
          if (surf.vertices.size() % 3)
            throw std::runtime_error("Unindexed triangle size invalid");
          for (uint32_t i = 0; i < surf.vertices.size(); ++i)
            surf.triangles.push_back(i);
        }
        if (prim.targets.size() > 128)
          throw std::runtime_error("Too many VRM morph targets");
        for (const auto &target : prim.targets) {
          std::vector<Vec3> delta(pos.a->count);
          if (auto it = target.find("POSITION"); it != target.end()) {
            Reader m(model, it->second);
            if (m.a->count != pos.a->count)
              throw std::runtime_error("VRM morph count mismatch");
            for (size_t i = 0; i < pos.a->count; ++i)
              delta[i] = m.position(i);
          }
          surf.morph_positions.push_back(std::move(delta));
        }
        out.surfaces.push_back(std::move(surf));
      }
    }
    if (out.surfaces.empty())
      throw std::runtime_error("No drawable VRM triangle meshes");
    const auto add_bind = [&](const std::string &name, const Value &item,
                              bool v1) {
      if (!item.IsObject())
        return;
      const auto &binds = field(item, v1 ? "morphTargetBinds" : "binds");
      if (!binds.IsArray())
        return;
      for (const auto &bind : binds.Get<Value::Array>()) {
        Bind b;
        b.mesh = v1 ? -1 : integer(field(bind, "mesh"));
        b.node = v1 ? integer(field(bind, "node")) : -1;
        b.target = integer(field(bind, "index"));
        b.weight =
            number(field(bind, "weight"), v1 ? 1 : 100) / (v1 ? 1.f : 100.f);
        if (b.target >= 0 && std::isfinite(b.weight))
          out.expressions[name].push_back(b);
      }
    };
    if (out.vrm_version == 0) {
      const auto &groups =
          field(field(vrm, "blendShapeMaster"), "blendShapeGroups");
      if (groups.IsArray())
        for (const auto &group : groups.Get<Value::Array>())
          add_bind(string(field(group, "name")), group, false);
    } else {
      const auto &expr = field(vrm, "expressions");
      for (const char *type : {"preset", "custom"}) {
        const auto &list = field(expr, type);
        if (list.IsObject())
          for (const auto &[name, item] : list.Get<Value::Object>())
            add_bind(name, item, true);
      }
    }
    return true;
  } catch (const std::exception &e) {
    error = e.what();
    out = {};
    return false;
  }
}
std::vector<FrameSurface> deform(const Asset &asset, const Pose &pose) {
  std::vector<Mat4> global(asset.nodes.size());
  std::vector<uint8_t> visited(asset.nodes.size(), 0);
  auto visit = [&](auto &&self, int n, int depth) -> void {
    if (depth > int(asset.nodes.size()) || visited[n] == 2)
      return;
    if (visited[n] == 1)
      throw std::runtime_error("Cycle in VRM nodes");
    visited[n] = 1;
    const auto &node = asset.nodes[n];
    auto local = node.use_matrix
                     ? node.matrix
                     : trs(node.translation, node.rotation, node.scale);
    for (const auto &[bone, index] : asset.bones)
      if (index == n) {
        if (auto it = pose.bone_rotation.find(bone);
            it != pose.bone_rotation.end())
          local = multiply(local, trs({}, it->second, {1, 1, 1}));
        break;
      }
    if (node.parent >= 0) {
      self(self, node.parent, depth + 1);
      global[n] = multiply(global[node.parent], local);
    } else
      global[n] = local;
    visited[n] = 2;
  };
  for (size_t i = 0; i < asset.nodes.size(); ++i)
    visit(visit, int(i), 0);
  std::vector<FrameSurface> out(asset.surfaces.size());
  for (size_t s = 0; s < asset.surfaces.size(); ++s) {
    const auto &surface = asset.surfaces[s];
    auto &dst = out[s].positions;
    dst.resize(surface.vertices.size());
    std::vector<float> weights(surface.morph_positions.size(), 0.f);
    for (const auto &[name, value] : pose.expressions) {
      const auto it = asset.expressions.find(name);
      if (it == asset.expressions.end() || value <= 0)
        continue;
      for (const auto &bind : it->second) {
        if (bind.target < 0 || size_t(bind.target) >= weights.size())
          continue;
        if (bind.node >= 0 && bind.node != surface.node)
          continue;
        if (bind.mesh >= 0 && bind.mesh != surface.mesh)
          continue;
        weights[bind.target] =
            std::clamp(weights[bind.target] + bind.weight * value, 0.f, 1.f);
      }
    }
    for (size_t i = 0; i < dst.size(); ++i) {
      auto p = surface.vertices[i].position;
      for (size_t m = 0; m < weights.size(); ++m) {
        const auto w = weights[m];
        if (w <= 0)
          continue;
        const auto &d = surface.morph_positions[m][i];
        p.x += d.x * w;
        p.y += d.y * w;
        p.z += d.z * w;
      }
      const auto &v = surface.vertices[i];
      if (valid(surface.skin, asset.skins.size())) {
        const auto &skin = asset.skins[surface.skin];
        Vec3 mixed{};
        float sum = 0;
        for (int k = 0; k < 4; ++k) {
          const float w = v.weights[k];
          if (w <= 0 || !valid(v.joints[k], skin.joints.size()))
            continue;
          const int joint = skin.joints[v.joints[k]];
          const auto skinned = transform_point(
              multiply(global[joint], skin.inverse_bind[v.joints[k]]), p);
          mixed.x += skinned.x * w;
          mixed.y += skinned.y * w;
          mixed.z += skinned.z * w;
          sum += w;
        }
        dst[i] = sum > 0 ? Vec3{mixed.x / sum, mixed.y / sum, mixed.z / sum}
                         : transform_point(global[surface.node], p);
      } else
        dst[i] = transform_point(global[surface.node], p);
    }
  }
  return out;
}
} // namespace sight::vrm
