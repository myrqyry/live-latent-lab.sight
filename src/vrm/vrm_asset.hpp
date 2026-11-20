#pragma once
#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace sight::vrm {
struct Vec3 {
  float x = 0, y = 0, z = 0;
};
struct Quat {
  float x = 0, y = 0, z = 0, w = 1;
};
using Mat4 = std::array<float, 16>;
struct Node {
  int parent = -1;
  std::vector<int> children;
  Vec3 translation{};
  Quat rotation{};
  Vec3 scale{1, 1, 1};
  Mat4 matrix{};
  bool use_matrix = false;
};
struct Vertex {
  Vec3 position{}, normal{};
  float u = 0, v = 0;
  std::array<int, 4> joints{};
  std::array<float, 4> weights{};
};
struct Surface {
  int node = -1, skin = -1, mesh = -1, material = -1;
  std::vector<Vertex> vertices;
  std::vector<uint32_t> triangles;
  std::vector<std::vector<Vec3>> morph_positions;
};
struct Skin {
  std::vector<int> joints;
  std::vector<Mat4> inverse_bind;
};
struct Image {
  int width = 0, height = 0;
  std::vector<uint8_t> rgba;
};
struct Material {
  int image = -1;
  std::array<float, 4> color{1, 1, 1, 1};
  bool blend = false, mask = false, double_sided = false;
  float alpha_cutoff = .5f;
};
struct Bind {
  int mesh = -1, node = -1, target = -1;
  float weight = 0;
};
struct Asset {
  int vrm_version = 0;
  std::vector<Node> nodes;
  std::vector<Skin> skins;
  std::vector<Surface> surfaces;
  std::vector<Image> images;
  std::vector<Material> materials;
  std::map<std::string, int> bones;
  std::map<std::string, std::vector<Bind>> expressions;
};
struct FrameSurface {
  std::vector<Vec3> positions;
};
struct Pose {
  std::map<std::string, Quat> bone_rotation;
  std::map<std::string, float> expressions;
};
Mat4 identity();
Mat4 multiply(const Mat4 &a, const Mat4 &b);
Vec3 transform_point(const Mat4 &m, Vec3 p);
Quat from_euler(float x, float y, float z);
Mat4 trs(Vec3 t, Quat q, Vec3 s);
bool load(const std::string &filename, Asset &out, std::string &error);
std::vector<FrameSurface> deform(const Asset &asset, const Pose &pose);
} // namespace sight::vrm
