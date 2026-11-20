#include "vrm/vrm_asset.hpp"
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <vector>
using sight::vrm::Asset;
using sight::vrm::Pose;
using sight::vrm::Vec3;

static void word(std::vector<uint8_t> &bytes, uint32_t value) {
  for (int i = 0; i < 4; ++i)
    bytes.push_back(uint8_t(value >> (i * 8)));
}
static std::filesystem::path fixture(const std::filesystem::path &root, bool v1,
                                     bool cycle = false) {
  nlohmann::json ext =
      v1 ? nlohmann::json{{"VRMC_vrm",
                           {{"specVersion", "1.0"},
                            {"humanoid",
                             {{"humanBones", {{"head", {{"node", 0}}}}}}},
                            {"expressions",
                             {{"preset",
                               {{"aa",
                                 {{"morphTargetBinds",
                                   nlohmann::json::array(
                                       {{{"node", 0},
                                         {"index", 0},
                                         {"weight", 0.75}}})}}}}}}}}}}
         : nlohmann::json{
               {"VRM",
                {{"specVersion", "0.0"},
                 {"humanoid",
                  {{"humanBones",
                    nlohmann::json::array({{{"bone", "head"}, {"node", 0}}})}}},
                 {"blendShapeMaster",
                  {{"blendShapeGroups",
                    nlohmann::json::array(
                        {{{"name", "A"},
                          {"binds",
                           nlohmann::json::array({{{"mesh", 0},
                                                   {"index", 0},
                                                   {"weight", 75}}})}}})}}}}}};
  nlohmann::json j = {
      {"asset", {{"version", "2.0"}}},
      {"extensionsUsed", nlohmann::json::array({v1 ? "VRMC_vrm" : "VRM"})},
      {"extensions", ext},
      {"buffers", nlohmann::json::array({{{"byteLength", 68}}})},
      {"bufferViews",
       nlohmann::json::array(
           {{{"buffer", 0}, {"byteOffset", 0}, {"byteLength", 36}},
            {{"buffer", 0}, {"byteOffset", 36}, {"byteLength", 6}},
            {{"buffer", 0}, {"byteOffset", 44}, {"byteLength", 24}}})},
      {"accessors",
       nlohmann::json::array({{{"bufferView", 0},
                               {"componentType", 5126},
                               {"count", 3},
                               {"type", "VEC3"},
                               {"min", nlohmann::json::array({0, 0, 0})},
                               {"max", nlohmann::json::array({1, 1, 0})}},
                              {{"bufferView", 1},
                               {"componentType", 5123},
                               {"count", 3},
                               {"type", "SCALAR"}},
                              {{"bufferView", 2},
                               {"componentType", 5126},
                               {"count", 3},
                               {"type", "VEC2"}}})},
      {"materials",
       nlohmann::json::array(
           {{{"pbrMetallicRoughness",
              {{"baseColorFactor", nlohmann::json::array({1.0, 1.0, 1.0, 1.0})},
               {"baseColorTexture", {{"index", 0}}}}}}})},
      {"images",
       nlohmann::json::array(
           {{{"uri",
              "data:image/"
              "png;base64,"
              "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAYAAABytg0kAAAAG0lEQVR4nGP4bxTw"
              "3+hDwH8Go4AP//9/sPkPAFiRCo2JnZK/AAAAAElFTkSuQmCC"}}})},
      {"textures", nlohmann::json::array({{{"source", 0}}})},
      {"meshes",
       nlohmann::json::array(
           {{{"primitives",
              nlohmann::json::array(
                  {{{"attributes", {{"POSITION", 0}, {"TEXCOORD_0", 2}}},
                    {"indices", 1},
                    {"material", 0},
                    {"mode", 4},
                    {"targets",
                     nlohmann::json::array({{{"POSITION", 0}}})}}})}}})},
      {"nodes", nlohmann::json::array({{{"mesh", 0}}})},
      {"scenes",
       nlohmann::json::array({{{"nodes", nlohmann::json::array({0})}}})},
      {"scene", 0}};
  if (cycle)
    j["nodes"] = nlohmann::json::array(
        {{{"mesh", 0}, {"children", nlohmann::json::array({1})}},
         {{"children", nlohmann::json::array({0})}}});
  std::vector<uint8_t> binary(68, 0);
  const float vertices[9] = {0, 0, 0, 1, 0, 0, 0, 1, 0};
  std::memcpy(binary.data(), vertices, sizeof(vertices));
  const uint16_t indices[3] = {0, 1, 2};
  std::memcpy(binary.data() + 36, indices, sizeof(indices));
  const float uvs[6] = {0, 0, 1, 0, 0, 1};
  std::memcpy(binary.data() + 44, uvs, sizeof(uvs));
  auto json = j.dump();
  while (json.size() % 4)
    json.push_back(' ');
  std::vector<uint8_t> bytes;
  word(bytes, 0x46546c67);
  word(bytes, 2);
  word(bytes, uint32_t(12 + 8 + json.size() + 8 + binary.size()));
  word(bytes, uint32_t(json.size()));
  word(bytes, 0x4e4f534a);
  bytes.insert(bytes.end(), json.begin(), json.end());
  word(bytes, uint32_t(binary.size()));
  word(bytes, 0x004e4942);
  bytes.insert(bytes.end(), binary.begin(), binary.end());
  const auto path = root / (cycle ? "cycle.vrm"
                            : v1  ? "avatar1.vrm"
                                  : "avatar0.vrm");
  std::ofstream file(path, std::ios::binary);
  file.write((const char *)bytes.data(), bytes.size());
  return path;
}
int main(int argc, char **argv) {
  if (argc == 3 && std::strcmp(argv[1], "--make-fixture") == 0) {
    const auto dest = std::filesystem::path(argv[2]);
    const auto path = fixture(dest.parent_path(), true);
    std::filesystem::rename(path, dest);
    std::cout << "VRM_TEST_FIXTURE_READY\n";
    return 0;
  }
  if (argc > 1) {
    Asset live;
    std::string error;
    if (!sight::vrm::load(argv[1], live, error)) {
      std::cerr << "Real VRM rejected: " << error << "\n";
      return 1;
    }
    const auto frame = sight::vrm::deform(live, {});
    size_t vertices = 0, triangles = 0;
    for (size_t i = 0; i < live.surfaces.size(); ++i) {
      assert(frame[i].positions.size() == live.surfaces[i].vertices.size());
      vertices += frame[i].positions.size();
      triangles += live.surfaces[i].triangles.size() / 3;
    }
    std::cout << "REAL_VRM_PARSED version=" << live.vrm_version
              << " surfaces=" << live.surfaces.size()
              << " vertices=" << vertices << " triangles=" << triangles
              << " humanoid=" << live.bones.size()
              << " expressions=" << live.expressions.size() << "\n";
    return vertices > 0 && triangles > 0 ? 0 : 1;
  }
  const auto dir =
      std::filesystem::temp_directory_path() / "sight-vrm-asset-test";
  std::filesystem::create_directories(dir);
  for (bool v1 : {false, true}) {
    Asset model;
    std::string error;
    const auto path = fixture(dir, v1);
    assert(sight::vrm::load(path.string(), model, error) && error.empty());
    assert(model.vrm_version == (v1 ? 1 : 0));
    assert(model.bones.at("head") == 0);
    assert(model.surfaces.size() == 1 &&
           model.surfaces[0].triangles.size() == 3);
    assert(model.surfaces[0].vertices.size() == 3);
    assert(model.images.size() == 1 && model.images[0].rgba.size() == 16);
    assert(model.materials.size() == 1 && model.materials[0].image == 0);
    assert(model.surfaces[0].morph_positions.size() == 1);
    const auto key = v1 ? "aa" : "A";
    assert(model.expressions.contains(key));
    Pose pose;
    pose.expressions[key] = 1.f;
    const auto frame = sight::vrm::deform(model, pose);
    assert(frame.size() == 1 && frame[0].positions.size() == 3);
    assert(std::fabs(frame[0].positions[1].x - 1.75f) < 0.001f);
    std::filesystem::remove(path);
  }
  Asset invalid;
  std::string error;
  const auto cycle_path = fixture(dir, true, true);
  assert(!sight::vrm::load(cycle_path.string(), invalid, error));
  std::filesystem::remove(cycle_path);
  assert(!sight::vrm::load("/does/not/exist.vrm", invalid, error));
  assert(!error.empty());
  std::filesystem::remove(dir);
  std::cout << "PASS: VRM 0.x/1.0 native GLB parses, morphs, rejects missing "
               "asset\\n";
}
