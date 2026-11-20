#pragma once
#include <filesystem>
#include <string>
#include <vector>
namespace sight::live2d {
struct MotionFile { std::string group; std::filesystem::path file, sound; float fade_in=-1, fade_out=-1; };
struct ExpressionFile { std::string name; std::filesystem::path file; };
struct Package {
    std::filesystem::path manifest, moc, physics, pose;
    std::vector<std::filesystem::path> textures;
    std::vector<MotionFile> motions;
    std::vector<ExpressionFile> expressions;
    std::vector<std::string> blink_ids, lip_ids;
};
// Structural intake only; never signifies Cubism Core acceptance.
bool inspect_package(const std::filesystem::path&, Package&, std::string& error);
}
