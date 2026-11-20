#pragma once
#include <array>
#include <filesystem>
#include <map>
#include <string>
#include <vector>
namespace sight::live2d {
struct Key {float time=0,value=0;};
struct Segment {int type=0;std::array<Key,4> points{};};
struct Curve {
    std::string id;
    bool part=false;
    Key first;
    std::vector<Segment> segments;
    float sample(float time) const;
};
struct Motion {
    float duration=0;
    std::vector<Curve> curves;
    void sample(double time,std::map<std::string,float>& parameters,std::map<std::string,float>& parts) const;
};
enum class ExpressionBlend {Add,Multiply,Overwrite};
struct ExpressionValue {std::string id;float value;ExpressionBlend blend;};
bool read_motion(const std::filesystem::path&,Motion&,std::string& error);
bool read_expression(const std::filesystem::path&,std::vector<ExpressionValue>&,std::string& error);
}
