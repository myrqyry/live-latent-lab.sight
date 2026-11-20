#pragma once
#include "core/tracking_signal_bus.hpp"
#include <map>
#include <string>
#include <string_view>
#include <vector>
namespace sight::live2d {
struct Parameter { std::string id; float minimum, neutral, maximum; };
using Values=std::map<std::string,float>;
std::string_view standard_parameter(std::string_view id);
// Only actual rig parameters supplied by the runtime may be written.
// Explicit lip IDs come from the package, not guessed from numeric order.
Values face_parameters(const TrackingSignal*,const std::vector<Parameter>&,
                       const std::vector<std::string>& lip_ids={});
}
