#pragma once

#include "tasks_runtime_abi.hpp"
#include <filesystem>
#include <string>

namespace sight {

// Find the separately installed, ABI-verified MediaPipe 1.2.3 runtime next
// to our OBS plugin binary. Never fall back to the legacy runtime or search
// arbitrary system paths. Unknown ABI lengths are never auto-selected.
//
// This is safe first-run DISCOVERY, not a cryptographic package signature
// check: the install/package path must still verify SHA-256 separately.
inline std::string holistic_runtime_default(
    const std::filesystem::path& plugin_module_path)
{
    namespace fs=std::filesystem;
    if(plugin_module_path.empty())return {};
    std::error_code ec;
    if(!fs::is_regular_file(plugin_module_path,ec) || ec)return {};
    const auto candidate=plugin_module_path.parent_path()/
                         "libmediapipe-1.2.3.so";
    ec.clear();
    const auto status=fs::symlink_status(candidate,ec);
    if(ec || !fs::is_regular_file(status))return {};
    ec.clear();
    const auto length=fs::file_size(candidate,ec);
    if(ec || tasks_runtime_abi(length)!=TasksRuntimeAbi::V123)return {};
    return candidate.string();
}

// Face, pose and gesture support both tested ABIs. Prefer the modern
// sibling for default settings, but retain compatibility with old installs.
inline std::string tracking_runtime_default(
    const std::filesystem::path& plugin_module_path)
{
    namespace fs=std::filesystem;
    if(auto modern=holistic_runtime_default(plugin_module_path); !modern.empty())
        return modern;
    std::error_code ec;
    if(!fs::is_regular_file(plugin_module_path,ec)||ec)return {};
    const auto candidate=plugin_module_path.parent_path()/"libmediapipe.so";
    const auto status=fs::symlink_status(candidate,ec);
    if(ec||!fs::is_regular_file(status))return {};
    const auto length=fs::file_size(candidate,ec);
    if(ec||tasks_runtime_abi(length)==TasksRuntimeAbi::Unknown)return {};
    return candidate.string();
}

} // namespace sight
