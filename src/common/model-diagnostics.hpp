#pragma once
#include <obs-module.h>
#include "../core/tasks_runtime_abi.hpp"
#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>

namespace sight_diagnostics {
enum class Format { Onnx, Task, Tflite, MediaPipe, MediaPipe123, OptionalMediaPipe };
struct Result { std::string text; obs_text_info_type type; };
inline Result check(std::string_view value, Format format)
{
    const bool runtime = format == Format::MediaPipe || format == Format::MediaPipe123 ||
                         format == Format::OptionalMediaPipe;
    const auto warning = OBS_TEXT_INFO_WARNING, error = OBS_TEXT_INFO_ERROR, normal = OBS_TEXT_INFO_NORMAL;
    if (value.empty()) {
        if (format == Format::OptionalMediaPipe)
            return {"No runtime selected. Required only for the MediaPipe CPU detector.", normal};
        return {runtime ? "No runtime selected. Choose the compatible MediaPipe library."
                        : "No model selected. Choose a compatible local model.", warning};
    }
    namespace fs = std::filesystem;
    std::error_code ec;
    const fs::path path(value);
    const auto status = fs::status(path, ec);
    if (ec == std::errc::no_such_file_or_directory || status.type() == fs::file_type::not_found)
        return {"Selected file not found.", error};
    if (ec) return {"Cannot access the selected file.", error};
    if (!fs::is_regular_file(status)) return {"Selection is not a regular file.", error};
    const auto bytes = fs::file_size(path, ec);
    if (ec) return {"Cannot determine selected file size.", error};
    if (!bytes) return {"Selected file is empty.", error};
    // Read at most eight bytes, never load a model or execute a library here.
    // ONNX protobuf has no reliable fixed magic. Extension alone is not proof.
    std::ifstream stream(path, std::ios::binary);
    if (!stream) return {"Selected file is unreadable.", error};
    char header[8]{};
    stream.read(header, sizeof(header));
    if (stream.bad() || stream.gcount() == 0) return {"Cannot read selected file.", error};
    std::string name = path.filename().string();
    std::transform(name.begin(), name.end(), name.begin(),
                   [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
    const auto ends = [&](std::string_view suffix) {
        return name.size() >= suffix.size() &&
               std::string_view(name).substr(name.size() - suffix.size()) == suffix;
    };
    if (runtime) {
        if (stream.gcount() < 4 || std::string_view(header,4) != "\x7f" "ELF")
            return {"Selected runtime is not an ELF shared library.", error};
        const auto abi = sight::tasks_runtime_abi(bytes);
        if (abi == sight::TasksRuntimeAbi::Unknown)
            return {"Runtime size does not match a supported MediaPipe ABI. See the runtime setup guide.", error};
        if ((format == Format::MediaPipe123 || format == Format::OptionalMediaPipe) &&
            abi != sight::TasksRuntimeAbi::V123)
            return {"This detector requires MediaPipe 1.2.3; the selected runtime matches the legacy size.", error};
        return {"Runtime size and header match the expected family. ABI and loading are not yet verified.", normal};
    }
    if (format == Format::Onnx && !ends(".onnx"))
        return {"Expected an .onnx model. Verify the selected format.", warning};
    if (format == Format::Task && !ends(".task"))
        return {"Expected a MediaPipe .task model. Verify the selected format.", warning};
    if (format == Format::Tflite) {
        if (!ends(".tflite")) return {"Expected a .tflite model. Verify the selected format.", warning};
        if (stream.gcount() < 8 || std::string_view(header+4,4) != "TFL3")
            return {"Missing TFLite identifier (TFL3). File does not look like a TFLite model.", error};
    }
    return {"File is present and readable. Model compatibility is not yet verified; actual loading determines support.", normal};
}

template<Format F>
inline bool changed(obs_properties_t *props, obs_property_t *property, obs_data_t *settings)
{
    const auto key = std::string(obs_property_name(property));
    auto *label = obs_properties_get(props, (key + "_diagnostic").c_str());
    if (!label) return false;
    const auto result = check(obs_data_get_string(settings,key.c_str()), F);
    const bool differs = result.text != obs_property_description(label) ||
                         result.type != obs_property_text_info_type(label);
    if (differs) {
        obs_property_set_description(label,result.text.c_str());
        obs_property_text_set_info_type(label,result.type);
    }
    return differs;
}
template<Format F>
inline void add(obs_properties_t *props, obs_property_t *picker)
{
    const auto key = std::string(obs_property_name(picker)) + "_diagnostic";
    auto *label = obs_properties_add_text(props,key.c_str(),"Model/runtime check",OBS_TEXT_INFO);
    obs_property_text_set_info_word_wrap(label,true);
    obs_property_set_modified_callback(picker,changed<F>);
}
}
