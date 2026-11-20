#pragma once

#include "litert_model_catalog.hpp"

#include <filesystem>
#include <functional>
#include <optional>
#include <string>

namespace lll {

class LiteRtModelManager {
public:
    using Downloader = std::function<bool(const std::string &url,
                                          const std::filesystem::path &destination,
                                          std::string *error)>;

    explicit LiteRtModelManager(std::filesystem::path model_dir,
                                Downloader downloader = {});

    std::optional<std::filesystem::path> install(
        const LiteRtModelSpec &spec, std::string *error);
    std::optional<std::filesystem::path> resolve(
        const std::string &model_id) const;

private:
    std::filesystem::path model_dir_;
    Downloader downloader_;
};

}  // namespace lll
