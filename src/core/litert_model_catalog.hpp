#pragma once

#include <string>
#include <vector>

namespace lll {

struct LiteRtModelSpec {
    std::string id;
    std::string display_name;
    std::string url;
    std::string sha256;
    std::string filename;
};

const std::vector<LiteRtModelSpec> &litert_model_catalog();
const LiteRtModelSpec *find_litert_model(const std::string &model_id);

}  // namespace lll
