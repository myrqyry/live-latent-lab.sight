#pragma once

#include <filesystem>
#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace lll {

enum class LiteRtDevice { Gpu, Cpu };

struct LiteRtRunOptions {
    bool allow_cpu_fallback = true;
};

class LiteRtRuntime {
public:
    LiteRtRuntime();
    ~LiteRtRuntime();
    static LiteRtRuntime &instance();

    bool initialize();
    void shutdown();
    bool ready() const;
    std::string last_error() const;
    size_t compilation_count() const;
    size_t cached_session_count() const;

    bool run(const std::filesystem::path &model_path,
             LiteRtDevice device,
             const std::vector<float> &input,
             std::vector<float> &output,
             const LiteRtRunOptions &options = {});

    bool run_multi(const std::filesystem::path &model_path,
                   LiteRtDevice device,
                   const std::vector<float> &input,
                   std::vector<std::vector<float>> &outputs,
                   const LiteRtRunOptions &options = {});

private:

    bool run_once_multi(const std::filesystem::path &model_path,
                        LiteRtDevice device,
                        const std::vector<float> &input,
                        std::vector<std::vector<float>> &output_values);
    void set_error(std::string error);

    mutable std::mutex mutex_;
    std::unique_ptr<class RuntimeState> state_;
    std::string last_error_;
};

}  // namespace lll
