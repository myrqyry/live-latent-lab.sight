#pragma once

#include <onnxruntime_cxx_api.h>
#include <algorithm>
#include <cstdlib>
#include <memory>
#include <string_view>
#include <stdexcept>

#ifdef SIGHT_ANIMEGAN_ONLY
#include "../common/sight-log.h"
#include <util/base.h>
#endif

namespace lll {
inline std::unique_ptr<Ort::Session> open_onnx_session(
    const Ort::Env &env, const char *path, const Ort::SessionOptions &cpu_options,
    bool prefer_gpu = true, const char **configured_provider = nullptr,
    bool require_cuda = false)
{
    if(configured_provider)*configured_provider=nullptr;
    const char *requested = std::getenv("SIGHT_ONNX_DEVICE");
    const bool cpu_override=requested && std::string_view(requested) == "cpu";
    if(require_cuda && cpu_override)
        throw std::runtime_error("CUDA disabled by SIGHT_ONNX_DEVICE=cpu");
    if(require_cuda && !prefer_gpu)
        throw std::runtime_error("Conflicting CPU and required CUDA requests");
    if (prefer_gpu && !cpu_override) {
        try {
            const auto providers = Ort::GetAvailableProviders();
            if (std::find(providers.begin(), providers.end(), "CUDAExecutionProvider") != providers.end()) {
                auto options = cpu_options.Clone();
                OrtCUDAProviderOptions cuda;
                // Avoid expensive exhaustive tuning as models load or input sizes change.
                cuda.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchHeuristic;
                options.AppendExecutionProvider_CUDA(cuda);
                auto session = std::make_unique<Ort::Session>(env, path, options);
#ifdef SIGHT_ANIMEGAN_ONLY
                sight_logf(LOG_INFO, "[Sight/ONNX] SIGHT_ONNX_CUDA_4BE7: CUDA session ready for %s", path);
#endif
                if(configured_provider)*configured_provider="ONNX Runtime CUDA (CPU partitions possible)";
                return session;
            }
        } catch (const Ort::Exception &error) {
            if(require_cuda)throw;
#ifdef SIGHT_ANIMEGAN_ONLY
            sight_logf(LOG_WARNING, "[Sight/ONNX] CUDA initialization failed; retrying on CPU: %s", error.what());
#endif
        }
    }
    if(require_cuda)throw std::runtime_error("CUDA execution provider unavailable");
    auto session = std::make_unique<Ort::Session>(env, path, cpu_options);
#ifdef SIGHT_ANIMEGAN_ONLY
    sight_logf(LOG_INFO, "[Sight/ONNX] CPU session ready for %s", path);
#endif
    if(configured_provider)*configured_provider="ONNX Runtime CPU";
    return session;
}
}
